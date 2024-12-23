from copy import deepcopy
from typing import Dict, List, Tuple
import numpy as np
import json, random
random.seed(86)
from torch.nn import functional as F

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook
from ...util.generate import generate_fast

from ..key import repr_tools
from .key_hparams import KEYHyperParams
from types import MethodType

CONTEXT_TEMPLATES_CACHE = None
NEURONS_CACHE = None
is_llama=None
is_qwen=None
is_bloom=None


def apply_multikey_to_model_batch(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: List[Dict],
    hparams: KEYHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    for MzsRE
    """
    # assert request["source_lang"] != "en" or request["source_lang"] != "zh"
    print("******____apply_multikey_to_model_batch____******")


    if copy:
        model = deepcopy(model)
    assert return_orig_weights == True and keep_original_weight == True, "This method do not change the original model."
    weights_copy = {}

    # starting editing
    layers_modify = hparams.layers
    global is_llama, is_qwen, is_bloom, delta_key, target_init

    
    target_init, kl_distr_init = [], None

    if "llama" in model.config._name_or_path.lower():
        is_llama = True
    if "qwen" in hparams.model_name.lower():
        is_qwen = True
    if "bloom" in hparams.model_name.lower():
        is_bloom = True
    
    intermediate_size = 0
    if is_llama or is_qwen:
        intermediate_size = model.config.intermediate_size
    elif is_bloom:
        intermediate_size = model.config.hidden_size * 4
    
    delta_key = torch.zeros((len(layers_modify), intermediate_size), requires_grad=True, device=f"cuda:{hparams.device}")


    def factory(idx, lookup_idxs):
        def new_forward(self, x):
            global delta_key, target_init
            activation = self.act_fn(self.gate_proj(x))
            
            if type(target_init) == list:
                target_init.append(activation[0, lookup_idxs[0]].detach().clone().unsqueeze(0))
            for uu, vv in enumerate(lookup_idxs):
                activation[uu, vv, :] += delta_key[idx]
            down_proj = self.down_proj(activation * self.up_proj(x))  # 原来模型的返回值
            # print("@@@", activation.size())
            # sum1[idx, :] += activation.sum(dim=(0,1))
            # over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
            return down_proj

        def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
            out = F.dropout(x, p=prob, training=training)
            out = residual + out
            return out
        
        def bloom_forward(self, x, residual):
            global delta_key, target_init
            activation = self.gelu_impl(self.dense_h_to_4h(x))
            print("@@@-modified forward func:", idx, delta_key[idx].size(), delta_key[idx].norm(), x.size())
            if type(target_init) == list:
                target_init.append(activation[0, lookup_idxs[0]].detach().clone().unsqueeze(0))
            for uu, vv in enumerate(lookup_idxs):
                activation[uu, vv, :] += delta_key[idx]
            intermediate_output = self.dense_4h_to_h(activation)
            output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)

            return output
        
        if is_llama or is_qwen:
            return new_forward
        elif is_bloom:
            return bloom_forward


    opt = torch.optim.Adam([delta_key], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    for i, cur_layer in enumerate(layers_modify):
        if is_llama or is_qwen:
            obj = model.model.layers[cur_layer].mlp
        elif is_bloom:
            obj = model.transformer.h[cur_layer].mlp
        weights_copy[cur_layer] = obj.forward

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        print("!!!!!", it)
        opt.zero_grad()

        # Forward propagation
        # train_bsz_size = 2
        # all_logits = []
        # for i_bsz in range(int(input_tok["input_ids"].shape[0]/train_bsz_size)+1):
        #     train_st = i_bsz * train_bsz_size
        #     train_ed = min((i_bsz + 1) * train_bsz_size, input_tok["input_ids"].shape[0])
        #     if train_ed == train_st: break
        
        #     for i, cur_layer in enumerate(layers_modify):
        #         obj = model.model.layers[cur_layer].mlp
        #         weights_copy[cur_layer] = obj.forward
        #         obj.forward = MethodType(factory(i, lookup_idxs[train_st:train_ed]), obj)
            
        #     cur_logits = model(input_tok["input_ids"][train_st:train_ed], 
        #         input_tok["attention_mask"][train_st:train_ed]).logits
        #     all_logits.append(cur_logits)
        #     print("@@@!!!-tmp-logits", cur_logits.size())
        # logits = torch.cat(all_logits, dim=0).cuda()


        for i_lang in request["source_lang"].split("_"):
            # print("!!!!!!!", i_lang)
            if i_lang == "": continue
            cur_request = deepcopy(request[i_lang])
            # print("!!!!!!!-1")
            
            if cur_request['subject'] not in cur_request['prompt']: 
                # print("!!!!!!!-2")
                continue
            if('{}' not in cur_request['prompt']):
                assert cur_request['subject'] in cur_request['prompt'] or \
                    print(f"Subject:{cur_request['subject']} do not exist in prompt: {cur_request['prompt']}")

                cur_request['prompt'] = cur_request['prompt'].replace(cur_request['subject'], '{}', 1)
            # print("!!!!!!!-3")
            # print(
            #     f"Executing KEY algorithm for the update: "
            #     f"[{cur_request['prompt'].format(cur_request['subject'])}] -> [{cur_request['target_new']}]"
            # )

            cur_target_ids = tok(cur_request["target_new"], return_tensors="pt").to(f"cuda:{hparams.device}")[
                "input_ids"
            ][0]
            if "llama" in hparams.model_name.lower():
                cur_target_ids = cur_target_ids[1:]

            # print("@@@-1-target_ids:", cur_target_ids)
            target_ids_size = []
            # Compile list of rewriting and KL x/y pairs
            if i_lang == "en":
                cur_context_templates = ["{}",
                                "The Clean Power. {}",
                                "The following are the. {}",
                                "Therefore, the only. {}",
                                "Because you are reading. {}"]
                cur_kl_prompts = ["{} is a"]
            elif i_lang == "zh":
                cur_context_templates = ["{}",
                                "我知道的， {}",
                                "你怎么认为呢？ {}",
                                "这是正确的说法。 {}",
                                "我确定是对的。 {}"]
                cur_kl_prompts = ["{}是"]
            elif i_lang == "es":
                cur_context_templates = ["{}",
                                'élève de Jean-Baptiste-. {}', 
                                'bienvenue a notre blog. The. {}', 
                                'porque la verdadera vida de la vida. {}', 
                                'Síle is a native of the West. {}']
                cur_kl_prompts = ["{}Sí"]
            elif i_lang == "vi":
                cur_context_templates = ["{}",
                                'Bạn đang ��. {}', 
                                'KHÔNG CẢM. {}', 
                                'Phải chăng m��. {}', 
                                'tayo the Little Bus Coloring P. {}']
                cur_kl_prompts = ["{}Đúng"]
            elif i_lang == "fr":
                cur_context_templates = ["{}",
                                'jefferson county, texas property records. {}', 
                                'toi_api_calls: ". {}', 
                                'il y a des gens que je trouve plus. {}', 
                                'OMSA is proud to announce that. {}']
                cur_kl_prompts = ["{}Oui"]
            elif i_lang == "de":
                cur_context_templates = ["{}",
                                'Jaipur is the capital and the largest. {}', 
                                'Duane D. Lester: 5. {}', 
                                'Rechtsanwalt Stephan Rüter,. {}', 
                                'Kommen Sie und sehen Sie sich das akt. {}']
                cur_kl_prompts = ["{}Ja"]
            elif i_lang == "tr":
                cur_context_templates = ["{}",
                                'BENEFITS OF STAIN. {}', 
                                'OUR FIRM’S COMM. {}', 
                                'HAYIR GÖRÜŞ. {}', 
                                'Sağlam, H.; Çel. {}']
                cur_kl_prompts = ["{}Evet"]
            elif i_lang == "ru":
                cur_context_templates = ["{}",
                                'онлайн-кинотеатр. {}', 
                                'Хотя, как мы видели, в. {}', 
                                'Дата и время: 201. {}', 
                                'Спасибо, Вася! . {}']
                cur_kl_prompts = ["{}да"]
            elif i_lang == "th":
                cur_context_templates = ["{}",
                                'เขาใส่. {}', 
                                'แม้ว่าเป. {}', 
                                'แต่งตัวแ. {}', 
                                'ดังนั้น,. {}']
                cur_kl_prompts = ["{}ใช่"]
            elif i_lang == "du":   # nl 荷兰语
                cur_context_templates = ["{}",
                                'Jijiji, a leading online classified. {}', 
                                'Hijab, Islam, and Muslims in. {}', 
                                'Hoewel dit in de eerste plaats een. {}', 
                                'daarom is de vleugelslag. {}']
                cur_kl_prompts = ["{}Ja"]
            elif i_lang == "pt":
                cur_context_templates = ["{}",
                                'maseru-bhutan.. {}', 
                                'Embora não haja um modelo, o. {}', 
                                'você não tem certeza?. {}', 
                                'Quemar: A New Kind of F. {}']
                cur_kl_prompts = ["{}sim"]
            elif i_lang == "cz":
                cur_context_templates = ["{}",
                                'já que os seus irmãos est. {}', 
                                'že joe biden je prezidentem. {}', 
                                'hahaha. It is not the same. {}', 
                                'tento, ten, teni, tent. {}']
                cur_kl_prompts = ["{}Ano"]
            

            
            cur_rewriting_prompts = [
                context.format(cur_request["prompt"]) + tok.decode(cur_target_ids[:-1])
                for context in cur_context_templates
            ]

            # Compute indices of the tokens where the fact is looked up
            cur_rewrite_lookup_idxs = [
                find_fact_lookup_idx(
                    prompt, cur_request["subject"], tok, hparams.fact_token, verbose=False
                )
                for i, prompt in enumerate(cur_rewriting_prompts)
            ]
            # print("@@@-4-lookup_ids:", cur_rewrite_lookup_idxs)

            cur_kl_lookup_idxs = [
                find_fact_lookup_idx(
                    prompt, cur_request["subject"], tok, hparams.fact_token, verbose=False
                )
                for i, prompt in enumerate(cur_kl_prompts)
            ]
            # print("@@@-5-lookup_ids:", cur_kl_lookup_idxs)
            
            target_ids_size.extend([len(cur_target_ids) for _ in cur_rewriting_prompts])

            all_prompts = cur_rewriting_prompts + cur_kl_prompts
            all_prompts = [_ii.format(cur_request["subject"]) for _ii in all_prompts]
            # print("!!!!!", all_prompts)
            lookup_idxs = cur_rewrite_lookup_idxs + cur_kl_lookup_idxs
            # print("all-lookup_idxs: ", lookup_idxs)
            # print("!!! target_ids_size: ", target_ids_size)
            target_ids_size = torch.tensor(target_ids_size, device=f"cuda:{hparams.device}")
            # print(tok.padding_side)
            
            input_tok = tok(
                all_prompts,
                return_tensors="pt",
                padding=True,
            ).to(f"cuda:{hparams.device}")
            # print("@@@-2-input_tok:", input_tok["input_ids"].size(), input_tok["input_ids"][0],  input_tok["input_ids"][-1])

            # Compute rewriting targets
            rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
                len(cur_rewriting_prompts), *input_tok["input_ids"].shape[1:]
            )
            # print("@@@-3-rewriting_targets:", rewriting_targets.size(), rewriting_targets[0])

            for i in range(len(cur_rewriting_prompts)):
                ex_len = input_tok["attention_mask"][i].sum()
                rewriting_targets[i, ex_len - len(cur_target_ids) : ex_len] = cur_target_ids

            # print("@@@-4-rewriting_targets:", rewriting_targets.size(), rewriting_targets[0], rewriting_targets[-1])

            for i, cur_layer in enumerate(layers_modify):
                if is_llama or is_qwen:
                    obj = model.model.layers[cur_layer].mlp
                elif is_bloom:
                    obj = model.transformer.h[cur_layer].mlp
                obj.forward = MethodType(factory(i, lookup_idxs), obj)

            # model forward
            # print("@@@!!!-tmp", input_tok["input_ids"].size())
            logits = model(**input_tok).logits 
            # print("@@@-logits:", logits.size())
            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(cur_kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(cur_kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

            # Compute loss on rewriting targets
            log_probs = torch.log_softmax(logits, dim=2)

            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
            ).squeeze(2)
            # print("@@@@-5-loss:", loss.size())
            mask = (rewriting_targets != -100).float()

            # Aggregate total losses
            nll_loss_each = -(loss * mask).sum(1) / target_ids_size
            nll_loss = nll_loss_each.mean()
            kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
                kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
            )
            if type(target_init) == list:
                target_init = torch.cat(target_init, dim=0)
        
            weight_decay = hparams.v_weight_decay * (
                torch.norm(delta_key) / torch.norm(target_init) ** 2
            )
            # weight_decay = hparams.v_weight_decay * torch.norm(delta_key) ** 2
            loss = nll_loss + kl_loss + weight_decay
            print(
                f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                f"avg prob of [{cur_request['target_new']}] "
                f"{torch.exp(-nll_loss_each).mean().item()}"
            )
            if loss < 5e-2:
                break

            if it == hparams.v_num_grad_steps - 1:
                break

            # Backpropagate
            loss.backward()

        opt.step()
        
        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta_key.norm() > max_norm:
            with torch.no_grad():
                delta_key[...] = delta_key * max_norm / delta_key.norm()


    for i, cur_layer in enumerate(layers_modify):
        if is_llama or is_qwen:
            obj = model.model.layers[cur_layer].mlp
        elif is_bloom:
            obj = model.transformer.h[cur_layer].mlp
        obj.forward = weights_copy[cur_layer]

    delta_key.requires_grad = False
    return delta_key, target_init


def apply_multikey_to_model_v4(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: List[Dict],
    hparams: KEYHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    LU-LAFNs
    """
    # assert request["source_lang"] != "en" or request["source_lang"] != "zh"
    
    print("******____apply_multikey_to_model_v4____******")

    rewriting_prompts = []
    rewriting_number = [0]
    kl_prompts = []

    target_ids = []
    target_ids_size = []
    all_targets = []

    rewriting_look_up_idxs = []
    kl_look_up_idxs = []

    for i_lang in request["source_lang"].split("_"):
        if i_lang == "": continue
        cur_request = request[i_lang]
        all_targets.append(cur_request["target_new"])
    
        if('{}' not in cur_request['prompt']):
            assert cur_request['subject'] in cur_request['prompt'] or \
                print(f"Subject:{cur_request['subject']} do not exist in prompt: {cur_request['prompt']}")

            cur_request['prompt'] = cur_request['prompt'].replace(cur_request['subject'], '{}')

        print(
            f"Executing KEY algorithm for the update: "
            f"[{cur_request['prompt'].format(cur_request['subject'])}] -> [{cur_request['target_new']}]"
        )

        cur_target_ids = tok(cur_request["target_new"], return_tensors="pt").to(f"cuda:{hparams.device}")[
            "input_ids"
        ][0]
        if "llama" in hparams.model_name.lower():
            cur_target_ids = cur_target_ids[1:]

        target_ids.append(cur_target_ids)

        print("@@@-1-target_ids:", cur_target_ids)

        # Compile list of rewriting and KL x/y pairs
        if i_lang == "en":
            cur_context_templates = ["{}",
                            "The Clean Power. {}",
                            "The following are the. {}",
                            "Therefore, the only. {}",
                            "Because you are reading. {}"]
            cur_kl_prompts = ["{} is a"]
        else:
            cur_context_templates = ["{}",
                            "我知道的， {}",
                            "你怎么认为呢？ {}",
                            "这是正确的说法。 {}",
                            "我确定是对的。 {}"]
            cur_kl_prompts = ["{}是"]
        # fixed for a fast and fair comparison
        
        cur_rewriting_prompts = [
            context.format(cur_request["prompt"]) + tok.decode(cur_target_ids[:-1])
            for context in cur_context_templates
        ]

        # Compute indices of the tokens where the fact is looked up
        cur_rewrite_lookup_idxs = [
            find_fact_lookup_idx(
                prompt, cur_request["subject"], tok, hparams.fact_token, verbose=(i == 0)
            )
            for i, prompt in enumerate(cur_rewriting_prompts)
        ]
        print("@@@-4-lookup_ids:", cur_rewrite_lookup_idxs)
        rewriting_look_up_idxs.extend(cur_rewrite_lookup_idxs)

        cur_kl_lookup_idxs = [
            find_fact_lookup_idx(
                prompt, cur_request["subject"], tok, hparams.fact_token, verbose=(i == 0)
            )
            for i, prompt in enumerate(cur_kl_prompts)
        ]
        print("@@@-5-lookup_ids:", cur_kl_lookup_idxs)
        kl_look_up_idxs.extend(cur_kl_lookup_idxs)

        rewriting_prompts.extend([_i.format(cur_request["subject"]) for _i in cur_rewriting_prompts])
        target_ids_size.extend([len(cur_target_ids) for _ in cur_rewriting_prompts])
        kl_prompts.extend([_i.format(cur_request["subject"]) for _i in cur_kl_prompts])
        rewriting_number.append(len(rewriting_prompts))

    
    all_prompts = rewriting_prompts + kl_prompts
    print("!!!!!", all_prompts)
    lookup_idxs = rewriting_look_up_idxs + kl_look_up_idxs
    print("all-lookup_idxs: ", lookup_idxs)
    print("!!! target_ids_size: ", target_ids_size)
    target_ids_size = torch.tensor(target_ids_size, device=f"cuda:{hparams.device}")
    print(tok.padding_side)
    
    input_tok = tok(
        all_prompts,
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{hparams.device}")
    print("@@@-2-input_tok:", input_tok["input_ids"].size(), input_tok["input_ids"][0],  input_tok["input_ids"][-1])

    # Compute rewriting targets
    assert rewriting_number[-1] == len(rewriting_prompts)
    rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    print("@@@-3-rewriting_targets:", rewriting_targets.size(), rewriting_targets[0])

    for i, i_num in enumerate(rewriting_number):
        if i_num == 0: continue
        st = rewriting_number[i-1]
        ed = rewriting_number[i]
        print("!!! ", st, ed)
        for i_r in range(st, ed):
            ex_len = input_tok["attention_mask"][i_r].sum()
            rewriting_targets[i_r, ex_len - len(target_ids[i-1]) : ex_len] = target_ids[i-1]
    print("@@@-4-rewriting_targets:", rewriting_targets.size(), rewriting_targets[0], rewriting_targets[-1])


    if copy:
        model = deepcopy(model)
    assert return_orig_weights == True and keep_original_weight == True, "This method do not change the original model."
    weights_copy = {}

    # starting editing
    layers_modify = hparams.layers
    global is_llama, is_qwen, is_bloom, delta_key, target_init

    if "llama" in hparams.model_name.lower():
        is_llama = True
    if "qwen" in hparams.model_name.lower():
        is_qwen = True
    if "bloom" in hparams.model_name.lower():
        is_bloom = True
    
    intermediate_size = 0
    if is_llama or is_qwen:
        intermediate_size = model.config.intermediate_size
    elif is_bloom:
        intermediate_size = model.config.hidden_size * 4

    print("@@@@@-case_id:", kwargs["case_id"])
    global NEURONS_CACHE
    NEURONS_CACHE = None
    if NEURONS_CACHE == None:
        assert kwargs["neurons_path"] != "" and hparams.alg_name == "MKEYPG"
        NEURONS_CACHE = json.load(open(kwargs["neurons_path"]))
    neurons_mask = []
    for i_layer in layers_modify:
        tmp_tensor = torch.zeros(intermediate_size, dtype=torch.int)
        tmp_list = torch.tensor(NEURONS_CACHE[str(kwargs["case_id"])][str(i_layer)])
        tmp_tensor.index_fill_(0, tmp_list, 1)
        tmp_mask = tmp_tensor.eq(1)
        print("@@@@@-mask", i_layer, kwargs["case_id"], tmp_mask.size(), tmp_mask.sum(), len(NEURONS_CACHE[str(kwargs["case_id"])][str(i_layer)]))
        neurons_mask.append(tmp_mask.unsqueeze(0))
    neurons_mask = torch.cat(neurons_mask, dim=0).cuda()
       

    delta_key = torch.zeros((len(layers_modify), intermediate_size), requires_grad=True, device=f"cuda:{hparams.device}")
    
    assert delta_key.size() == neurons_mask.size()
    target_init, kl_distr_init = [], None


    def factory(idx, lookup_idxs):
        def new_forward(self, x):
            global delta_key, target_init
            activation = self.act_fn(self.gate_proj(x))
            print("@@@-modified forward func:", idx, delta_key[idx].size(), delta_key[idx].norm(), x.size())
            if type(target_init) == list:
                target_init.append(activation[0, lookup_idxs[0]].detach().clone().unsqueeze(0))
            for uu, vv in enumerate(lookup_idxs):
                activation[uu, vv, :] += delta_key[idx]
            down_proj = self.down_proj(activation * self.up_proj(x))  # 原来模型的返回值
            # print("@@@", activation.size())
            # sum1[idx, :] += activation.sum(dim=(0,1))
            # over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
            return down_proj
        
        def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
            out = F.dropout(x, p=prob, training=training)
            out = residual + out
            return out
        
        def bloom_forward(self, x, residual):
            global delta_key, target_init
            activation = self.gelu_impl(self.dense_h_to_4h(x))
            print("@@@-modified forward func:", idx, delta_key[idx].size(), delta_key[idx].norm(), x.size())
            if type(target_init) == list:
                target_init.append(activation[0, lookup_idxs[0]].detach().clone().unsqueeze(0))
            for uu, vv in enumerate(lookup_idxs):
                activation[uu, vv, :] += delta_key[idx]
            intermediate_output = self.dense_4h_to_h(activation)
            output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)

            return output
        
        if is_llama or is_qwen:
            return new_forward
        elif is_bloom:
            return bloom_forward


    for i, cur_layer in enumerate(layers_modify):
        if is_llama or is_qwen:
            obj = model.model.layers[cur_layer].mlp
        elif is_bloom:
            obj = model.transformer.h[cur_layer].mlp
        weights_copy[cur_layer] = obj.forward
        obj.forward = MethodType(factory(i, lookup_idxs), obj)

    opt = torch.optim.Adam([delta_key], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        
        logits = model(**input_tok).logits

        print("@@@-logits:", logits.size())
        # Compute distribution for KL divergence
        kl_logits = torch.stack(
            [
                logits[i - len(kl_prompts), idx, :]
                for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
            ],
            dim=0,
        )
        kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
        if kl_distr_init is None:
            kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        print("@@@@-5-loss:", loss.size())
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids_size
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        if type(target_init) == list:
            target_init = torch.cat(target_init, dim=0)
        

        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta_key) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta_key) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{all_targets}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        with torch.no_grad():
            delta_key[...] = delta_key * neurons_mask
        
        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta_key.norm() > max_norm:
            with torch.no_grad():
                delta_key[...] = delta_key * max_norm / delta_key.norm()

    # 恢复模型
    for i, cur_layer in enumerate(layers_modify):
        if is_llama or is_qwen:
            obj = model.model.layers[cur_layer].mlp
        elif is_bloom:
            obj = model.transformer.h[cur_layer].mlp
        obj.forward = weights_copy[cur_layer]

    delta_key.requires_grad = False
    return delta_key, target_init


def apply_multivalue_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: List[Dict],
    hparams: KEYHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, List[str]]:


    # assert request["source_lang"] != "en" or request["source_lang"] != "zh"
    
    rewriting_prompts = []
    rewriting_number = [0]
    kl_prompts = []

    target_ids = []
    target_ids_size = []
    all_targets = []

    rewriting_look_up_idxs = []
    kl_look_up_idxs = []


    for i_lang in request["source_lang"].split("_"):
        if i_lang == "": continue
        cur_request = request[i_lang]
        all_targets.append(cur_request["target_new"])
    
        if('{}' not in cur_request['prompt']):
            assert cur_request['subject'] in cur_request['prompt'] or \
                print(f"Subject:{cur_request['subject']} do not exist in prompt: {cur_request['prompt']}")

            cur_request['prompt'] = cur_request['prompt'].replace(cur_request['subject'], '{}')

        print(
            f"Executing KEY algorithm for the update: "
            f"[{cur_request['prompt'].format(cur_request['subject'])}] -> [{cur_request['target_new']}]"
        )

        cur_target_ids = tok(cur_request["target_new"], return_tensors="pt").to(f"cuda:{hparams.device}")[
            "input_ids"
        ][0]
        target_ids.append(cur_target_ids)

        print("@@@-1-target_ids:", cur_target_ids)

        # Compile list of rewriting and KL x/y pairs
        if i_lang == "en":
            cur_context_templates = ["{}",
                            "The Clean Power. {}",
                            "The following are the. {}",
                            "Therefore, the only. {}",
                            "Because you are reading. {}"]
            cur_kl_prompts = ["{} is a"]
        else:
            cur_context_templates = ["{}",
                            "我知道的， {}",
                            "你怎么认为呢？ {}",
                            "这是正确的说法。 {}",
                            "我确定是对的。 {}"]
            cur_kl_prompts = ["{}是"]

        
        cur_rewriting_prompts = [
            context.format(cur_request["prompt"]) + tok.decode(cur_target_ids[:-1])
            for context in cur_context_templates
        ]

        # Compute indices of the tokens where the fact is looked up
        cur_rewrite_lookup_idxs = [
            find_fact_lookup_idx(
                prompt, cur_request["subject"], tok, hparams.fact_token, verbose=(i == 0)
            )
            for i, prompt in enumerate(cur_rewriting_prompts)
        ]
        print("@@@-4-lookup_ids:", cur_rewrite_lookup_idxs)
        rewriting_look_up_idxs.extend(cur_rewrite_lookup_idxs)

        cur_kl_lookup_idxs = [
            find_fact_lookup_idx(
                prompt, cur_request["subject"], tok, hparams.fact_token, verbose=(i == 0)
            )
            for i, prompt in enumerate(cur_kl_prompts)
        ]
        print("@@@-5-lookup_ids:", cur_kl_lookup_idxs)
        kl_look_up_idxs.extend(cur_kl_lookup_idxs)

        rewriting_prompts.extend([_i.format(cur_request["subject"]) for _i in cur_rewriting_prompts])
        target_ids_size.extend([len(cur_target_ids) for _ in cur_rewriting_prompts])
        kl_prompts.extend([_i.format(cur_request["subject"]) for _i in cur_kl_prompts])
        rewriting_number.append(len(rewriting_prompts))

    
    all_prompts = rewriting_prompts + kl_prompts
    print("!!!!!", all_prompts)
    lookup_idxs = rewriting_look_up_idxs + kl_look_up_idxs
    print("all-lookup_idxs: ", lookup_idxs)
    print("!!! target_ids_size: ", target_ids_size)
    target_ids_size = torch.tensor(target_ids_size, device=f"cuda:{hparams.device}")
    print(tok.padding_side)
    
    input_tok = tok(
        all_prompts,
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{hparams.device}")
    print("@@@-2-input_tok:", input_tok["input_ids"].size(), input_tok["input_ids"][0],  input_tok["input_ids"][-1])

    # Compute rewriting targets
    assert rewriting_number[-1] == len(rewriting_prompts)
    rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    print("@@@-3-rewriting_targets:", rewriting_targets.size(), rewriting_targets[0])

    for i, i_num in enumerate(rewriting_number):
        if i_num == 0: continue
        st = rewriting_number[i-1]
        ed = rewriting_number[i]
        print("!!! ", st, ed)
        for i_r in range(st, ed):
            ex_len = input_tok["attention_mask"][i_r].sum()
            rewriting_targets[i_r, ex_len - len(target_ids[i-1]) : ex_len] = target_ids[i-1]
    print("@@@-4-rewriting_targets:", rewriting_targets.size(), rewriting_targets[0], rewriting_targets[-1])


    if copy:
        model = deepcopy(model)
    assert return_orig_weights == True and keep_original_weight == True, "This method do not change the original model."
    weights_copy = {}

    # starting editing
    layers_modify = hparams.layers
    global is_llama, delta_key, target_init


    delta_key = torch.zeros((len(layers_modify), model.config.hidden_size), requires_grad=True, device=f"cuda:{hparams.device}")
    target_init, kl_distr_init = [], None

    def factory(idx, lookup_idxs):
        def llama_forward(self, x):
            global delta_key, target_init
            activation = self.act_fn(self.gate_proj(x))

            print("@@@-modified forward func:", idx, delta_key[idx].size(), delta_key[idx].norm(), x.size())            
            down_proj = self.down_proj(activation * self.up_proj(x))  
            if type(target_init) == list:
                target_init.append(down_proj[0, lookup_idxs[0]].detach().clone().unsqueeze(0))
            for uu, vv in enumerate(lookup_idxs):
                down_proj[uu, vv, :] += delta_key[idx]

            return down_proj


        return llama_forward


    for i, cur_layer in enumerate(layers_modify):
        obj = model.model.layers[cur_layer].mlp
        weights_copy[cur_layer] = obj.forward
        obj.forward = MethodType(factory(i, lookup_idxs), obj)

    opt = torch.optim.Adam([delta_key], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        
        logits = model(**input_tok).logits

        print("@@@-logits:", logits.size())
        # Compute distribution for KL divergence
        kl_logits = torch.stack(
            [
                logits[i - len(kl_prompts), idx, :]
                for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
            ],
            dim=0,
        )
        kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
        if kl_distr_init is None:
            kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        print("@@@@-5-loss:", loss.size())
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids_size
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        if type(target_init) == list:
            target_init = torch.cat(target_init, dim=0)
            # target_init.requires_grad = False
            # target_init_mask = target_init.gt(0)
            # print("!!!!!!target-init", target_init.size(), target_init_mask.sum())
            print("!!!!!!target-init", target_init.size())

        # assert delta_key.size() == target_init.size()
        # with torch.no_grad():
        #     delta_key = delta_key * target_init_mask

        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta_key) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta_key) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{all_targets}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta_key.norm() > max_norm:
            with torch.no_grad():
                delta_key[...] = delta_key * max_norm / delta_key.norm()

    for i, cur_layer in enumerate(layers_modify):
        obj = model.model.layers[cur_layer].mlp
        obj.forward = weights_copy[cur_layer]

    delta_key.requires_grad = False
    return delta_key, target_init



def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        # print("error-!!!", prompt, subject)
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret

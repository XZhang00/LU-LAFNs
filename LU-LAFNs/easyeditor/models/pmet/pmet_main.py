import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.layer_stats import layer_stats
from ...util import nethook
from ...util.generate import generate_fast
from ...util.globals import *

from .compute_ks import compute_ks, compute_ks_parallel
from .compute_zs import compute_zs, compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .pmet_hparams import PMETHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
CONTEXT_TEMPLATES_CACHE = {'es': [['{}'], ['élève de Jean-Baptiste-. {}', 'bienvenue a notre blog. The. {}', 'porque la verdadera vida de la vida. {}', 'A pesar de la apariencia, la g. {}', 'Síle is a native of the West. {}']], 
                           'fr': [['{}'], ['droite\n* 22 mars : le. {}', 'jefferson county, texas property records. {}', 'toi_api_calls: ". {}', 'il y a des gens que je trouve plus. {}', 'OMSA is proud to announce that. {}']], 
                           'vi': [['{}'], ['Bạn đang ��. {}', 'cô ấy 101\n. {}', 'KHÔNG CẢM. {}', 'Phải chăng m��. {}', 'tayo the Little Bus Coloring P. {}']], 
                           'de': [['{}'], ['Jaipur is the capital and the largest. {}', 'Duane D. Lester: 5. {}', 'Rechtsanwalt Stephan Rüter,. {}', 'gehenna.org\nA site to. {}', 'Kommen Sie und sehen Sie sich das akt. {}']], 
                           'tr': [['{}'], ['elmwood park il\nWe are a. {}', 'BENEFITS OF STAIN. {}', 'OUR FIRM’S COMM. {}', 'HAYIR GÖRÜŞ. {}', 'Sağlam, H.; Çel. {}']], 
                           'ru': [['{}'], ['Приходить до причастя\n* . {}', 'онлайн-кинотеатр. {}', 'Хотя, как мы видели, в. {}', 'Дата и время: 201. {}', 'Спасибо, Вася! . {}']], 
                           'zh': [['{}'], ['我们的父��. {}', '他们拿了��. {}', '请在此页面上��. {}', '你好,你好,你好. {}', '这是個關. {}']], 
                           'en': [['{}'], ['The 50th edition of the International. {}', 'Therefore, the question of whether or not a. {}', 'Because I was a child of divorce,. {}', 'I’m a bit behind in my blog. {}', "You're here: oChristian.. {}"]], 
                           'th': [['{}'], ['เขาใส่. {}', 'แม้ว่าเป. {}', 'แต่งตัวแ. {}', 'ดังนั้น,. {}', 'ฉันใ. {}']], 
                           'du': [['{}'], ['Jijiji, a leading online classified. {}', 'Hijab, Islam, and Muslims in. {}', 'rechtswissenschaften.de/\nRe. {}', 'Hoewel dit in de eerste plaats een. {}', 'daarom is de vleugelslag. {}']], 
                           'pt': [['{}'], ['portanto, se você tiver um. {}', 'maseru-bhutan.. {}', 'Embora não haja um modelo, o. {}', 'você não tem certeza?. {}', 'Quemar: A New Kind of F. {}']], 
                           'cz': [['{}'], ['vyshnoboy\nI was thinking. {}', 'já que os seus irmãos est. {}', 'že joe biden je prezidentem. {}', 'hahaha. It is not the same. {}', 'tento, ten, teni, tent. {}']]}
COV_CACHE = {}
KZ_CACHE= {}


def apply_pmet_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: PMETHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    
    deltas = execute_pmet(model, tok, requests, hparams, cache_template=cache_template) #存储了Equ14 左右的值

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items(): #w_name, adj_k, resid
            upd_matrix = upd_matrix.to("cuda:0")
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix.float() #w[...]高级索引，表示对w中每个元素进行操作

    print(f"\nNew weights successfully inserted into {list(deltas.keys())}")
    if not keep_original_weight:
        weights_copy = {}
    return model, weights_copy


def execute_pmet(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: PMETHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Update target and print info
    ori_requests = deepcopy(requests)
    all_requests = []
    all_langs = set()
    
    for i, request in enumerate(ori_requests):
        # if request["target_new"][0] != " ":
        #     # Space required for correct tokenization
        #     requests[i]["target_new"] = " " + request["target_new"]
        if "_" not in request["source_lang"]:
            tmp_req = request[request["source_lang"]]
            all_langs.add(request["source_lang"])
            tmp_req["lang"] = request["source_lang"]
            assert tmp_req['subject'] in tmp_req['prompt']
            tmp_req['prompt'] = tmp_req['prompt'].replace(tmp_req['subject'], '{}', 1)
            all_requests.append(tmp_req)
        else:
            for i_lang in request["source_lang"].split("_"):
                all_langs.add(i_lang)
                tmp_req = request[i_lang]
                tmp_req["lang"] = i_lang
                if tmp_req['subject'] not in tmp_req['prompt']: continue
                tmp_req['prompt'] = tmp_req['prompt'].replace(tmp_req['subject'], '{}', 1)
                all_requests.append(tmp_req)
        
    print("tmp@@@-1", all_langs, all_requests)

    for request in all_requests[:10]:
        print(
            f"PMET request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter( # transformer.h.{}.attn.out_proj
            model, f"{rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
        for rewrite_module_tmp in hparams.rewrite_module_tmps
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    rewrite_module_names = hparams.rewrite_module_tmps

    # Compute z for final layer
    context_templates = get_context_templates(model, tok, list(all_langs))
    z_layer = hparams.layers[-1]
    z_list = dict()
    for rewrite_module_name in rewrite_module_names:
        z_list[rewrite_module_name] = []
    # get zs
    for request in all_requests:
        # Retrieve k/v pair if already stored in cache
        for rewrite_module_name in rewrite_module_names:
            block_name = "attn" if "attn" in rewrite_module_name else "mlp"
            cache_fname = (
                Path(
                    str(cache_template).format(
                        z_layer, block_name, hparams.clamp_norm_factor, request["case_id"]
                    )
                )
                if cache_template is not None
                else None
            )
            data_loaded = False
            if (
                cache_fname is not None  # Require cache template
                and cache_fname.exists()  # Cache file must exist
            ):
                try:
                    data = np.load(cache_fname)
                    z_list[rewrite_module_name].append(torch.from_numpy(data["v_star"]).to("cuda:0"))
                    data_loaded = True
                except Exception as e:
                    print(f"Error reading cache file due to {e}. Recomputing...")

            # Compute k/v pair if not loaded from cache
            if not data_loaded:
                if len(rewrite_module_names) == 2:
                    cur_z_attn, cur_z_mlp = compute_zs( 
                            model,
                            tok,
                            request,
                            hparams,
                            z_layer,
                            context_templates[request["lang"]],
                    )
                    z_list[rewrite_module_names[0]].append(cur_z_attn if "attn" in rewrite_module_names[0] else cur_z_mlp)
                    z_list[rewrite_module_names[1]].append(cur_z_attn if "attn" in rewrite_module_names[1] else cur_z_mlp)
                    for rewrite_module_name in rewrite_module_names:
                        block_name = "attn" if "attn" in rewrite_module_name else "mlp"
                        cache_fname = (
                            Path(
                                str(cache_template).format(
                                    z_layer, block_name, hparams.clamp_norm_factor, request["case_id"]
                                )
                            )
                            if cache_template is not None
                            else None
                        )
                        if cache_fname is not None:
                            cache_fname.parent.mkdir(exist_ok=True, parents=True)
                            if block_name == "attn":
                                np.savez(
                                    cache_fname,
                                    **{
                                        "v_star": cur_z_attn.detach().cpu().numpy(),
                                    },
                                )
                            else:
                                np.savez(
                                    cache_fname,
                                    **{
                                        "v_star": cur_z_mlp.detach().cpu().numpy(),
                                    },
                                )
                            print(f"Cached k/v pair at {cache_fname}")
                else:
                    cur_z_attn, cur_z_mlp = compute_zs( 
                    model,
                    tok,
                    request,
                    hparams,
                    z_layer,
                    context_templates[request["lang"]],
                )
                    if "attn" == block_name:
                        cur_z = cur_z_attn
                    else:
                        cur_z = cur_z_mlp
                    z_list[rewrite_module_name].append(cur_z)
                    if cache_fname is not None:
                        cache_fname.parent.mkdir(exist_ok=True, parents=True)
                        np.savez(
                            cache_fname,
                            **{
                                "v_star": cur_z.detach().cpu().numpy(),
                            },
                        )
                        print(f"Cached k/v pair at {cache_fname}")
                break

    for k, v in z_list.items():
        z_list[k] = torch.stack(v, dim=1)
        print("tmp@@@-3", len(v), z_list[k].size())
    # Insert
    torch.cuda.empty_cache()
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n") 
        layers_ks = None
        # force_recompute = layer != hparams.layers[0]
        for rewrite_module_name in rewrite_module_names:
            # Get current model activations

            if 'gpt-j' in model.config._name_or_path and len(rewrite_module_names) == 2:
                if layers_ks == None:
                    layers_ks = compute_ks_parallel(model, tok, all_requests, hparams, layer, context_templates)  #K eqn 19
            else:
                layers_ks = compute_ks(model, tok, all_requests, hparams, rewrite_module_name, layer, context_templates)

            print(f"Writing {layers_ks[rewrite_module_name].size(0)} key/value pair(s) into layers")
            cur_zs = get_module_input_output_at_words( # hidden states eqn 2
                model,
                tok,
                z_layer,
                context_templates=[request["prompt"] for request in all_requests],
                words=[request["subject"] for request in all_requests],
                module_template=rewrite_module_name,
                fact_token_strategy=hparams.fact_token,
            )[1].T
            targets = z_list[rewrite_module_name]  - cur_zs #z_i - h_i^L
            if "qwen" not in model.config._name_or_path.lower():
                layer_ks, targets = (
                    layers_ks[rewrite_module_name].T.double(),
                    targets.double()
                )
            else:
                print("&&&&&&&&&&&&&&&&&& to cuda1!")
                layer_ks, targets = (
                    layers_ks[rewrite_module_name].T.double().to("cuda:1"),
                    targets.double().to("cuda:1")
                )
            # Load covariance matrix
            force_recompute = False
            # force_recompute = layer != hparams.layers[0]
            cov = get_cov(
                model,
                tok,
                rewrite_module_name.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples
                if not force_recompute
                else hparams.mom2_n_samples // 10,
                hparams.mom2_dtype,
                force_recompute=force_recompute,
                hparams=hparams
            )

            repeat_factor = (layer_ks.size(1) // targets.size(1))
            targets = targets.repeat_interleave(repeat_factor, dim=1) #r
            # tmp_res = torch.linalg.solve(
            #     layer_ks @ layer_ks.T + hparams.mom2_update_weight * cov.double(),
            #     layer_ks
            # )
            # upd_matrix =  (targets / np.sqrt((len(hparams.layers) - i ))) @ tmp_res.T
            print("tmp@@@-2", targets.size(), layer_ks.size(), targets.dtype, layer_ks.dtype)
            upd_matrix =  (targets / np.sqrt((len(hparams.layers) - i ))) @ layer_ks.T @ torch.inverse(layer_ks @ layer_ks.T + 
                                             hparams.mom2_update_weight * cov.double())
            print("tmp@@@-2", targets.size(), layer_ks.size(), upd_matrix.size())
            weight_name = f"{rewrite_module_name.format(layer)}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            print(weight_name, ":\norig norm", torch.linalg.norm(weights[weight_name]))
            print("upd norm", torch.linalg.norm(upd_matrix))

            # Update model weights and record desired changes in `delta` variable
            with torch.no_grad():
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float().to("cuda:0")
                deltas[weight_name] = upd_matrix

            # Clear GPU memory

            for x in [layer_ks, cur_zs, targets]:
                x.cpu()
                del x
            torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, _ in weights.items():
            nethook.get_parameter(model, k)[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )
def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
    hparams=None,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats( # download
            model,
            tok,
            layer_name,
            hparams.stats_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    if "qwen" not in model.config._name_or_path.lower():
        return (
            torch.inverse(COV_CACHE[key].to("cuda:0")) if inv else COV_CACHE[key].to("cuda:0")
        )
    else:
        return (
            torch.inverse(COV_CACHE[key].to("cuda:1")) if inv else COV_CACHE[key].to("cuda:1")
        )


def get_context_templates(model, tok, langs=None):
    global CONTEXT_TEMPLATES_CACHE
    lang_prefix_dict = {
        "en": ["The", "Therefore", "Because", "I", "You"],
        "zh": ["我", "他", "请", "你", "这"],
        "cz": ["vy", "já", "že jo", "ha", "tento"],
        "vi": ["Bạn", "cô ấy", "KHÔNG", "Phải", "tay"],
        "tr": ["el", "BEN", "O", "HAYIR", "Sağ"],
        "fr": ["droite", "je", "toi", "il", "OMS"],
        "es": ["él", "bien", "porque", "A pesar de", "Sí"],
        "de": ["Ja", "Du", "Rechts", "gehen", "Kommen"],
        "ru": ["Приходить", "он", "Хотя", "Да", "Спасибо"],
        "du": ["Jij", "Hij", "rechts", "Hoewel", "daarom"],
        "pt": ["portanto", "mas", "Embora", "você", "Quem"],
        "th": ["เขา", "แม้ว่า", "แต่", "ดังนั้น", "ฉัน"],
    }

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = {}
        if langs == None:
            tmp_cache = [["{}"]] + [
                [
                    f.replace("{", " ").replace("}", " ") + ". {}"
                    for f in generate_fast(
                        model,
                        tok,
                        ["The", "Therefore", "Because", "I", "You"],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                    )
                ]
                for length, n_gen in [(10, 5)]  # Be careful about changing this.
            ]
            CONTEXT_TEMPLATES_CACHE["en"] = tmp_cache
            
        else:
            for i_lang in langs:
                tmp_cache = [["{}"]] + [
                [
                    f.replace("{", " ").replace("}", " ") + ". {}"
                    for f in generate_fast(
                        model,
                        tok,
                        lang_prefix_dict[i_lang],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                    )
                ]
                for length, n_gen in [(10, 5)]  # Be careful about changing this.
                ]
                CONTEXT_TEMPLATES_CACHE[i_lang] = []
                for tmp in tmp_cache:
                    if tmp.count("{") > 1 or tmp.count("}") > 1: continue
                    CONTEXT_TEMPLATES_CACHE[i_lang].append(tmp)
    print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")
    return CONTEXT_TEMPLATES_CACHE


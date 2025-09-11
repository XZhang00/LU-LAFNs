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

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .memit_hparams import MEMITHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
# CONTEXT_TEMPLATES_CACHE = {'zh': [['{}'], ['我爱阅读。  猴妈妈教. {}', '他以笔为武器,坚持革命斗争,. {}', '请根据下面的歌词,写出歌词的上. {}', '你是一个在线问诊机器医生,请根据患者. {}', '这幅图主要反映了我国的( )\n. {}']], 'en': [['{}'], ['The number of goals scored in a game against Bar. {}', 'Therefore, the number of samples that are used is. {}', 'Because of budget cuts one IT company established new non. {}', 'I know that $\\cos \\theta $ normally has. {}', 'You are an AI assistant. User will you give. {}']]}
COV_CACHE = {}


def apply_memit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
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

    deltas = execute_memit(model, tok, requests, hparams, cache_template=cache_template)

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to(f"cuda:{hparams.device}"), val_mat.to(f"cuda:{hparams.device}")
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")


    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def execute_memit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
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
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok, list(all_langs))
    z_layer = hparams.layers[-1]
    z_list = []

    for request in all_requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
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
                z_list.append(torch.from_numpy(data["v_star"]).to(f"cuda:{hparams.device}"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates[request["lang"]],
            )

            z_list.append(cur_z)
            # print("tmp@@@-2", cur_z.size())


            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                # print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)
    print("tmp@@@-3", len(z_list), zs.size(), zs.dtype)


    # Insert
    for i, layer in enumerate(hparams.layers):
        # print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, all_requests, hparams, layer, context_templates).T
        # print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")
        print("tmp@@@-4", layer_ks.size(), layer_ks.dtype)


        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in all_requests],
            words=[request["subject"] for request in all_requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
            track='out'
        ).T
        print("tmp@@@-4-2", cur_zs.size(), cur_zs.dtype)

        targets = zs - cur_zs
        print("tmp@@@-5", layer, targets.size(), cur_zs.size())

        # print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        print("tmp@@@-5", targets.size(), layer_ks.size())

        # Load covariance matrix
        force_recompute = False
        # force_recompute = layer != hparams.layers[0]
        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
            hparams=hparams
        )
        print("tmp@@@-6", cov.size())

        # Compute update in double precision
        layer_ks, targets = (
            layer_ks.double(),
            targets.double(),
        )

        adj_k = torch.linalg.solve(
            hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,
            layer_ks,
        )
        # print("tmp@@@-7", adj_k.size())
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = resid @ adj_k.T
        print("tmp@@@-8", resid.size(), adj_k.size(), cov.size())

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        # print("orig norm", torch.linalg.norm(weights[weight_name]))
        # print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
            )

        # Clear GPU memory
        cov.cpu()
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    # model
    # print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


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

    model_name = "_".join(model.config._name_or_path.split("/")[-2:])
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            hparams.stats_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            hparams=hparams,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to(f"cuda:{hparams.device}")) if inv else COV_CACHE[key].to(f"cuda:{hparams.device}")
    )


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


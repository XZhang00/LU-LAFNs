from copy import deepcopy
from typing import Dict, List, Tuple

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook
from ...util.generate import generate_fast

from .compute_u import compute_u
from .compute_v import compute_v
from .rome_hparams import ROMEHyperParams
torch.manual_seed(1234)

CONTEXT_TEMPLATES_CACHE = None
# CONTEXT_TEMPLATES_CACHE = {'zh': ['{}', '我爱阅读。  猴妈妈教. {}', '他以笔为武器,坚持革命斗争,. {}', '请根据下面的歌词,写出歌词的上. {}', '你是一个在线问诊机器医生,请根据患者. {}', '这幅图主要反映了我国的( )\n. {}'], 'en': ['{}', 'The number of goals scored in a game against Bar. {}', 'Therefore, the number of samples that are used is. {}', 'Because of budget cuts one IT company established new non. {}', 'I know that $\\cos \\theta $ normally has. {}', 'You are an AI assistant. User will you give. {}']}


def apply_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: List[Dict],
    hparams: ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """
    request = request[0]
    request = request[request["source_lang"]]
    if copy:
        model = deepcopy(model)

    weights_copy = {}

    deltas = execute_rome(model, tok, request, hparams)

    with torch.no_grad():
        for w_name, (delta_u, delta_v) in deltas.items():
            upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def apply_mrome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: List[Dict],
    hparams: ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """
    request = request[0]

    if copy:
        model = deepcopy(model)

    weights_copy = {}

    deltas = execute_mrome(model, tok, request, hparams)

    with torch.no_grad():
        for w_name, (delta_u, delta_v) in deltas.items():
            upd_matrix = delta_u @ delta_v.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
            print("tmp@@@", upd_matrix.size())

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def execute_mrome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    ori_request = deepcopy(request)
    all_requests = []
    all_langs = set()
    
    assert "_" in ori_request["source_lang"]
    for i_lang in ori_request["source_lang"].split("_"):
        all_langs.add(i_lang)
        tmp_req = ori_request[i_lang]
        tmp_req["lang"] = i_lang
        if tmp_req['subject'] not in tmp_req['prompt']: continue
        tmp_req['prompt'] = tmp_req['prompt'].replace(tmp_req['subject'], '{}', 1)
        all_requests.append(tmp_req)
        
    print("tmp@@@-1", all_langs, all_requests)

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    context_templates = get_context_templates(model, tok, hparams.context_template_length_params, list(all_langs))
    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        left_vectors_list = []
        right_vectors_list = []
        for i_request in all_requests:
            print("!!!-1", i_request)
            # print(tok)
            left_vector: torch.Tensor = compute_u(
                model,
                tok,
                i_request,
                hparams,
                layer,
                context_templates[i_request["lang"]],
            )
            left_vectors_list.append(left_vector)
            print("Left vector shape:", left_vector.shape, left_vector.dtype)
            print("!!!-2", i_request)
            # print(tok)
            right_vector: torch.Tensor = compute_v(
                model,
                tok,
                i_request,
                hparams,
                layer,
                left_vector,
                context_templates[i_request["lang"]],
            )
            print("Right vector shape:", right_vector.shape, right_vector.dtype)
            right_vector = right_vector.to(left_vector.dtype)
            print("Right vector shape:", right_vector.shape, right_vector.dtype)
            right_vectors_list.append(right_vector)

        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            left_vectors = torch.stack(left_vectors_list, dim=1)
            right_vectors = torch.stack(right_vectors_list, dim=1)
            upd_matrix = left_vectors @ right_vectors.T
            print("tmp@@@", left_vectors.size(), right_vectors.size(), upd_matrix.size())
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # Update model weights and record desired changes in `delta` variable
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vectors.detach(),
                right_vectors.detach(),
            )

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def execute_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    request = deepcopy(request)
    print(request)
    # if request["target_new"] != " ":
    #     # Space required for correct tokenization
    #     request["target_new"] = " " + request["target_new"]

    if('{}' not in request['prompt']):
        assert request['subject'] in request['prompt'] or \
               print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")

        request['prompt'] = request['prompt'].replace(request['subject'], '{}')

    print(
        f"Executing ROME algorithm for the update: "
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

    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        left_vector: torch.Tensor = compute_u(
            model,
            tok,
            request,
            hparams,
            layer,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        print("Left vector shape:", left_vector.shape, left_vector.dtype)
        right_vector: torch.Tensor = compute_v(
            model,
            tok,
            request,
            hparams,
            layer,
            left_vector,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        print("Right vector shape:", right_vector.shape, right_vector.dtype)
        right_vector = right_vector.to(left_vector.dtype)
        print("Right vector shape:", right_vector.shape, right_vector.dtype)


        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # Update model weights and record desired changes in `delta` variable
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

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
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )


# def get_context_templates(model, tok, length_params):
#     global CONTEXT_TEMPLATES_CACHE

#     if CONTEXT_TEMPLATES_CACHE is None:
        # CONTEXT_TEMPLATES_CACHE = ["{}"] + [
        #     x + ". {}"
        #     for x in sum(
        #         (
        #             generate_fast(
        #                 model,
        #                 tok,
        #                 ["The", "Therefore", "Because", "I", "You"],
        #                 n_gen_per_prompt=n_gen // 5,
        #                 max_out_len=length,
        #             )
        #             for length, n_gen in length_params
        #         ),
        #         [],
        #     )
        # ]

#         print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

#     return CONTEXT_TEMPLATES_CACHE


def get_context_templates(model, tok, length_params, langs=None):
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
        if langs == None:
            CONTEXT_TEMPLATES_CACHE = ["{}"] + [
                x + ". {}"
                for x in sum(
                    (
                        generate_fast(
                            model,
                            tok,
                            ["The", "Therefore", "Because", "I", "You"],
                            n_gen_per_prompt=n_gen // 5,
                            max_out_len=length,
                        )
                        for length, n_gen in length_params
                    ),
                    [],
                )
            ]
        else:
            CONTEXT_TEMPLATES_CACHE = {}
            for i_lang in langs:
                tmp_cache = ["{}"] + [
                    x + ". {}"
                    for x in sum(
                        (
                            generate_fast(
                                model,
                                tok,
                                lang_prefix_dict[i_lang],
                                n_gen_per_prompt=n_gen // 5,
                                max_out_len=length,
                            )
                            for length, n_gen in length_params
                        ),
                        [],
                    )
                ]
                CONTEXT_TEMPLATES_CACHE[i_lang] = []
                for tmp in tmp_cache:
                    if tmp.count("{") > 1 or tmp.count("}") > 1: continue
                    CONTEXT_TEMPLATES_CACHE[i_lang].append(tmp)
    print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")
    return CONTEXT_TEMPLATES_CACHE


"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain
from typing import List

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
# from .portability_evaluate import compute_portability_quality
from types import MethodType
from torch.nn import functional as F

is_llama=None
is_qwen=None
is_bloom=None


def compute_icl_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    icl_examples,
    record: typing.Dict,
    device,
    pre_edit: bool = False,
    source_lang: str = "en",
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target_new_en, target_new_zh, ground_truth = (
        record[x] for x in ["target_new_en", "target_new_zh", "ground_truth"]
    )
    prompt = record["prompt"]
    rephrase_en = record["rephrase_prompt_en"] if 'rephrase_prompt_en' in record.keys() else None
    rephrase_zh = record["rephrase_prompt_zh"] if 'rephrase_prompt_zh' in record.keys() else None
    # locality_prompt = record["locality_prompt"] if 'locality_prompt' in record.keys() else None
    # locality_ground_truth = record["locality_ground_truth"] if 'locality_ground_truth' in record.keys() else None

    # one_hop_prompt = record["one_hop_prompt"] if 'one_hop_prompt' in record.keys() else None
    # one_hop_ground_truth = record["one_hop_ground_truth"] if 'one_hop_ground_truth' in record.keys() else None
    # synonym_prompt = record["synonym_prompt"] if 'synonym_prompt' in record.keys() else None
    # synonym_ground_truth = record["synonym_ground_truth"] if 'synonym_ground_truth' in record.keys() else None
    # inverse_relation_prompt = record["inverse_relation_prompt"] if 'inverse_relation_prompt' in record.keys() else None
    # inverse_relation_ground_truth = record["inverse_relation_ground_truth"] if 'inverse_relation_ground_truth' in record.keys() else None

    if source_lang == "en":
        new_fact = f'New Fact: {prompt} {target_new_en}\nPrompt: {prompt}'
    elif source_lang == "zh":
        new_fact = f'New Fact: {prompt} {target_new_zh}\nPrompt: {prompt}'
    else:
        raise NotImplementedError()

    if pre_edit:
        if source_lang == "en":
            edit_acc_ans, edit_acc_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_new_en, prompt)
        else:
            edit_acc_ans, edit_acc_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_new_zh, prompt)
    else:
        if source_lang == "en":
            edit_acc_ans, edit_acc_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_new_en, new_fact)
        else:
            edit_acc_ans, edit_acc_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_new_zh, new_fact)

    ret = {
        f"rewrite_acc": {
            "ans": edit_acc_ans,
            "target": edit_acc_target
        }
    }
    ret['locality_en'] = {}
    ret['locality_zh'] = {}
    ret['portability_en'] = {}
    ret['portability_zh'] = {}

    if rephrase_en is not None:
        rephrase_acc_en_ans, rephrase_acc_en_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target_new_en, f'New Fact: {prompt} {target_new_en}\nPrompt: {rephrase_en}')
        ret['rephrase_acc_en'] = {
            "ans": rephrase_acc_en_ans,
            "target": rephrase_acc_en_target
        }

    if rephrase_zh is not None:
        rephrase_acc_zh_ans, rephrase_acc_zh_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target_new_zh, f'New Fact: {prompt} {target_new_zh}\nPrompt: {rephrase_zh}')
        ret['rephrase_acc_zh'] = {
            "ans": rephrase_acc_zh_ans,
            "target": rephrase_acc_zh_target
        }

    if 'locality_en' in record.keys() and any(record['locality_en']):
        for locality_key in record['locality_en'].keys():
            pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''], record['locality_en'][locality_key]['ground_truth'],
                                       f"New Fact: {prompt} {target_new_en}\nPrompt: {record['locality_en'][locality_key]['prompt']}", neighborhood=True)
            post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['locality_en'][locality_key]['ground_truth'],
                                        f"New Fact: {prompt} {target_new_en}\nPrompt: {record['locality_en'][locality_key]['prompt']}", neighborhood=True)
            # if type(pre_neighbor) is not list:
            #     pre_neighbor = [pre_neighbor, ]
            # if type(post_neighbor) is not list:
            #     post_neighbor = [post_neighbor, ]
            # assert len(pre_neighbor) == len(post_neighbor)

            # ret['locality_en'][f'{locality_key}_acc_en'] = np.mean(np.equal(pre_neighbor, post_neighbor))

            ret['locality_en'][f'{locality_key}_acc_en'] = {
                "pre": pre_neighbor,
                "post": post_neighbor
            }

    if 'locality_zh' in record.keys() and any(record['locality_zh']):
        for locality_key in record['locality_zh'].keys():
            pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''], record['locality_zh'][locality_key]['ground_truth'],
                                       f"New Fact: {prompt} {target_new_zh}\nPrompt: {record['locality_zh'][locality_key]['prompt']}", neighborhood=True)
            post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['locality_zh'][locality_key]['ground_truth'],
                                        f"New Fact: {prompt} {target_new_zh}\nPrompt: {record['locality_zh'][locality_key]['prompt']}", neighborhood=True)
            # if type(pre_neighbor) is not list:
            #     pre_neighbor = [pre_neighbor, ]
            # if type(post_neighbor) is not list:
            #     post_neighbor = [post_neighbor, ]
            # assert len(pre_neighbor) == len(post_neighbor)

            # ret['locality_zh'][f'{locality_key}_acc'] = np.mean(np.equal(pre_neighbor, post_neighbor))

            ret['locality_zh'][f'{locality_key}_acc_zh'] = {
                "pre": pre_neighbor,
                "post": post_neighbor
            }

    
    # Form a list of lists of prefixes to test.
    if 'portability_en' in record.keys() and any(record['portability_en']):
        for portability_key in record['portability_en'].keys():
            if pre_edit:
                portability_acc_ans, portability_acc_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['portability_en'][portability_key]['ground_truth'],
                                              record['portability_en'][portability_key]['prompt'])
            else:
                portability_acc_ans, portability_acc_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['portability_en'][portability_key]['ground_truth'],
                                              f"New Fact: {prompt} {target_new_en}\nPrompt: {record['portability_en'][portability_key]['prompt']}")
            ret['portability_en'][f'{portability_key}_acc_en'] = {
                "ans": portability_acc_ans,
                "target": portability_acc_target
            }

    if 'portability_zh' in record.keys() and any(record['portability_zh']):
        for portability_key in record['portability_zh'].keys():
            if pre_edit:
                portability_acc_ans, portability_acc_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['portability_zh'][portability_key]['ground_truth'],
                                              record['portability_zh'][portability_key]['prompt'])
            else:
                portability_acc_ans, portability_acc_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['portability_zh'][portability_key]['ground_truth'],
                                              f"New Fact: {prompt} {target_new_zh}\nPrompt: {record['portability_zh'][portability_key]['prompt']}")
            ret['portability_zh'][f'{portability_key}_acc_zh'] = {
                "ans": portability_acc_ans,
                "target": portability_acc_target
            }
    # if one_hop_prompt is not None:
    #     one_hop_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
    #                            one_hop_ground_truth, f'New Fact: {prompt} {target_new}\nPrompt: {one_hop_prompt}')
    #     ret['one_hop_acc'] = one_hop_acc
    # if synonym_prompt is not None:
    #     synonym_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
    #                            synonym_ground_truth, f'New Fact: {prompt} {target_new}\nPrompt: {synonym_prompt}')
    #     ret['synonym_acc'] = synonym_acc
    # if inverse_relation_prompt is not None:
    #     inverse_relation_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
    #                            inverse_relation_ground_truth, f'New Fact: {prompt} {target_new}\nPrompt: {inverse_relation_prompt}')
    #     ret['inverse_relation_acc'] = inverse_relation_acc
    return ret

def icl_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        neighborhood=False
)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')
    if 't5' in model_name.lower():
        target_len = len(tokenizer.encode(target))
        target_ids = tokenizer(f'{x} {target}', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples), return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).logits
            ans = torch.argmax(logits, dim=-1)[:,-target_len:-1].squeeze()
            target_ids = target_ids[:,-target_len:-1]
            if neighborhood:
                return ans.squeeze().detach().cpu().numpy().tolist()
            return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    elif 'llama' in model_name.lower() or 'baichuan' in model_name.lower():
        target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,1:]
        
        ans_idss = ans.detach().cpu().numpy().tolist()
        target_idss = target_ids.detach().cpu().squeeze().numpy().tolist()
        if not isinstance(ans_idss, list):
            ans_idss = [ans_idss]

        textual_ans = tokenizer.decode(ans_idss, skip_special_tokens=True)
        textual_target = tokenizer.decode(target_idss, skip_special_tokens=True)

        if neighborhood:
            return textual_ans
            # return ans.squeeze().detach().cpu().numpy().tolist()

        

        return textual_ans, textual_target
        # return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()        
    else:
        target_ids = tokenizer(' ' + target + '\n', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,:-1]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()

# TODO: Support GPT Evaluation(predict token one by one)
def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    device,
    test_rephrase: bool = False,
    delta_key = None,
    subject = None,
) -> typing.Dict:

    if 't5' in model_name.lower():
        stuff_probs = test_seq2seq_batch_prediction_acc(model, tok, hparams,
                                                        prompt,
                                                        target_new,
                                                        device)
    elif 'gpt' in model_name.lower():
        target_tok = tok(target_new, truncation=True, max_length=hparams.max_length)["input_ids"]
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        # inp_targets = [
        #     tok.decode(target_tok[i])
        #     for i in range(len(target_tok))
        # ]
        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)
    elif 'llama' in model_name.lower():
        target_tok = tok(target_new, truncation=True, max_length=hparams.max_length)["input_ids"] 
        if target_tok[0] == tok.bos_token_id or hparams.alg_name == 'SERAC':
            target_tok = target_tok[1:]
        if hparams.alg_name == 'LORAFT':
            prompt = f"Question: {prompt} \nAnswer: "
            target_tok = [tok.bos_token_id] + target_tok
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        # print(inp_prompts)
        # print("@@@-2", target_new, target_tok, inp_prompts)
        if delta_key is not None and subject is not None:
            print("@@@", subject, prompt, (subject in prompt))
            if subject in prompt:
                if hparams.alg_name == 'MVALUE':
                    textual_ans, textual_target = test_batch_prediction_acc_value(subject, delta_key, model, tok, hparams, inp_prompts, target_tok, device)
                else:
                    textual_ans, textual_target = test_batch_prediction_acc_key(subject, delta_key, model, tok, hparams, inp_prompts, target_tok, device)
            else:
                textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)
        else:
            textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)
    elif 'qwen' in model_name.lower() or 'bloom' in model_name.lower():
        target_tok = tok(target_new, truncation=True, max_length=hparams.max_length)["input_ids"] 
        print("!!-target_tok", target_new, target_tok)
        if target_tok[0] == tok.unk_token_id or hparams.alg_name == 'SERAC':
            target_tok = target_tok[1:]
        if hparams.alg_name == 'LORAFT':
            prompt = f"Question: {prompt} \nAnswer: "
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        print(inp_prompts)
        # print("@@@-2", target_new, target_tok, inp_prompts)
        if delta_key is not None and subject is not None:
            print("@@@", subject, prompt, (subject in prompt))
            if subject in prompt:
                if hparams.alg_name == 'MVALUE':
                    textual_ans, textual_target = test_batch_prediction_acc_value(subject, delta_key, model, tok, hparams, inp_prompts, target_tok, device)
                else:
                    textual_ans, textual_target = test_batch_prediction_acc_key(subject, delta_key, model, tok, hparams, inp_prompts, target_tok, device)
            else:
                textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)
        else:
            textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)

    elif 'baichuan' in model_name.lower():
        target_tok = tok(target_new, truncation=True, max_length=hparams.max_length)["input_ids"] 
        if target_tok[0] == tok.unk_token_id or hparams.alg_name == 'SERAC':
            target_tok = target_tok[1:]
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)


    # Structure the restuls as a dictionary.

    if not test_rephrase:
        key = 'rewrite'
    else:
        key = 'rephrase'
    
    ret = {
        f"{key}_acc": {
            "ans": textual_ans,
            "target": textual_target
        }
    }

    return ret


def compute_locality_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: str,
    locality_ground_truth: str,
    device,
    delta_key = None,
    subject = None
) -> typing.Dict:

    if 't5' in model_name.lower():
        locality_correct = test_seq2seq_batch_prediction_acc(model, tok, hparams,
                                                                 prompt,
                                                                 locality_ground_truth,
                                                                 device,
                                                                 locality=True)
    elif 'gpt' in model_name.lower():
        target_tok = tok(locality_ground_truth, truncation=True, max_length=hparams.max_length)["input_ids"]
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])

        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device, locality=True)
    elif 'llama' in model_name.lower():
        target_tok = tok(locality_ground_truth, truncation=True, max_length=hparams.max_length)["input_ids"] # erase bos_token_id
        if target_tok[0] == tok.bos_token_id or hparams.alg_name == 'SERAC':
            target_tok = target_tok[1:]
        # if hparams.alg_name == 'LORAFT':
        #     prompt = f"Question: {prompt} \nAnswer: "
        #     target_tok = [tok.bos_token_id] + target_tok
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        # print(inp_prompts)
        if delta_key is not None and subject is not None:
            if subject in prompt:
                if hparams.alg_name == 'MVALUE':
                    textual_ans, textual_target = test_batch_prediction_acc_value(subject, delta_key, model, tok, hparams, inp_prompts, target_tok, device)
                else:
                    textual_ans, textual_target = test_batch_prediction_acc_key(subject, delta_key, model, tok, hparams, inp_prompts, target_tok, device)
            else:
                textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)
        else:
            textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)
    elif 'qwen' in model_name.lower() or 'bloom' in model_name.lower():
        target_tok = tok(locality_ground_truth, truncation=True, max_length=hparams.max_length)["input_ids"] # erase bos_token_id
        if target_tok[0] == tok.unk_token_id or hparams.alg_name == 'SERAC':
            target_tok = target_tok[1:]
        # if hparams.alg_name == 'LORAFT':
        #     prompt = f"Question: {prompt} \nAnswer: "
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        # print(inp_prompts)
        if delta_key is not None and subject is not None:
            if subject in prompt:
                if hparams.alg_name == 'MVALUE':
                    textual_ans, textual_target = test_batch_prediction_acc_value(subject, delta_key, model, tok, hparams, inp_prompts, target_tok, device)
                else:
                    textual_ans, textual_target = test_batch_prediction_acc_key(subject, delta_key, model, tok, hparams, inp_prompts, target_tok, device)
            else:
                textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)
        else:
            textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)

    elif 'baichuan' in model_name.lower():
        target_tok = tok(locality_ground_truth, truncation=True, max_length=hparams.max_length)["input_ids"] # erase bos_token_id
        if target_tok[0] == tok.unk_token_id or hparams.alg_name == 'SERAC':
            target_tok = target_tok[1:]
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device, locality=True)
    
    ret = {
        f"{locality_key}_output": {
            "ans": textual_ans,
            "target": textual_target
        }
    }
    return ret


def compute_portability_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    portability_key: str,
    prompt: str,
    ground_truth: str,
    device,
    delta_key = None,
    subject = None
) -> typing.Dict:

    if 't5' in model_name.lower():
        portability_correct = test_seq2seq_batch_prediction_acc(model, tok, hparams,
                                                                 prompt,
                                                                 ground_truth,
                                                                 device)
    elif 'gpt' in model_name.lower():
        target_tok = tok(" " + ground_truth, truncation=True, max_length=hparams.max_length)["input_ids"]
        inp_prompts = [
            prompt + tok.decode(target_tok[:i])
            for i in range(len(target_tok))
        ]
        inp_targets = [
            tok.decode(target_tok[i])
            for i in range(len(target_tok))
        ]

        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, inp_targets, device)
    elif 'llama' in model_name.lower():
        target_tok = tok(ground_truth, truncation=True, max_length=hparams.max_length)["input_ids"]
        if target_tok[0] == tok.bos_token_id or hparams.alg_name == 'SERAC':
            target_tok = target_tok[1:]
        if hparams.alg_name == 'LORAFT':
            prompt = f"Question: {prompt} \nAnswer: "
            target_tok = [tok.bos_token_id] + target_tok
        inp_prompts = [
            prompt + tok.decode(target_tok[:i])
            for i in range(len(target_tok))
        ]
        # print(inp_prompts)
        inp_targets = [
            tok.decode(target_tok[i])
            for i in range(len(target_tok))
        ]
        if delta_key is not None and subject is not None:
            if subject in prompt:
                if hparams.alg_name == 'MVALUE':
                    textual_ans, textual_target = test_batch_prediction_acc_value(subject, delta_key, model, tok, hparams, inp_prompts, target_tok, device)
                else:
                    textual_ans, textual_target = test_batch_prediction_acc_key(subject, delta_key, model, tok, hparams, inp_prompts, target_tok, device)
            else:
                textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)
        else:
            textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)
    
    elif 'qwen' in model_name.lower() or 'bloom' in model_name.lower():
        target_tok = tok(ground_truth, truncation=True, max_length=hparams.max_length)["input_ids"]
        if target_tok[0] == tok.unk_token_id or hparams.alg_name == 'SERAC':
            target_tok = target_tok[1:]
        if hparams.alg_name == 'LORAFT':
            prompt = f"Question: {prompt} \nAnswer: "
        inp_prompts = [
            prompt + tok.decode(target_tok[:i])
            for i in range(len(target_tok))
        ]
        # print(inp_prompts)
        inp_targets = [
            tok.decode(target_tok[i])
            for i in range(len(target_tok))
        ]
        if delta_key is not None and subject is not None:
            if subject in prompt:
                if hparams.alg_name == 'MVALUE':
                    textual_ans, textual_target = test_batch_prediction_acc_value(subject, delta_key, model, tok, hparams, inp_prompts, target_tok, device)
                else:
                    textual_ans, textual_target = test_batch_prediction_acc_key(subject, delta_key, model, tok, hparams, inp_prompts, target_tok, device)
            else:
                textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)
        else:
            textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)

    elif 'baichuan' in model_name.lower():
        target_tok = tok(" " + ground_truth, truncation=True, max_length=hparams.max_length)["input_ids"]
        inp_prompts = [
            prompt + tok.decode(target_tok[:i])
            for i in range(len(target_tok))
        ]
        inp_targets = [
            tok.decode(target_tok[i])
            for i in range(len(target_tok))
        ]

        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, inp_targets, device)


    ret = {
        f"{portability_key}_acc": {
            "ans": textual_ans,
            "target": textual_target
        }
    }
    return ret


def compute_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    edit_lang="en",
    test_langs=["en", "zh"]
) -> typing.Dict:

    res = {}
    for i_lang in test_langs:
        target_new = record[i_lang]["target_new"]
        rewrite_prompts = record[i_lang]["prompt"]
        rephrase_prompts = record[i_lang]["rephrase_prompt"]
        ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok, rewrite_prompts, target_new, device=device)
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok, rephrase_prompts, target_new, device=device, test_rephrase=True)
        )
        ret['locality'] = {}
        for locality_key in record[i_lang]['locality'].keys():
            ret['locality'].update(
                compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                         record[i_lang]['locality'][locality_key]['prompt'],
                                         record[i_lang]['locality'][locality_key]['ground_truth'], device=device)
            )
        if "loc_sub_list" in record[i_lang]:
            ret['locality_sub'] = {}
            for i_, i_loc_item in enumerate(record[i_lang]["loc_sub_list"]):
                locality_key = "loc_sub_" + str(i_)
                ret["locality_sub"].update(
                    compute_locality_quality(model, model_name, hparams, tok, locality_key, 
                                        i_loc_item["loc-sub"], i_loc_item["answer"], device)
                    )
        ret['portability'] = {}
        for portability_key in record[i_lang]['portability'].keys():
            ret['portability'].update(
                compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            record[i_lang]['portability'][portability_key]['prompt'],
                                            record[i_lang]['portability'][portability_key]['ground_truth'], device=device)
            )
        res[i_lang] = ret

    return res


def compute_edit_remake_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    edit_lang="en",
    test_langs=["en", "zh"]
) -> typing.Dict:

    res = {}
    for i_lang in test_langs:
        cur_prefix = ""
        for j_lang in edit_lang.split("_"):
            # if j_lang == i_lang: continue
            # cur_prefix += "Question: " + record[j_lang]["prompt"] + "\nAnswer: " + record[j_lang]["target_new"] + "\n"
            cur_prefix += "Question: " + record[j_lang]["rephrase_prompt"] + "\nAnswer: " + record[j_lang]["target_new"] + "\n"
        assert cur_prefix != ""
        print("@@@-edit-lang:", edit_lang, "test-lang:", i_lang, "prefix-length:", len(cur_prefix), cur_prefix)
        target_new = record[i_lang]["target_new"]
        rewrite_prompts = record[i_lang]["prompt"]
        rephrase_prompts = record[i_lang]["rephrase_prompt"]
        ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok, 
                                                  cur_prefix + "Question: " + rewrite_prompts + "\nAnswer: ", 
                                                  target_new, device=device)
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok, 
                                                cur_prefix + "Question: " + rephrase_prompts + "\nAnswer: ", 
                                                target_new, device=device, test_rephrase=True)
        )
        ret['locality'] = {}
        for locality_key in record[i_lang]['locality'].keys():
            ret['locality'].update(
                compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                         record[i_lang]['locality'][locality_key]['prompt'],
                                         record[i_lang]['locality'][locality_key]['ground_truth'], device=device)
            )
        if "loc_sub_list" in record[i_lang]:
            ret['locality_sub'] = {}
            for i_, i_loc_item in enumerate(record[i_lang]["loc_sub_list"]):
                locality_key = "loc_sub_" + str(i_)
                ret["locality_sub"].update(
                    compute_locality_quality(model, model_name, hparams, tok, locality_key, 
                                            i_loc_item["loc-sub"], i_loc_item["answer"], device)
                    )
        ret['portability'] = {}
        for portability_key in record[i_lang]['portability'].keys():
            ret['portability'].update(
                compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            cur_prefix + "Question: " + record[i_lang]['portability'][portability_key]['prompt'] + "\nAnswer: ",
                                            record[i_lang]['portability'][portability_key]['ground_truth'], device=device)
            )
        res[i_lang] = ret

    return res


def compute_key_edit_quality(
    delta_key,
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    edit_lang="en",
    test_langs=["en", "zh"]
) -> typing.Dict:
    
    res = {}
    for i_lang in test_langs:
        prefix = ""
        # if "_" in edit_lang:
        #     if i_lang == "en":
        #         prefix = "Please answer in English. "
        #     else:
        #         prefix = "Please answer in Chinese. "
        target_new = record[i_lang]["target_new"]
        rewrite_prompts = record[i_lang]["prompt"]
        rephrase_prompts = record[i_lang]["rephrase_prompt"]
        if "_" in edit_lang:
            """
            这里的逻辑还不太对，
            现在是：一个句子里有中文subject，用中文的subject匹配；英文的用英文的匹配；
            如果一个句子里中英文的都有，或者中文里有英文的，或者英文里有中文的？这些情况怎么处理？
            """
            ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok, prefix+rewrite_prompts, target_new, 
                                                    device=device, delta_key=delta_key, subject=record[i_lang]["subject"])
            ret.update(
                compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok, prefix+rephrase_prompts, target_new, 
                                                    device=device, test_rephrase=True, delta_key=delta_key, subject=record[i_lang]["subject"])
            )
            ret['locality'] = {}
            for locality_key in record[i_lang]['locality'].keys():
                ret['locality'].update(
                    compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                            prefix+record[i_lang]['locality'][locality_key]['prompt'],
                                            record[i_lang]['locality'][locality_key]['ground_truth'], device=device,
                                            delta_key=delta_key, subject=record[i_lang]["subject"])
                )
            if "loc_sub_list" in record[i_lang]:
                ret['locality_sub'] = {}
                for i_, i_loc_item in enumerate(record[i_lang]["loc_sub_list"]):
                    locality_key = "loc_sub_" + str(i_)
                    ret["locality_sub"].update(
                        compute_locality_quality(model, model_name, hparams, tok, locality_key, 
                                            i_loc_item["loc-sub"], i_loc_item["answer"], device, delta_key=delta_key, subject=record[i_lang]["subject"])
                        )
            ret['portability'] = {}
            for portability_key in record[i_lang]['portability'].keys():
                ret['portability'].update(
                    compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                                prefix+record[i_lang]['portability'][portability_key]['prompt'],
                                                record[i_lang]['portability'][portability_key]['ground_truth'], device=device,
                                                delta_key=delta_key, subject=record[i_lang]["subject"])
                )
        else:
            ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok, rewrite_prompts, target_new, 
                                                    device=device, delta_key=delta_key, subject=record[edit_lang]["subject"])
            ret.update(
                compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok, rephrase_prompts, target_new, 
                                                    device=device, test_rephrase=True, delta_key=delta_key, subject=record[edit_lang]["subject"])
            )
            ret['locality'] = {}
            for locality_key in record[i_lang]['locality'].keys():
                ret['locality'].update(
                    compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                            record[i_lang]['locality'][locality_key]['prompt'],
                                            record[i_lang]['locality'][locality_key]['ground_truth'], device=device,
                                            delta_key=delta_key, subject=record[edit_lang]["subject"])
                )
            ret['portability'] = {}
            for portability_key in record[i_lang]['portability'].keys():
                ret['portability'].update(
                    compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                                record[i_lang]['portability'][portability_key]['prompt'],
                                                record[i_lang]['portability'][portability_key]['ground_truth'], device=device,
                                                delta_key=delta_key, subject=record[edit_lang]["subject"])
                )
        
        res[i_lang] = ret

    return res
    
    

def test_batch_prediction_acc_key(subject, delta_key, model, tok, hparams, prompts, target, device, locality=False):
    # 如果有subject再执行这个，先把model改了，再把model恢复
    assert subject in prompts[0]
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")
    print("&&&&", prompts)
    # print("prompt_tok", prompt_tok["input_ids"].size(), prompt_tok["input_ids"])

    all_prompts = [jj.replace(subject, '{}') for jj in prompts]
    # print("&&&&", all_prompts)

    # lookup_idxs = [
    #     find_fact_lookup_idx(
    #         tmp_prompt, subject, tok, hparams.fact_token, verbose=(i == 0)
    #     )
    #     for i, tmp_prompt in enumerate(all_prompts[0])
    # ] 
    # print("!!!", prompts[0].replace(subject, '{}'))
    if prompts[0].replace(subject, '{}').count('{}') > 1:
        print("&"*10, "many subjects in sentence.")
        lookup_idx = find_fact_lookup_idx(
                prompts[0].replace(subject, '{}', 1), subject, tok, hparams.fact_token, verbose=True
            )   # 先直接替换第一个，这里后面可以再优化一下
    else:
        lookup_idx = find_fact_lookup_idx(
                prompts[0].replace(subject, '{}'), subject, tok, hparams.fact_token, verbose=True
            )


    print("!!!!!-lookup_idx:", lookup_idx)

    weights_copy = {}

    global is_llama, is_qwen, is_bloom

    if "llama" in hparams.model_name.lower():
        is_llama = True
    if "qwen" in hparams.model_name.lower():
        is_qwen = True
    if "bloom" in hparams.model_name.lower():
        is_bloom = True

    def factory(idx, lookup_idx):
        def new_forward(self, x):
            activation = self.act_fn(self.gate_proj(x))
            # 怎么相加？+ delta_key[idx]
            print("@@@-modified forward func:", idx, delta_key[idx].size(), delta_key[idx].norm())
            for uu in range(x.shape[0]):
                activation[uu, lookup_idx, :] += delta_key[idx]
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
            activation = self.gelu_impl(self.dense_h_to_4h(x))
            print("@@@-modified forward func:", idx, delta_key[idx].size(), delta_key[idx].norm(), x.size())
            for uu in range(x.shape[0]):
                activation[uu, lookup_idx, :] += delta_key[idx]
            intermediate_output = self.dense_4h_to_h(activation)
            output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)

            return output
        
        if is_llama or is_qwen:
            return new_forward
        elif is_bloom:
            return bloom_forward

    for i, cur_layer in enumerate(hparams.layers):
        if is_llama or is_qwen:
            obj = model.model.layers[cur_layer].mlp
        elif is_bloom:
            obj = model.transformer.h[cur_layer].mlp
        weights_copy[cur_layer] = obj.forward
        obj.forward = MethodType(factory(i, lookup_idx), obj)

    with torch.no_grad():
        tmp_logits = []
        cnt = int(prompt_tok["input_ids"].shape[0] / hparams.eval_bsz) + 1
        for i_iter in range(cnt):
            st = i_iter * hparams.eval_bsz
            ed = min((i_iter + 1) * hparams.eval_bsz, prompt_tok["input_ids"].shape[0])
            if ed == st: continue
            tmp_logits.append(model(input_ids=prompt_tok["input_ids"][st:ed], 
                                    attention_mask=prompt_tok["attention_mask"][st:ed]).logits)
        logits = torch.cat(tmp_logits, dim=0)
        print("z_logits", prompt_tok["input_ids"].shape[0], logits.size())
        # outputs = model(**prompt_tok)
        # if type(outputs) is torch.Tensor:
        #     logits = outputs
        # else:
        #     logits = outputs.logits
        # print("true", logits.size())

        if tok.padding_side == 'left':
            ans = torch.argmax(logits, dim=-1)[:, -1].squeeze()
        else:
            last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
            to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
            gathered = torch.gather(logits, 1, to_gather).squeeze(1)
            ans = torch.argmax(gathered, dim=1)
            # print("@@@-3", last_non_masked, to_gather.size(), gathered.size(), ans)

        ans = ans.squeeze().detach().cpu().numpy().tolist()

        # if locality:
        #     return ans
        
        textual_ans = tok.decode(ans, skip_special_tokens=True)
        textual_target = tok.decode(target, skip_special_tokens=True)
        # print("@@@-4", textual_ans, textual_target)
    
    for i, cur_layer in enumerate(hparams.layers):
        if is_llama or is_qwen:
            obj = model.model.layers[cur_layer].mlp
        elif is_bloom:
            obj = model.transformer.h[cur_layer].mlp
        obj.forward = weights_copy[cur_layer]

    return textual_ans, textual_target
        # return np.mean(np.equal(ans, target)), textual_ans, textual_target


def test_batch_prediction_acc_value(subject, delta_key, model, tok, hparams, prompts, target, device, locality=False):

    assert subject in prompts[0]
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")
    print("&&&&", prompts)
    # print("prompt_tok", prompt_tok["input_ids"].size(), prompt_tok["input_ids"])

    all_prompts = [jj.replace(subject, '{}') for jj in prompts]
    # print("&&&&", all_prompts)

    # lookup_idxs = [
    #     find_fact_lookup_idx(
    #         tmp_prompt, subject, tok, hparams.fact_token, verbose=(i == 0)
    #     )
    #     for i, tmp_prompt in enumerate(all_prompts[0])
    # ] 
    # print("!!!", prompts[0].replace(subject, '{}'))
    if prompts[0].replace(subject, '{}').count('{}') > 1:
        print("&"*10, "many subjects in sentence.")
        lookup_idx = find_fact_lookup_idx(
                prompts[0].replace(subject, '{}', 1), subject, tok, hparams.fact_token, verbose=True
            )   
    else:
        lookup_idx = find_fact_lookup_idx(
                prompts[0].replace(subject, '{}'), subject, tok, hparams.fact_token, verbose=True
            )


    print("!!!!!-lookup_idx:", lookup_idx)

    weights_copy = {}

    def factory(idx, lookup_idx):
        def new_forward(self, x):
            activation = self.act_fn(self.gate_proj(x))
           
            print("@@@-modified forward func:", idx, delta_key[idx].size(), delta_key[idx].norm())
            down_proj = self.down_proj(activation * self.up_proj(x))  # 原来模型的返回值
            for uu in range(x.shape[0]):
                down_proj[uu, lookup_idx, :] += delta_key[idx]

            return down_proj


        return new_forward

    for i, cur_layer in enumerate(hparams.layers):
        obj = model.model.layers[cur_layer].mlp
        weights_copy[cur_layer] = obj.forward
        obj.forward = MethodType(factory(i, lookup_idx), obj)

    with torch.no_grad():
        tmp_logits = []
        cnt = int(prompt_tok["input_ids"].shape[0] / hparams.eval_bsz) + 1
        for i_iter in range(cnt):
            st = i_iter * hparams.eval_bsz
            ed = (i_iter + 1) * hparams.eval_bsz
            tmp_logits.append(model(input_ids=prompt_tok["input_ids"][st:ed], 
                                    attention_mask=prompt_tok["attention_mask"][st:ed]).logits)
        logits = torch.cat(tmp_logits, dim=0)
        print("z_logits", prompt_tok["input_ids"].shape[0], logits.size())
        # outputs = model(**prompt_tok)
        # if type(outputs) is torch.Tensor:
        #     logits = outputs
        # else:
        #     logits = outputs.logits
        # print("true", logits.size())

        if tok.padding_side == 'left':
            ans = torch.argmax(logits, dim=-1)[:, -1].squeeze()
        else:
            last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
            to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
            gathered = torch.gather(logits, 1, to_gather).squeeze(1)
            ans = torch.argmax(gathered, dim=1)
            # print("@@@-3", last_non_masked, to_gather.size(), gathered.size(), ans)

        ans = ans.squeeze().detach().cpu().numpy().tolist()

        # if locality:
        #     return ans
        
        textual_ans = tok.decode(ans, skip_special_tokens=True)
        textual_target = tok.decode(target, skip_special_tokens=True)
        # print("@@@-4", textual_ans, textual_target)
    
    for i, cur_layer in enumerate(hparams.layers):
        obj = model.model.layers[cur_layer].mlp
        obj.forward = weights_copy[cur_layer]

    return textual_ans, textual_target
        # return np.mean(np.equal(ans, target)), textual_ans, textual_target


def test_batch_prediction_acc(model, tok, hparams, prompts, target, device, locality=False):
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")
    # print("@@@-prompt", prompts)
    # print("@@@-prompt-tok", prompt_tok)

    with torch.no_grad():
        tmp_logits = []
        cnt = int(prompt_tok["input_ids"].shape[0] / hparams.eval_bsz) + 1
        for i_iter in range(cnt):
            st = i_iter * hparams.eval_bsz
            ed = min((i_iter + 1) * hparams.eval_bsz, prompt_tok["input_ids"].shape[0])
            if ed == st: continue
            tmp_logits.append(model(input_ids=prompt_tok["input_ids"][st:ed], 
                                    attention_mask=prompt_tok["attention_mask"][st:ed]).logits)
        logits = torch.cat(tmp_logits, dim=0)
        print("z_logits", prompt_tok["input_ids"].shape[0], logits.size())
        # outputs = model(**prompt_tok)
        # if type(outputs) is torch.Tensor:
        #     logits = outputs
        # else:
        #     logits = outputs.logits
        # print("true", logits.size())

        if tok.padding_side == 'left':
            ans = torch.argmax(logits, dim=-1)[:, -1].squeeze()
        else:
            last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
            to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
            gathered = torch.gather(logits, 1, to_gather).squeeze(1)
            ans = torch.argmax(gathered, dim=1)
            # print("@@@-3", last_non_masked, to_gather.size(), gathered.size(), ans)

        ans = ans.squeeze().detach().cpu().numpy().tolist()

        # if locality:
        #     return ans
        
        textual_ans = tok.decode(ans, skip_special_tokens=True)
        textual_target = tok.decode(target, skip_special_tokens=True)
        # print("@@@-4", textual_ans, textual_target)
        torch.cuda.empty_cache()
        return textual_ans, textual_target
        # return np.mean(np.equal(ans, target)), textual_ans, textual_target


def test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target, device, locality=False):
    prompt_tok = tok(
        prompt,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    trg_tok = tok(
        target,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    prompt_tok['labels'] = trg_tok['input_ids']
    # prompt_tok['decoder_attention_mask'] = trg_tok['attention_mask']


    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        assert logits.size(1) == trg_tok['input_ids'].size(1)
        ans = torch.argmax(logits, dim=-1)
        if locality:
            return ans.squeeze().detach().cpu().numpy().tolist()

        return torch.mean((trg_tok['input_ids'][:,:-1] == ans[:,:-1]).float(), dim=-1).detach().cpu().numpy().tolist()[0]


def get_words_idxs_in_templates(
    tok: AutoTokenizer, context_templates: str, words: str, subtoken: str
) -> int:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    """

    print("@@@-get_words_idxs_in_templates", context_templates)
    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), "We currently do not support multiple fill-ins for context"



    prefixes_len, words_len, suffixes_len, inputs_len = [], [], [], []
    for i, context in enumerate(context_templates):
        prefix, suffix = context.split("{}")
        prefix_len = len(tok.encode(prefix))
        prompt_len = len(tok.encode(prefix + words[i]))
        input_len = len(tok.encode(prefix + words[i] + suffix))
        prefixes_len.append(prefix_len)
        words_len.append(prompt_len - prefix_len)
        suffixes_len.append(input_len - prompt_len)
        inputs_len.append(input_len)

    # Compute prefixes and suffixes of the tokenized context
    # fill_idxs = [tmp.index("{}") for tmp in context_templates]
    # prefixes, suffixes = [
    #     tmp[: fill_idxs[i]] for i, tmp in enumerate(context_templates)
    # ], [tmp[fill_idxs[i] + 2 :] for i, tmp in enumerate(context_templates)]
    # words = deepcopy(words)
    #
    # # Pre-process tokens
    # for i, prefix in enumerate(prefixes):
    #     if len(prefix) > 0:
    #         assert prefix[-1] == " "
    #         prefix = prefix[:-1]
    #
    #         prefixes[i] = prefix
    #         words[i] = f" {words[i].strip()}"
    #
    # # Tokenize to determine lengths
    # assert len(prefixes) == len(words) == len(suffixes)
    # n = len(prefixes)
    # batch_tok = tok([*prefixes, *words, *suffixes])
    # if 'input_ids' in batch_tok:
    #     batch_tok = batch_tok['input_ids']
    # prefixes_tok, words_tok, suffixes_tok = [
    #     batch_tok[i : i + n] for i in range(0, n * 3, n)
    # ]
    # prefixes_len, words_len, suffixes_len = [
    #     [len(el) for el in tok_list]
    #     for tok_list in [prefixes_tok, words_tok, suffixes_tok]
    # ]

    # Compute indices of last tokens
    if subtoken == "last" or subtoken == "first_after_last":
        return [
            [
                prefixes_len[i]
                + words_len[i]
                - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0)
            ]
            # If suffix is empty, there is no "first token after the last".
            # So, just return the last token of the word.
            for i in range(len(context_templates))
        ]
    elif subtoken == "first":
        return [[prefixes_len[i] - inputs_len[i]] for i in range(len(context_templates))]
    else:
        raise ValueError(f"Unknown subtoken type: {subtoken}")


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
        ret = get_words_idxs_in_templates(
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

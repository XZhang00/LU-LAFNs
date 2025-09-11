import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
torch.manual_seed(1234)
import logging, copy
import numpy as np
import os
from peft import PeftModel


from transformers import AutoTokenizer, AutoModelForCausalLM, BloomForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
from ..util.globals import *
from .singleton_editor import SingletonEditor
from .batch_editor import BatchEditor
from ..evaluate import compute_edit_quality, compute_rewrite_quality_zsre, compute_rewrite_quality_counterfact, compute_icl_edit_quality, compute_key_edit_quality, compute_edit_remake_quality
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)


LOG = logging.getLogger(__name__)


def make_logs():

    f_h, s_h = get_handler('your_path/Bi-ZsRE/logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


class BaseEditor:
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None or print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        print("!! alg_name:", self.alg_name)

        make_logs()

        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            if 't5' in self.model_name.lower():
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
                self.tok = T5Tokenizer.from_pretrained(self.model_name)
            elif 'gpt' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'llama' in self.model_name.lower():
                self.model = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, device_map="auto")
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
                print(self.model_name, "is loaded!")
            elif 'baichuan' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id     
            elif 'qwen' in  self.model_name.lower():  
                self.model = Qwen2ForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", device_map="cuda:0")
                self.tok = Qwen2Tokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
                print(self.model_name, "is loaded!")
            elif 'bloom' in self.model_name.lower(): 
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", device_map="auto")
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                print(self.model_name, "is loaded!")
            else:
                raise NotImplementedError

            if (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer)) \
                    and (hparams.alg_name not in ['ROME', 'MEMIT', 'PMET', 'KEY', 'MKEY', 
                                                  'MKEYPG', 'MVALUE', 'MROME', 'MMEMIT', 'MPMET', 
                                                  'LORAFT']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                self.tok.padding_side = 'left'
        else:
            self.model, self.tok = self.model_name
        # device_map = {
        #     0: [_ for _ in range(0, 16)],
        #     1: [_ for _ in range(16, 32)],
        #     2: [_ for _ in range(32, 48)]
        # }
        # self.model.parallelize(device_map=device_map)

        if hasattr(hparams, 'device'):
            self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams

    def edit(self,
             requests,
             keep_original_weight=False,
             verbose=True,
             source_lang="en",
             test_langs=["en", "zh"],
             **kwargs
             ):
        """
        `prompts`: dict{"en": [], "zh": [], ...}
            the multiligual prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        # assert source_lang in ["en", "zh"], "source language should in en or zh"

        # ground_truth = ['<|endoftext|>' for _ in range(len(prompts_edits[source_lang]))]


        # requests = self._prepare_requests(
        #     prompts_edits, 
        #     targets_new,
        #     ground_truth, 
        #     rephrase_prompts,
        #     locality_inputs, 
        #     portability_inputs, 
        #     source_lang, 
        #     test_langs
        # **kwargs)

        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1 or \
                      print(f'Single Edit, pls set the batch_size to 1....')


        all_metrics = []
        if kwargs["pre_res_path"] == "":
            for i, request in tqdm(enumerate(requests)):
                # if self.alg_name == 'IKE':
                #     assert 'train_ds' in kwargs.keys() or print('IKE need train_ds(For getting In-Context prompt)')
                #     metrics = {
                #         "pre": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                #                                          request, self.hparams.device, pre_edit=True, source_lang=source_lang)
                #     }
                # else:
                # metrics = {
                #         "pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                #                                 self.hparams.device, edit_lang=source_lang, test_langs=test_langs)
                #     }
                
                metrics = {
                    "pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                                            self.hparams.device, edit_lang=source_lang, test_langs=test_langs)
                }
                # except:
                #     metrics = {"pre": "error"}
                all_metrics.append(metrics)
                torch.cuda.empty_cache()
                if len(test_langs) == 2:
                    pre_res_path = f"your_path/Bi-ZsRE/results-new/{self.model_name.lower().split('/')[-1]}-BiZsRE-pre_res.json"
                else:
                    assert len(test_langs) > 2
                    pre_res_path = f"your_path/Bi-ZsRE/results-new/{self.model_name.lower().split('/')[-1]}-MzsRE-pre_res.json"
                json.dump(all_metrics, open(pre_res_path, 'w', encoding="utf-8"), ensure_ascii=False, indent=4)
        else:
            all_metrics = json.load(open(kwargs["pre_res_path"]))

        print("@@@@@@@@@-----start editing-----@@@@@@@@@")

        edited_model, weights_copy = None, None

        if self.alg_name == 'LORAFT':
            # 合并lora权重到原始模型变成edited_model
            edited_model = PeftModel.from_pretrained(self.model, self.hparams.lora_model_name)
            for i, request in tqdm(enumerate(requests)):
                try:
                    all_metrics[i].update({
                                'case_id': i,
                                "requested_rewrite": request,
                                "post": compute_edit_quality(edited_model, self.model_name, self.hparams, 
                                                            self.tok, request, self.hparams.device, edit_lang=source_lang, test_langs=test_langs),
                            })
                except:
                    all_metrics[i].update({
                                'case_id': i,
                                "requested_rewrite": request,
                                "post": "error"
                            })
                torch.cuda.empty_cache()
                json.dump(all_metrics, open(kwargs["res_path"], 'w'), ensure_ascii=False, indent=4)
                
            return all_metrics, edited_model, weights_copy

        

        for i, request in tqdm(enumerate(requests)):
            start = time()

            if "post" in  all_metrics[i] and all_metrics[i]["post"] != "error": continue
            if  all_metrics[i]["pre"] == "error": continue

            if self.alg_name == 'IKE':
                prepare_request = request.copy()
                prepare_request["target_new"] = prepare_request["target_new_%s"%source_lang]

                assert 'train_ds' in kwargs.keys() or print('IKE need train_ds(For getting In-Context prompt)')
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    prepare_request,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds']
                )
                exec_time = time() - start
                # LOG.info(f"Execution {i} editing took {exec_time}")
                start = time()
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                     request, self.hparams.device, source_lang=source_lang),
                })
                # all_metrics[i]['pre'].pop('locality')

                # LOG.info(f"Evaluation took {time() - start}")

                # if verbose:
                #     LOG.info(
                #         f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                #     )

            elif self.alg_name == 'KEY' or self.alg_name == 'MKEY' \
                or self.alg_name == 'MVALUE' or self.alg_name == 'MKEYPG':
                edited_model, weights_copy = None, None
                prepare_request = copy.deepcopy(request)
                if not os.path.exists(kwargs["saved_keys_path"]):
                    os.makedirs(kwargs["saved_keys_path"])

                # try:
                exec_time = time() - start
                # LOG.info(f"Execution {i} editing took {exec_time}")

                if os.path.exists((kwargs["saved_keys_path"] + "/case_" + str(i) + "_delta-key.pt")):
                    delta_key = torch.load((kwargs["saved_keys_path"] + "/case_" + str(i) + "_delta-key.pt"))
                    init_key = torch.load((kwargs["saved_keys_path"] + "/case_" + str(i) + "_init-key.pt"))
                else:
                    delta_key, init_key  = self.apply_algo(
                        self.model,
                        self.tok,
                        prepare_request,
                        self.hparams,
                        copy=False,
                        return_orig_weights=True,
                        keep_original_weight=keep_original_weight,
                        saved_keys_path=kwargs["saved_keys_path"],
                        neurons_path=kwargs["neurons_path"],
                        case_id=i,
                    )
                    torch.save(delta_key, (kwargs["saved_keys_path"] + "/case_" + str(i) + "_delta-key.pt"))
                    torch.save(init_key, (kwargs["saved_keys_path"] + "/case_" + str(i) + "_init-key.pt"))
                start = time()
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_key_edit_quality(delta_key, self.model, self.model_name, self.hparams, 
                                            self.tok, request, self.hparams.device, edit_lang=source_lang, test_langs=test_langs),
                    })

                # except:
                #     all_metrics[i].update({
                #         'case_id': i,
                #         "requested_rewrite": request,
                #         "post": "error"
                #     })
                #     edited_model = "error"
                #     weights_copy = "error"

                torch.cuda.empty_cache()
            else:
                prepare_request = copy.deepcopy(request)

                try:
                    edited_model, weights_copy = self.apply_algo(
                        self.model,
                        self.tok,
                        [prepare_request],
                        self.hparams,
                        copy=False,
                        return_orig_weights=True,
                        keep_original_weight=keep_original_weight,
                        train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                    )
                    exec_time = time() - start
                    # LOG.info(f"Execution {i} editing took {exec_time}")

                    start = time()
                    all_metrics[i].update({
                        'case_id': i,
                        "requested_rewrite": request,
                        "time": exec_time,
                        "post": compute_edit_quality(edited_model, self.model_name, self.hparams, 
                                                    self.tok, request, self.hparams.device, edit_lang=source_lang, test_langs=test_langs),
                    })
                    if self.alg_name == 'KN':
                        with torch.no_grad():
                            weights_copy() # unpatch_fn
                    else:
                        with torch.no_grad():
                            for k, v in weights_copy.items():
                                nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
                    

                except:
                    all_metrics[i].update({
                        'case_id': i,
                        "requested_rewrite": request,
                        "post": "error"
                    })
                    edited_model = "error"
                    weights_copy = "error"

                torch.cuda.empty_cache()
                
            json.dump(all_metrics, open(kwargs["res_path"], 'w', encoding="utf-8"), ensure_ascii=False, indent=4)
           

        return all_metrics, edited_model, weights_copy
    


    def batch_edit(self,
                   prompts: List[str],
                   target_new_en: List[str],
                   target_new_zh: List[str],
                   ground_truth: Optional[List[str]] = None,
                   rephrase_prompts_en: Optional[List[str]] = None,
                   rephrase_prompts_zh: Optional[List[str]] = None,
                   locality_prompts_en: Optional[List[str]] = None,
                   locality_prompts_zh: Optional[List[str]] = None,
                   locality_ground_truth: Optional[List[str]] = None,
                   keep_original_weight=False,
                   verbose=True,
                   **kwargs
                   ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new_en)
        assert len(prompts) == len(target_new_zh)

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


        assert BatchEditor.is_batchable_method(self.alg_name) \
               or print(f'The Method {self.alg_name} can not batch edit examples.')

        requests = self._prepare_requests(prompts, target_new_en, target_new_zh, ground_truth, rephrase_prompts_en, rephrase_prompts_zh,
                                          locality_prompts_en, locality_prompts_zh, locality_ground_truth, **kwargs)

        assert hasattr(self.hparams, 'batch_size') or \
               print(f'Method {self.alg_name} found, pls specify the batch_size....')

        for record_chunks in self._chunks(requests, self.hparams.batch_size):
            start = time()

            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                record_chunks,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=keep_original_weight,
            )
            exec_time = time() - start
            # LOG.info(f"Execution editing took {exec_time}")


            start = time()
            all_metrics = []
            for i, request in enumerate(record_chunks):

                metrics = {
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device),
                }

                all_metrics.append(metrics)

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            for i, request in enumerate(record_chunks):
                all_metrics[i]["pre"] = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device)

            #     if verbose:
            #         LOG.info(
            #             f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
            #         )

            # LOG.info(f"Evaluation took {time() - start}")


        return all_metrics, edited_model, weights_copy

    def edit_dataset(self,
                     ds: Dataset,
                     keep_original_weight=False,
                     verbose=True
                     ):
        # Make Sure dataset supported
        assert sum([isinstance(ds, ds_in_dict) for ds_in_dict in DS_DICT.values()]) > 0 \
        or print(f'DataSet {ds} not supported yet.')

        is_singleton = SingletonEditor.is_singleton_method(self.alg_name)



        if is_singleton:
            num_edits = 1 # Single editor method found
        else:
            assert hasattr(self.hparams, 'batch_size') or \
                   print(f'Method {self.alg_name} found, pls set the batch_size correctly')

            num_edits = self.hparams.batch_size

        all_metrics = []

        for record_chunks in tqdm(self._chunks(ds, num_edits), desc='Editing dataset', total=len(ds)/num_edits):

            start = time()
            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                record_chunks,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=keep_original_weight
            )
            exec_time = time() - start
            # LOG.info(f"Execution took {exec_time}")

            start = time()
            all_metrics = []
            for i, request in enumerate(record_chunks):

                metrics = {
                    'case_id': request['case_id'],
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device),
                }
                all_metrics.append(metrics)

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            for i, request in enumerate(record_chunks):
                all_metrics[i]["pre"] = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                                                      self.hparams.device)

            #     if verbose:
            #         LOG.info(
            #             f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
            #         )

            # LOG.info(f"Evaluation took {time() - start}")

        return all_metrics, edited_model, weights_copy




    def _chunks(self, arr, n):
        """Yield successive n-sized chunks from arr."""
        for i in range(0, len(arr), n):
            yield arr[i: i + n]



    # def _prepare_requests(self,
    #                     prompts_edits, 
    #                     targets_new,
    #                     ground_truth, 
    #                     rephrase_prompts,
    #                     locality_inputs, 
    #                     portability_inputs,
    #                     source_lang, 
    #                     test_langs,
    #                     **kwargs
    #                       ):

    #     requests = []
    #     requests = [{
    #         'prompt': prompt,
    #         'target_new_en': target_new_en_,
    #         'target_new_zh': target_new_zh_,
    #         'ground_truth': ground_truth_,
    #         'portability_en': {},
    #         'portability_zh': {},
    #         'locality_en': {},
    #         'locality_zh': {}
    #     }
    #     for prompt, ground_truth_, target_new_en_, target_new_zh_ in zip(prompts, ground_truth, target_new_en, target_new_zh)
    #     ]

    #     data_num = len(prompts_edits[source_lang])

    #     for _i in range(data_num):
    #         cur_request = {}
    #         cur_request["prompt"] = prompts_edits[][_i]



    #     if 'subject' in kwargs:
    #         if isinstance(kwargs['subject'], str):
    #             kwargs['subject'] = [kwargs['subject'],]
    #         else:
    #             assert len(kwargs['subject']) == len(prompts)
    #         for prompt_, subject_ in zip(prompts, kwargs['subject']):
    #             assert subject_ in prompt_ or print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

    #         for i, request in enumerate(requests):
    #             request.update(
    #                 {
    #                     'subject': kwargs['subject'][i]
    #                 }
    #             )

    #     if rephrase_prompts_en is not None:
    #         if isinstance(rephrase_prompts_en, str):
    #             rephrase_prompts_en = [rephrase_prompts_en,]

    #         for i, request in enumerate(requests):
    #             request.update(
    #                 {
    #                     'rephrase_prompt_en': rephrase_prompts_en[i],
    #                 }
    #             )

    #     if rephrase_prompts_zh is not None:
    #         if isinstance(rephrase_prompts_zh, str):
    #             rephrase_prompts_zh = [rephrase_prompts_zh,]

    #         for i, request in enumerate(requests):
    #             request.update(
    #                 {
    #                     'rephrase_prompt_zh': rephrase_prompts_zh[i],
    #                 }
    #             )

    #     if locality_inputs_en is not None:
    #         for locality_key in locality_inputs_en.keys():
    #             if isinstance(locality_inputs_en[locality_key]['prompt'], str):
    #                 locality_inputs_en[locality_key]['prompt'] = [locality_inputs_en[locality_key]['prompt'],]
    #                 locality_inputs_en[locality_key]['ground_truth'] = [locality_inputs_en[locality_key]['ground_truth'], ]
    #             assert len(locality_inputs_en[locality_key]['prompt']) == len(locality_inputs_en[locality_key]['ground_truth']) \
    #             == len(requests) or print('One Edit instance needs one locality input.....')

    #             for i, request in enumerate(requests):
    #                 request['locality_en'].update(
    #                     {
    #                         locality_key: {
    #                             f'prompt': locality_inputs_en[locality_key]['prompt'][i],
    #                             f'ground_truth': locality_inputs_en[locality_key]['ground_truth'][i]
    #                         }
    #                     }
    #                 )

    #     if locality_inputs_zh is not None:
    #         for locality_key in locality_inputs_zh.keys():
    #             if isinstance(locality_inputs_zh[locality_key]['prompt'], str):
    #                 locality_inputs_zh[locality_key]['prompt'] = [locality_inputs_zh[locality_key]['prompt'],]
    #                 locality_inputs_zh[locality_key]['ground_truth'] = [locality_inputs_zh[locality_key]['ground_truth'], ]
    #             assert len(locality_inputs_zh[locality_key]['prompt']) == len(locality_inputs_zh[locality_key]['ground_truth']) \
    #             == len(requests) or print('One Edit instance needs one locality input.....')

    #             for i, request in enumerate(requests):
    #                 request['locality_zh'].update(
    #                     {
    #                         locality_key: {
    #                             f'prompt': locality_inputs_zh[locality_key]['prompt'][i],
    #                             f'ground_truth': locality_inputs_zh[locality_key]['ground_truth'][i]
    #                         }
    #                     }
    #                 )

    #     if portability_inputs_en is not None:
    #         for portability_key in portability_inputs_en.keys():
    #             if isinstance(portability_inputs_en[portability_key]['prompt'], str):
    #                 portability_inputs_en[portability_key]['prompt'] = [portability_inputs_en[portability_key]['prompt'],]
    #                 portability_inputs_en[portability_key]['ground_truth'] = [portability_inputs_en[portability_key]['ground_truth'], ]
    #             assert len(portability_inputs_en[portability_key]['prompt']) == len(portability_inputs_en[portability_key]['ground_truth']) \
    #             == len(requests) or print('One Edit instance needs one portability input.....')

    #             for i, request in enumerate(requests):
    #                 request['portability_en'].update(
    #                     {
    #                         portability_key: {
    #                             'prompt': portability_inputs_en[portability_key]['prompt'][i],
    #                             'ground_truth': portability_inputs_en[portability_key]['ground_truth'][i]
    #                         }
    #                     }
    #                 )

    #     if portability_inputs_zh is not None:
    #         for portability_key in portability_inputs_zh.keys():
    #             if isinstance(portability_inputs_zh[portability_key]['prompt'], str):
    #                 portability_inputs_zh[portability_key]['prompt'] = [portability_inputs_zh[portability_key]['prompt'],]
    #                 portability_inputs_zh[portability_key]['ground_truth'] = [portability_inputs_zh[portability_key]['ground_truth'], ]
    #             assert len(portability_inputs_zh[portability_key]['prompt']) == len(portability_inputs_zh[portability_key]['ground_truth']) \
    #             == len(requests) or print('One Edit instance needs one portability input.....')

    #             for i, request in enumerate(requests):
    #                 request['portability_zh'].update(
    #                     {
    #                         portability_key: {
    #                             'prompt': portability_inputs_zh[portability_key]['prompt'][i],
    #                             'ground_truth': portability_inputs_zh[portability_key]['ground_truth'][i]
    #                         }
    #                     }
    #                 )

    #     return requests


# if __name__ == "__main__":
#
#     editor = BaseEditor(alg_name='KN', model_name='/nature/peng/serac/hugging_cache/t5-3b-finetuned-counterfact-10000', hparams_fname='t5-3b.json')
#
#     editor.edit(
#         prompts='What university did Watts Humphrey attend?',
#         ground_truth='Illinois Institute of Technology',
#         target_new='University of Michigan'
#     )
#
#     metrics, edited_model, _ = editor.edit(prompts='What university did Watts Humphrey attend?', ground_truth='Illinois Institute of Technology', target_new='University of Michigan')



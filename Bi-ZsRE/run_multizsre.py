import os.path
import sys
sys.path.append('..')
import json
import random
from easyeditor import (FTHyperParams, IKEHyperParams, KNHyperParams, 
                MEMITHyperParams, ROMEHyperParams, MENDHyperParams, 
                SERACHparams, KEYHyperParams, PMETHyperParams, 
                LORAFTHyperParams, REMAKEHyperParams)
from easyeditor import BaseEditor


import argparse
from collections import defaultdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str, help="only support ROME/MEMIT/KEY")
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--res_save_dir', default='', type=str)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--source_lang", type=str, default="en")
    parser.add_argument('--test_langs', type=str, nargs='+', help='<Required> Set flag', required=True, default=["en", "zh"])
    parser.add_argument("--backbone", type=str, default="llama2_7b")
    parser.add_argument("--res_path", type=str, default="")
    parser.add_argument("--saved_keys_path", type=str, default="")
    parser.add_argument("--neurons_path", type=str, default="")
    parser.add_argument("--pre_res_path", type=str, default="")



    args = parser.parse_args()

    # if args.editing_method == 'FT':
    #     editing_hparams = FTHyperParams
    # elif args.editing_method == 'IKE':
    #     editing_hparams = IKEHyperParams
    if args.editing_method == 'MEMIT' or args.editing_method == 'MMEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME' or args.editing_method == 'MROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'PMET' or args.editing_method == 'MPMET':
        editing_hparams = PMETHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == "LORAFT":
        editing_hparams = LORAFTHyperParams
    elif args.editing_method == "REMAKE":
        editing_hparams = REMAKEHyperParams
    # elif args.editing_method == 'SERAC':
    #     editing_hparams = SERACHparams
    # elif args.editing_method == 'MEND':
    #     editing_hparams = MENDHyperParams
    elif args.editing_method == 'KEY' or args.editing_method == 'MKEY' \
        or args.editing_method == 'MVALUE' or args.editing_method == 'MKEYPG':
        editing_hparams = KEYHyperParams
    else:
        raise NotImplementedError
    

    if "Bi-ZsRE-data" in args.data_dir:
        with open(args.data_dir + "/bizsre_test-add_loc.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
    if "MzsRE-data" in args.data_dir:
        with open(args.data_dir + "/mzsre_test-fix.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
    print("!!! data num:", len(test_data))
    print("!!! test langs:", args.test_langs)
    for tmp in args.test_langs:
        assert tmp in list(test_data[0].keys()), "The input data do not have the assigned language."
    if args.ds_size is not None:
        # test_data = random.sample(test_data, args.ds_size)
        test_data = test_data[:args.ds_size]



    requests = []
    for _item in test_data:
        cur_request = {}
        cur_request["source_lang"] = args.source_lang
        cur_request["test_langs"] = args.test_langs
        for i_lang in args.test_langs:
            cur_request[i_lang] = {}
            cur_request[i_lang]["prompt"] = _item[i_lang]["src"]
            cur_request[i_lang]["target_new"] = _item[i_lang]["alt"]
            cur_request[i_lang]["subject"] = _item[i_lang]["subject"]
            cur_request[i_lang]["ground_truth"] = '<|endoftext|>'
            cur_request[i_lang]["rephrase_prompt"] = _item[i_lang]["rephrase"]
            cur_request[i_lang]["locality"] = {"neighborhood":{"prompt": _item[i_lang]["loc"], "ground_truth": _item[i_lang]["loc_ans"]}}
            cur_request[i_lang]["portability"] = {"one_hop": {"prompt": _item[i_lang]["portability"]["New Question"], "ground_truth": _item[i_lang]["portability"]["New Answer"]}}
            if "Bi-ZsRE-data" in args.data_dir:
                cur_request[i_lang]["loc_sub_list"] = _item[i_lang]["loc_sub_list"]
        requests.append(cur_request)
    

    hparams = editing_hparams.from_hparams(args.hparams_dir)
    print("!!!! load editor...")
    editor = BaseEditor.from_hparams(hparams)
    

    print("&&&&&   start editing!   &&&&&")

    if args.res_path == "":
        res_path = f'your_path/Bi-ZsRE/results/{args.backbone}_{args.editing_method}_edit_{args.source_lang}_results.json'
    else:
        res_path = args.res_path

    
    metrics, edited_model, _ = editor.edit(
        requests,
        keep_original_weight=True, # 设置为True的时候代表不是序列式地edit，设置为False的时候代表序列式地edit，每次edit完都会改变参数
        source_lang=args.source_lang,
        test_langs=args.test_langs,
        saved_keys_path=args.saved_keys_path,
        neurons_path=args.neurons_path,
        pre_res_path=args.pre_res_path,
        res_path = res_path,
    )
    
    if args.res_path == "":
        json.dump(metrics, open(res_path, 'w'), ensure_ascii=False, indent=4)
    else:
        json.dump(metrics, open(res_path, "w"), ensure_ascii=False, indent=4)

    

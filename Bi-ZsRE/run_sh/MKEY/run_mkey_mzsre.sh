layer=?

method=MKEY
source_lang=cz_vi_tr_fr_es_zh_en_de_ru_du_pt_th
CUDA_VISIBLE_DEVICES=0 /opt/anaconda3/bin/python -u your_path/Bi-ZsRE/run_multizsre.py \
    --editing_method=${method} \
    --hparams_dir=your_path/Bi-ZsRE/hparams/${method}/xxx-mzsre.yaml \
    --data_dir=your_path/Bi-ZsRE/data/MzsRE-data \
    --source_lang=${source_lang} \
    --test_langs cz vi tr fr es zh en de ru du pt th \
    --layers ${layer} \
    --saved_keys_path=your_path/Bi-ZsRE/results-MzsRE/saved-keys/M-batch \
    --pre_res_path=your_path/Bi-ZsRE/results-MzsRE/xxx_MKEYPG_edit_M_results-3.json \
    --res_path=your_path/Bi-ZsRE/results-MzsRE/xxx_${method}_edit_M_results-4.json \
    --backbone xxx >> your_path/Bi-ZsRE/logs/${method}-M-2.txt 2>&1 

# your/Bi-ZsRE/easyeditor/models/key/key_main.py  apply_multikey_to_model_batch()

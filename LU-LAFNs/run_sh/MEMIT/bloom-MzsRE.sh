export HF_HOME=your_path/.cache
method=MMEMIT
log_path=your_path/Bi-ZsRE/logs-new/${method}
if [ -d "${log_path}" ]; then
    echo "log Directory exists: ${log_path}"
else
    mkdir -p "${log_path}"
    chmod -R 777 "${log_path}"
    echo "log Directory created: ${log_path}"
fi
res_path=your_path/Bi-ZsRE/results-new/${method}
if [ -d "${res_path}" ]; then
    echo "res Directory exists: ${res_path}"
else
    mkdir -p "${res_path}"
    chmod -R 777 "${res_path}"
    echo "res Directory created: ${res_path}"
fi

source_lang=cz_vi_tr_fr_es_zh_en_de_ru_du_pt_th
backbone=bloomz-7b1-mt
CUDA_VISIBLE_DEVICES=0 /opt/anaconda3/bin/python -u your_path/Bi-ZsRE/run_multizsre.py \
    --editing_method=${method} \
    --hparams_dir=your_path/Bi-ZsRE/hparams/${method}/${backbone}-mzsre.yaml \
    --data_dir=your_path/Bi-ZsRE/data/MzsRE-data \
    --source_lang=${source_lang} \
    --test_langs cz vi tr fr es zh en de ru du pt th \
    --pre_res_path=your_path/Bi-ZsRE/results-new/bloomz-7b1-mt-MzsRE-pre_res.json \
    --res_path=${res_path}/${backbone}_${method}_edit_mzsre_results.json \
    --backbone=${backbone}  >> ${log_path}/${backbone}-${method}-mzsre.txt 2>&1 


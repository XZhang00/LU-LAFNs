export HF_HOME=your_path/.cache
method=MKEYPG

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

source_lang=en_zh
backbone=llama3.1-8b
layer=11_12_13_31
thres_array=(0.1)


for thres in ${thres_array[@]};
do
    CUDA_VISIBLE_DEVICES=0 /opt/anaconda3/bin/python -u your_path/Bi-ZsRE/run_multizsre.py \
        --editing_method=${method} \
        --hparams_dir=your_path/Bi-ZsRE/hparams/${method}/${backbone}.yaml \
        --data_dir=your_path/Bi-ZsRE/data/Bi-ZsRE-data \
        --source_lang=${source_lang} \
        --test_langs en zh \
        --layers ${layer} \
        --saved_keys_path=${res_path}/saved-keys-${backbone}/layer-${layer}-thres-${thres} \
        --res_path=${res_path}/${backbone}_${method}_edit_${source_lang}_results-pgs-layer-${layer}-thres-${thres}.json \
        --pre_res_path=your_path/Bi-ZsRE/results-new/meta-llama-3.1-8b-BiZsRE-pre_res.json \
        --neurons_path=your_path/Neurons/results_neurons/pgs_neurons/${backbone}-inter-neurons-${thres}-layer-${layer}.json \
        --backbone ${backbone} >> ${log_path}/${backbone}-${method}-${source_lang}-pgs-layer-${layer}-thres-${thres}.txt 2>&1 
    wait
done





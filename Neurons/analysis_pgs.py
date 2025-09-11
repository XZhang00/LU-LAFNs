import json
import torch
import pickle, os
from numpy import mean


act_path = "your_path/Neurons/results_activations/pgs_results"
neurons_path = "your_path/Neurons/results_neurons/pgs_neurons"

if not os.path.exists(neurons_path):
    os.makedirs(neurons_path)

neurons_dict = {}
inter_dict = {}

# the setting for each model
cur_lang = "en"
layers = [11, 12, 13, 31]
threshold = 0.1
backbone = "llama3.1-8b"

cur_lang = "en"
layers = [19, 20, 21, 27]
threshold = 0.0
backbone = "qwen2-7b"

cur_lang = "en"
layers = [9, 10, 11, 29]
threshold = 0.2
backbone = "bloomz-7b1-mt"

all_inter_nums = []

for id in range(1037):
    neurons_dict[id] = {}
    inter_dict[id] = {}
    for i in layers:
        neurons_dict[id][i] = {}
        for i_lang in [cur_lang]:
            activation = torch.load(f"{act_path}/activation.case_{str(id)}-{i_lang}.{backbone}")
            tmp = activation["over_zero"][i] / activation["n"]
            mask = tmp.gt(tmp.max() * threshold)
            # print(mask.sum().item())
            # print(i, id, i_lang, activation["n"], activation["over_zero"][i].max(), tmp.max(), mask.sum())
            # print(id, i, i_lang, activation["n"], mask.sum())
            neurons_dict[id][i][i_lang] = torch.nonzero(mask == True).squeeze().tolist()
        inter_dict[id][i] = neurons_dict[id][i][cur_lang]
        all_inter_nums.append(len(inter_dict[id][i]))
        # if id % 100 == 0 and i in layers:
        print(id, i, len(inter_dict[id][i]))
# json.dump(neurons_dict, open(f"{neurons_path}/neurons-{threshold}.json", 'w', encoding="utf-8"), indent=4) 
layers  = [str(j) for j in layers]
json.dump(inter_dict, open(f"{neurons_path}/{backbone}-{cur_lang}-neurons-{threshold}-layer-{'_'.join(layers)}.json", 'w', encoding="utf-8"), indent=4) 

print("@@@", threshold, len(all_inter_nums), mean(all_inter_nums))

print("+++ Done!")





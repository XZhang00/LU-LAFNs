import json
import os
from tqdm import tqdm
from numpy import abs


from transformers import LlamaTokenizer, AutoTokenizer, Qwen2ForCausalLM, LlamaForCausalLM, BloomForCausalLM
tokenizer = AutoTokenizer.from_pretrained("your_path/Models/meta-llama/Meta-Llama-3.1-8B")

def obtain_f1_and_em(a,b):
    global tokenizer
    
    a_words = tokenizer.encode(a, add_special_tokens=False)
    b_words = tokenizer.encode(b, add_special_tokens=False)
    # if len(a_words) > 1 and len(b_words) > 1:
    #     if a_words[0] == 1 or b_words[0] == 1:
    #         print("@@@", a, b, a_words[0], b_words[0])
        # assert a_words[0] != 1 and b_words[0] != 1
    if len(a_words) == 0 and len(b_words) == 0:
        return 1.0, 1
    if len(a_words) == 0 or len(b_words) == 0:
        return 0.0, 0

    em = 1 if a == b else 0
    k = len(a_words) * len(b_words)

    intersecting_words = []
    for word in a_words.copy():
        if word in b_words:
            a_words.remove(word)
            b_words.remove(word)
            intersecting_words.append(word)

    f1_score = (len(intersecting_words) * len(intersecting_words)) / float(k)
    return f1_score, em


def my_avg(a):
    return round(sum(a)*100 / float(len(a)), 2)
    
def calculate_metrics(file_root, test_lang="en", num=1037):
    with open(file_root, "r", encoding="utf-8") as f:
        data = json.load(f)[:num]

    i_lang = test_lang
    reliablilty_f1_list = []
    reliablilty_em_list = []

    generalization_f1_list = []
    generalization_em_list = []

    locality_f1_list = []
    locality_em_list = []

    locality_f1_list_sub_pre= []
    locality_em_list_sub_pre = []

    locality_f1_list_sub = []
    locality_em_list_sub = []

    portablility_f1_list = []
    portablility_em_list = []

    for item in tqdm(data):
        if item["pre"] == "error": continue
        if "post" not in item or item["post"] == "error": continue
        reliablilty_f1, reliablilty_em = obtain_f1_and_em(item["post"][i_lang]["rewrite_acc"]["ans"], 
                                                            item["post"][i_lang]["rewrite_acc"]["target"])
        reliablilty_f1_list.append(reliablilty_f1)
        reliablilty_em_list.append(reliablilty_em)

        generalization_f1, generalization_em = obtain_f1_and_em(item["post"][i_lang]["rephrase_acc"]["ans"], 
                                                                item["post"][i_lang]["rephrase_acc"]["target"])
        generalization_f1_list.append(generalization_f1)
        generalization_em_list.append(generalization_em)


        locality_f1, locality_em = obtain_f1_and_em(item["post"][i_lang]["locality"]["neighborhood_output"]["ans"], 
                                                    item["pre"][i_lang]["locality"]["neighborhood_output"]["ans"])
        
        locality_f1_list.append(locality_f1)
        locality_em_list.append(locality_em)



        portablility_f1, portablility_em = obtain_f1_and_em(item["post"][i_lang]["portability"]["one_hop_acc"]["ans"], 
                                                            item["post"][i_lang]["portability"]["one_hop_acc"]["target"])
        portablility_f1_list.append(portablility_f1)
        portablility_em_list.append(portablility_em)

    print(file_root.split("/")[-1])
    print("test-num: ", len(reliablilty_f1_list), len(locality_f1_list_sub))
    print("F1 score >>>> test-lang: ", i_lang, my_avg(reliablilty_f1_list), "\t", my_avg(generalization_f1_list), "\t", my_avg(locality_f1_list), "\t", my_avg(portablility_f1_list))
    print("EM Score >>>>>>>>>", "test-lang: ", i_lang, my_avg(reliablilty_em_list), "\t", my_avg(generalization_em_list), "\t", my_avg(locality_em_list), "\t", my_avg(portablility_em_list))
    




if __name__ == "__main__":
    

    path = "your_path/Bi-ZsRE/results-new/MKEYPG/qwen2-7b_MKEYPG_edit_en_results-pgs-layer-20-thres-0.0.json"
    print(path)
    num = 1037

    test_lang = "en"
    calculate_metrics(path, test_lang, num=num)

    test_lang = "zh"
    calculate_metrics(path, test_lang, num=num)

    print("\n")
    


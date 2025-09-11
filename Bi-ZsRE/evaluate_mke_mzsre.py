import json
import os
from tqdm import tqdm


from transformers import LlamaTokenizer, LlamaForCausalLM, GemmaForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("your_path/Models/meta-llama/Meta-Llama-3.1-8B")

def obtain_f1_and_em(a,b):
    global tokenizer
    
    a_words = tokenizer.encode(a, add_special_tokens=False)
    b_words = tokenizer.encode(b, add_special_tokens=False)
    # if len(a_words) > 1 and len(b_words) > 1:
    #     assert a_words[0] != 1 and b_words[0] != 1
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
    
def calculate_metrics(file_root, test_lang="en", num=743):
    with open(file_root, "r", encoding="utf-8") as f:
        data = json.load(f)[:num]

    i_lang = test_lang
    reliablilty_f1_list = []
    reliablilty_em_list = []

    generalization_f1_list = []
    generalization_em_list = []

    locality_f1_list = []
    locality_em_list = []

    portablility_f1_list = []
    portablility_em_list = []

    for item in tqdm(data):
        if "post" not in item or item["post"] == "error": continue
        if item["pre"] == "error": continue
        if item["post"][i_lang] == "error": continue
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

    # print(reliablilty_f1_list, generalization_f1_list, locality_f1_list, portablility_f1_list)
    return [my_avg(reliablilty_f1_list), my_avg(generalization_f1_list), my_avg(locality_f1_list), my_avg(portablility_f1_list)], [my_avg(reliablilty_em_list), my_avg(generalization_em_list), my_avg(locality_em_list), my_avg(portablility_em_list)]


if __name__ == "__main__":

    langs = ['cz', 'vi', 'tr', 'fr', 'es', 'zh', 'en', 'de', 'ru', 'du', 'pt', 'th']
    
    num = 743

    reliablilty_res_f1 = []
    generalization_res_f1 = []
    locality_res_f1 = []
    portablility_res_f1 = []

    reliablilty_res_em = []
    generalization_res_em = []
    locality_res_em = []
    portablility_res_em = []

    cur_model = "llama3.1-8b"
    cur_model = "bloomz-7b1-mt"
    # cur_model = "qwen2-7b"

    method = "LORAFT"

    path = "your_path/Bi-ZsRE/results-new/MKEYPG/bloomz-7b1-mt_MKEYPG_edit_mzsre_results-pgs-layer-9_10_11_29.json"

    for i_lang in langs:
        
        f1_res, em_res = calculate_metrics(path, i_lang, num=num)

        assert len(f1_res) == len(em_res) == 4

        reliablilty_res_f1.append(str(f1_res[0]))
        generalization_res_f1.append(str(f1_res[1]))
        locality_res_f1.append(str(f1_res[2]))
        portablility_res_f1.append(str(f1_res[3]))

        reliablilty_res_em.append(str(em_res[0]))
        generalization_res_em.append(str(em_res[1]))
        locality_res_em.append(str(em_res[2]))
        portablility_res_em.append(str(em_res[3]))

    assert len(reliablilty_res_em) == len(generalization_res_f1) == len(langs)

    print(path)
    print(path.split("/")[-1])
    print("***** F1 score *****")
    print("\t".join(reliablilty_res_f1))
    print("\t".join(generalization_res_f1))
    print("\t".join(locality_res_f1))
    print("\t".join(portablility_res_f1))
    print("***** EM Score *****")
    print("\t".join(reliablilty_res_em))
    print("\t".join(generalization_res_em))
    print("\t".join(locality_res_em))
    print("\t".join(portablility_res_em))

    print("***** F1 score to overleaf *****")
    print(" & ".join(reliablilty_res_f1))
    print(" & ".join(generalization_res_f1))
    print(" & ".join(locality_res_f1))
    print(" & ".join(portablility_res_f1))

    print("***** EM score to overleaf *****")
    print(" & ".join(reliablilty_res_em))
    print(" & ".join(generalization_res_em))
    print(" & ".join(locality_res_em))
    print(" & ".join(portablility_res_em))

    print("\n")

import torch, transformers
from transformers import LlamaForCausalLM, LlamaModel, LlamaTokenizer, Qwen2ForCausalLM, Qwen2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import argparse, json
from types import MethodType
import os
from torch.nn import functional as F


def main(args):
    is_llama = None
    is_qwen = None
    is_bloom = None

    if "Llama" in args.model:
        is_llama = True
    if "Qwen" in args.model:
        is_qwen = True
    if "bloom" in args.model:
        is_bloom = True

    if is_llama:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        num_layers = model.config.num_hidden_layers
        intermediate_size = model.config.intermediate_size
        print("llama3.1-8b", num_layers, intermediate_size)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if is_qwen:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        num_layers = model.config.num_hidden_layers
        intermediate_size = model.config.intermediate_size
        print("qwen2-7b", num_layers, intermediate_size)
    if is_bloom:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        num_layers = model.config.n_layer
        intermediate_size = model.config.hidden_size * 4
        print("bloomz-7b1-mt", num_layers, intermediate_size)

    torch.cuda.memory_summary()

    global sum1, over_zero

    sum1 = torch.zeros(num_layers, intermediate_size).to('cuda')
    over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')


    def factory(idx):
        def new_forward(self, x):
            activation = self.act_fn(self.gate_proj(x))
            down_proj = self.down_proj(activation * self.up_proj(x))  # 原来模型的返回值
            # print("@@@", activation.size(), activation.sum(dim=(0,1)).size())
            sum1[idx, :] += activation.sum(dim=(0,1))
            over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
            return down_proj

        def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
            out = F.dropout(x, p=prob, training=training)
            out = residual + out
            return out
        
        def bloom_forward(self, x, residual):
            activation = self.gelu_impl(self.dense_h_to_4h(x))
            # print("@@@", activation.size(), activation.sum(dim=(0,1)).size())
            intermediate_output = self.dense_4h_to_h(activation)
            output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)
            sum1[idx, :] += activation.sum(dim=(0,1))
            over_zero[idx, :] += (activation > 0).sum(dim=(0,1))

            return output

        if is_llama or is_qwen:
            return new_forward
        if is_bloom:
            return bloom_forward
    

    # if is_llama:
    for i in range(num_layers):
        if is_llama or is_qwen:
            obj = model.model.layers[i].mlp
            obj.forward = MethodType(factory(i), obj)
        if is_bloom:
            obj = model.transformer.h[i].mlp
            obj.forward = MethodType(factory(i), obj)

    data = json.load(open(args.data_path))
    if not os.path.exists(args.res_path):
        os.makedirs(args.res_path)

    for id, item in enumerate(data):
        for i_lang in ["en", "zh"]:
            # if is_llama:
            #     if os.path.exists(f'{args.res_path}/activation.case_{str(id)}-{i_lang}.llama-7b'):
            #         continue
            # if is_qwen:
            #     if os.path.exists(f'{args.res_path}/activation.case_{str(id)}-{i_lang}.qwen-7b'):
            #         continue
            
            print(id, i_lang)
            sum1 = torch.zeros(num_layers, intermediate_size).to('cuda')
            over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
            print("@@@ sum1 & over_zero", sum1.sum(), over_zero.sum())

            # 原句 + 复述句
            # cur_list = [item[i_lang]["subject"]+item[i_lang]["alt"], item[i_lang]["rephrase"]+item[i_lang]["alt"]] + item[i_lang]["pgs"]
            
            # s+r
            # cur_list = [item[i_lang]["subject"]]

            # s+r+o
            cur_list = [item[i_lang]["subject"]+item[i_lang]["alt"]]

            for i in range(len(cur_list)):
                prompt = cur_list[i]
                if prompt == "": continue
                inputs = tokenizer(prompt, return_tensors="pt")
                # print(inputs)
                torch.cuda.empty_cache()
                # if i % 100 == 0:
                #     print(id, i, "@@@", inputs.input_ids.size())
                generate_ids = model.generate((inputs.input_ids).cuda(), do_sample=True, max_new_tokens=1)

            print("@@@!!! after- sum1 & over_zero", sum1.sum(), over_zero.sum())
            output = dict(n=len(cur_list), sum1=sum1.to('cpu'), over_zero=over_zero.to('cpu'))

            if is_llama:
                torch.save(output, f'{args.res_path}/activation.case_{str(id)}-{i_lang}.llama3.1-8b')
            if is_qwen:
                torch.save(output, f'{args.res_path}/activation.case_{str(id)}-{i_lang}.qwen2-7b')
            if is_bloom:
                torch.save(output, f'{args.res_path}/activation.case_{str(id)}-{i_lang}.bloomz-7b1-mt')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="your_path/Models/Qwen/Qwen2-7B")
    parser.add_argument("--data_path", type=str, 
                        default="your_path/Neurons/data/Bi-ZsRE-qwen2-72b-gen-pgs-v1.json")
    parser.add_argument("--res_path", type=str, 
                        default="your_path/Neurons/results_activations/pgs_results")
    args = parser.parse_args()

    main(args)



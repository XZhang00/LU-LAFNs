## LU-LAFNs
Code for COLING-2025 paper "Multilingual Knowledge Editing with Language-Agnostic Factual Neurons".


### 1. Locating the language-agnostic factual neurons

First, you need to generate 30 paraphrases for each question to be edited. (In our paper, we call the Qwen2-72B-instruct API to achieve it.)
Then we obtain the LAFNs for each model. You can download the LAFNs for each model from [Neurons](https://drive.google.com/file/d/1TSnbqestmeZ6vA6DGnkQO3IGIUQpmNNL/view?usp=drive_link).

```
python Neurons/get_activations_pgs.py
python Neurons/analysis_pgs.py
```

### 2. Updating the values of LAFNs

```
bash Bi-ZsRE/run_sh/MKEY/bloom-en_zh.sh
bash Bi-ZsRE/run_sh/MKEY/llama-en_zh.sh
bash Bi-ZsRE/run_sh/MKEY/qwen-en_zh.sh
```

### 3. Evaluation

```
python Bi-ZsRE/evaluate_mke.py
python Bi-ZsRE/evaluate_mke_mzsre.py
```


If you have any questions, please create an issue or email to 23111135@bjtu.edu.cn.


### Acknowledgement
- Our codes are based on [Bi_ZsRE](https://github.com/krystalan/Bi_ZsRE), and we thank their outstanding open-source contributions.
- Our data is based on Bi-ZsRE ([Jiaan Wang et al., 2023](https://arxiv.org/abs/2309.08952)) and MzsRE ([Weixuan Wang et al., 2023](https://arxiv.org/abs/2312.13040))


### Citation
If you find this work is useful, please consider cite our paper:

```
@misc{zhang2024multilingualknowledgeeditinglanguageagnostic,
      title={Multilingual Knowledge Editing with Language-Agnostic Factual Neurons}, 
      author={Xue Zhang and Yunlong Liang and Fandong Meng and Songming Zhang and Yufeng Chen and Jinan Xu and Jie Zhou},
      year={2024},
      eprint={2406.16416},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.16416}, 
}
```

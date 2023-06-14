# LLMRec

## Abstract

LLMRec, a LLM-based recommender system designed for benchmarking LLMs on various recommendation tasks. Specifically, we construct a set of well-designed templates to first transform the recommendation data into textual prompts and then benchmark several popular LLMs on five recommendation tasks, including rating prediction, sequential recommendation, direct recommendation, explanation generation, and review summarization. 

<img src="https://github.com/williamliujl/LLMRec/blob/main/docs/workflow.png" width="860" />

## Prompt Construction

We have used various prompts for off-the-shelf LLMs evaluation, as shown in the following figure.

<img src="https://github.com/williamliujl/LLMRec/blob/main/docs/off-the-shelf_test-prompts.png" width="860" />

To fully assess the recommendation capabilities of LLMs, we conducted task-specific finetuning on various open-source LLMs. We utilized LLMRec's prompt construction module to produce training data for finetuning. The prompt templates are presented below.

<img src="https://github.com/williamliujl/LLMRec/blob/main/docs/finetuing_prompts.png" width="860" />

## Data Acquire
Our prompt data is provided on Google Drive and Baidu Yun.

- Google Drive: https://drive.google.com/drive/folders/1Euz7DAbiWKAwiTErLBnebbUNkCY0Vo4C?usp=sharing

- Baidu Yun: https://pan.baidu.com/s/12mxZWih6D0yJCdw3Jo15Qg 提取码: 9ks9 


## To-Do

- [ ] add off-the-shelf LLMs evaluation scripts

- [ ] add LLMs finetuing scripts



## Citation
Is ChatGPT a Good Recommender? A Preliminary Study
https://arxiv.org/abs/2304.10149

```
@article{liu2023chatgpt,
  title={Is ChatGPT a Good Recommender? A Preliminary Study},
  author={Liu, Junling and Liu, Chao and Lv, Renjie and Zhou, Kang and Zhang, Yan},
  journal={arXiv preprint arXiv:2304.10149},
  year={2023}
}
```

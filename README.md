# The First Few Tokens Are All You Need: An Efficient and Effective Unsupervised Prefix Fine-Tuning Method for Reasoning Models

Are expensive labeled data and rejection sampling truly necessary for developing self-improving reasoning models?

We introduce Unsupervised Prefix Fine-Tuning (UPFT) -- an efficient method that trains models on only the first few tokens of single self-generated solutions, exploiting Prefix Self-Consistency across different reasoning trajectories.
This repo contains the source code of UPFT.

**The First Few Tokens Are All You Need: An Efficient and Effective Unsupervised Prefix Fine-Tuning Method for Reasoning Models** <br>
*Ke Ji, Jiahao Xu, Tian Liang, Qiuzhi Liu, Zhiwei He, Xingyu Chen, Xiaoyuan Liu, Zhijie Wang, Junying Chen, Benyou Wang, Zhaopeng Tu, Haitao Mi, Dong Yu* <br>
Paper: https://arxiv.org/abs/2503.02875 <br>


<p align="center">
<img src="figs/introduction.png" width=80%>
</p>

## Getting Started

```bash
git clone https://github.com/1KE-JI/UPFT.git
cd UPFT
conda create -n torch240 python==3.10.13
conda activate torch240
pip install -r requirements.txt
```

### Training

We utilize open-source framework OpenRLHF to conduct our training process.

Step 1: Use vllm for sampling

Step 2: Run command below to train from a 7B model. 

```bash
bash scripts/train_sft.sh
```

## Citation

```bash
@article{ji2025first,
  title={The first few tokens are all you need: An efficient and effective unsupervised prefix fine-tuning method for reasoning models},
  author={Ji, Ke and Xu, Jiahao and Liang, Tian and Liu, Qiuzhi and He, Zhiwei and Chen, Xingyu and Liu, Xiaoyuan and Wang, Zhijie and Chen, Junying and Wang, Benyou and others},
  journal={arXiv preprint arXiv:2503.02875},
  year={2025}
}
```

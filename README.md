# 先写结论

- xxx

# 发布时间表

- 2023-02-24 Meta AI发布[LLaMA](#llama)
- 2023-03-13 清华数据挖掘研究团队（THUDM）发布[ChatGLM-6B](#chatglm-6b)
- 2023-03-13 Stanford发布[Alpaca](#alpaca)
- 2023-03-14 Eric J. Wang（人肉了下，发现人在Stanford，什么身份不清楚）发布[Alpaca-LoRA](#alpaca-lora)
- 2023-03-19 一个队伍（from UC Berkeley、CMU, Stanford, and UC San Diego）发布[Vicuna](#vicuna)
- 2023-03-25 李煜东（人肉了下，发现是深圳大学博士生）发布[Chinese-ChatLLaMA](#chinese-chatllama)
- 2023-03-27 哈工大科大讯飞联合实验室（HFL）发布 [Chinese-LLaMA-Alpaca](#chinese-llama-alpaca)
- 2023-03-27 香港科技大学统计机器学习实验室（HKUST）发布 [LMFlow](#lmflow)

# 简介

## LLaMA

[代码仓库](https://github.com/facebookresearch/llama)

|   是否开源代码    | 是否开源训练数据 | 是否开源模型参数 |     训练数据大小      |     模型参数大小     |     训练设备      |   训练时长   |
|:-----------:|:--------:|:--------:|:---------------:|:--------------:|:-------------:|:--------:|
| 是（训练代码没有开源） |    是     |    是     | 1T、1T、1.4T、1.4T | 7B、13B、33B、65B | 65B用2048张A100 | 65B用时21天 |

- 13B版在很多测试上优于GPT3（175B）
- 65B版与许多当时的最优模型相比都具有竞争力，比如Chinchilla-70B和PaLM-540B
- 核心思想是说***同样预算***的情况下，用小模型+大量数据，可以得到比大模型+少量数据更好的效果，虽然训练时间更长

## ChatGLM-6B

[代码仓库](https://github.com/THUDM/ChatGLM-6B)

| 是否开源代码  | 是否开源训练数据 | 是否开源模型参数 | 训练数据大小 | 模型参数大小 | 训练设备 | 训练时长 |
|:-------:|:--------:|:--------:|:------:|:------:|:----:|:----:|
| 仅开源微调代码 |    否     |    是     |   1T   | 6B |  未知  |  未知  |

- 模型结构基于该团队自己以前发布的[General Language Model (GLM)](https://github.com/THUDM/GLM)，[论文](https://arxiv.org/abs/2103.10360)
- 集成了该团队自己开发的微调方法[P-Tuning v2](https://github.com/THUDM/P-tuning-v2)，[论文](https://arxiv.org/abs/2110.07602)，方便下游开发者针对自己的应用场景定制模型
- 效果不太行，130B版还在开发，可持续关注

## Alpaca

[代码仓库](https://github.com/tatsu-lab/stanford_alpaca)

|   是否开源代码    | 是否开源训练数据 | 是否开源模型参数 |     训练数据大小      |     模型参数大小     |   训练设备    |  训练时长   |
|:-----------:|:--------:|:--------:|:---------------:|:--------------:|:---------:|:-------:|
| 是 |    是     |    是     | 52K | 7B、13B | 7B用8张A100 | 7B用时3小时 |

- 数据来源特别有趣，是由gpt-3.5（text-davinci-003）生成的训练数据，方法改进自于论文[self-instruct paper](https://arxiv.org/abs/2212.10560)，该论文[开源代码](https://github.com/yizhongw/self-instruct)
- 该研究提到，它获取数据的成本（gpt-3.5生成）小于500美元，云服务器成本小于100美元（对于大多数云服务器商来说）

## Alpaca-LoRA

[代码仓库](https://github.com/tloen/alpaca-lora)

| 是否开源代码 |     是否开源训练数据     | 是否开源模型参数 |     训练数据大小      |     模型参数大小     |     训练设备     |   训练时长   |
|:------:|:----------------:|:--------:|:---------------:|:--------------:|:------------:|:--------:|
|  是  | 是（和Alpaca训练数据相同） |    是     | 52K | 7B、13B、33B、65B | 7B用1张RTX4090 | 7B用时几个小时 |

- 该项目的特点是只用单卡（RTX4090）就可以微调‘小’模型，原因是使用技术[low-rank adaptation (LoRA)](https://arxiv.org/abs/2106.09685)，[开源代码](https://github.com/microsoft/LoRA)，该技术原理大概为，冻结模型的大量参数，只让小部分参数参与微调，所以显著减小显存消耗，该项目使用的LoRA是HuggingFace的[PEFT](https://github.com/huggingface/peft)实现的版本
- 该项目说自己的模型可与Alpaca相媲美甚至更好
- 该项目后续更新了微软研究院的GPT4羊驼数据（[LLaMA-GPT4 dataset](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)），该数据的[论文](https://arxiv.org/abs/2304.03277)

## Vicuna

[代码仓库](https://github.com/lm-sys/FastChat)

| 是否开源代码 |    是否开源训练数据     | 是否开源模型参数 | 训练数据大小 |    模型参数大小    | 训练设备 | 训练时长 |
|:------:|:---------------:|:--------:|:------:|:------------:|:----:|:----:|
|       |                 |          |        |              |      |      |

- 1

## Chinese-ChatLLaMA

[代码仓库](https://github.com/ydli-ai/Chinese-ChatLLaMA)

| 是否开源代码 |     是否开源训练数据     | 是否开源模型参数 | 训练数据大小 |     模型参数大小     | 训练设备 | 训练时长 |
|:------:|:----------------:|:--------:|:------:|:--------------:|:----:|:----:|
|   是    | 是 |    是     |   未知   | 7B |  32 * A100  |  未知  |

- LLaMA在中文上性能不好，作者找来中英平行语料、中文维基、社区互动、新闻数据、科学文献（这些语料应该都是很好的数据）等语料再进行预训练和Alpaca 指令微调得到该模型
- 作者现在的训练方式使用Tencent pretrain框架，可能之后会支持HuggingFace框架
- 作者还在持续收集其他开源中文语料
- 模型效果在多轮对话、逻辑推理、知识问答等场景具有明显缺陷

## Chinese-LLaMA-Alpaca

[代码仓库](https://github.com/ymcui/Chinese-LLaMA-Alpaca)

| 是否开源代码 | 是否开源训练数据 | 是否开源模型参数 |      训练数据大小      | 模型参数大小 |  训练设备   | 训练时长 |
|:-------:|:--------:|:--------:|:----------------:|:--------------:|:-------:|:----:|
|    是（训练代码没有开源）     |    是     |    是     | 预训练20G，指令精调2M、3M |  7B、13B  | 16张A100 |  未知  |

- 基于LLaMA和Alpaca预训练（Pre-training）和指令精调（Instruction Fine-tuning）而来，扩充了中文词表，同样使用LoRA技术
- 开源了两个版本，一个是基于预训练的中文LLaMA大模型，另一个是经过指令精调的中文Alpaca大模型
- 可用个人电脑cpu进行本地部署
- 作者说LLaMA模型本身中文语料就不充分，Chinese-LLaMA-Alpaca的训练数据量少，且训练时长不够，存在中文训练不充分的问题
- 提到并使用一个[中文语料库](https://github.com/brightmart/nlp_chinese_corpus)

## LMFlow

[代码仓库](https://github.com/OptimalScale/LMFlow)

| 是否开源代码 |     是否开源训练数据     | 是否开源模型参数 | 训练数据大小 |            模型参数大小             | 训练设备 | 训练时长 |
|:------:|:----------------:|:--------:|:------:|:-----------------------------:|:----:|:----:|
|  是  | 是 |    是     |   未知   | 作者训练的都是LLaMA羊驼：7B、13B、33B、65B |  未知  |  未知  |

- 该项目是一个高效、便利的微调框架，支持所有HuggingFace中的decoder models（比如LLaMA、T5、Glactica、GPT-2、ChatGLM），同样使用LoRA技术
- 提供了作者训练并部署到线上的LLaMA羊驼模型给人免费试用
- 目前支持三种微调方式Task Tuning（加强模型在专业领域，比如医疗，上的表现）、Instruction Tuning（就是指令精调，让模型学会遵循命令行事，我们常说的利用提示语prompt调教模型就是用指令精调得到的功能），Parameter-Efficient Tuning（就是HuggingFace的PEFT）
- 作者运用Task Tuning（enhance a language model’s proficiency in a particular field）训练出来的LLaMA羊驼模型和ChatGPT、InstructGPT-175B等模型在医疗领域比了比，效果确实不错，并在MMLU（Massive Multitask Language Understanding）上测试了一下，发现在非领域知识回答上性能也没有太多下降
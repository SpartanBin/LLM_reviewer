# 先写结论

- 2023-04-18: 其实众多微调方法或框架使用技术都类似，且有一半小羊驼模型都微调自[LLaMA](#llama)，决定模型质量的因素主要是数据量、数据质量、算力成本，如果要制作自己专业领域的羊驼模型，个人认为应以LLaMA作为预训练模型，收集尽可能多的中英文语料（假设你的模型要部署到中文生产环境），对LLaMA做再训练，这一步如果效果不好，可以考虑用[ChatYuan](#chatyuan)替代，然后用[Task Tuning](#lmflow)和专业领域数据进行微调，最后收集指令微调数据，并进行指令微调，[LMFlow](#lmflow)和[DeepSpeed-Chat](#deepspeed-chat)都可以成为很好的微调框架

# 重要羊驼大模型发布时间表

- 2023-02-24 Meta AI发布[LLaMA](#llama)
- 2023-03-10 TOGETHER发布[OpenChatKit](#openchatkit)
- 2023-03-13 清华数据挖掘研究团队（THUDM）发布[ChatGLM-6B](#chatglm-6b)
- 2023-03-13 Stanford发布[Alpaca](#alpaca)
- 2023-03-14 Eric J. Wang（人肉了下，发现人在Stanford，什么身份不清楚）发布[Alpaca-LoRA](#alpaca-lora)
- 2023-03-19 一个队伍（from UC Berkeley、CMU, Stanford, and UC San Diego）发布[Vicuna](#vicuna)
- 2023-03-23 元语智能发布[ChatYuan](#chatyuan)
- 2023-03-25 Databricks发布[Dolly](#dolly)
- 2023-03-27 哈工大科大讯飞联合实验室（HFL）发布 [Chinese-LLaMA-Alpaca](#chinese-llama-alpaca)
- 2023-03-27 香港科技大学统计机器学习实验室（HKUST）发布 [LMFlow](#lmflow)
- 2023-04-03 Berkeley Artificial Intelligence Research Lab (BAIR) of UC Berkeley发布[Koala](#koala)
- 2023-04-06 微软研究院（Microsoft Research）发布[GPT-4-LLM](#gpt-4-llm)
- 2023-04-12 微软发布[DeepSpeed-Chat](#deepspeed-chat)

# 其他的一些羊驼大模型发布时间表和信息

- 2023-03-17 链家科技（Ke Technologies）发布[BELLE](https://github.com/LianjiaTech/BELLE)，目前已发布了ChatBELLE App一个支持MacOS的APP，已发布7B模型
- 2023-03-23 Facico（Facico是他github的名字，可以确定的是他是个中国人，其他没人肉到任何信息）发布[Chinese-Vicuna](#chinese-vicuna)，使用[BELLE](https://github.com/LianjiaTech/BELLE)和[Guanaco](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)数据作为训练数据，开源7B、13B模型，提到之后会增加多轮对话数据
- 2023-03-25 李煜东（人肉了下，发现是深圳大学博士生）发布[Chinese-ChatLLaMA](https://github.com/ydli-ai/Chinese-ChatLLaMA)，作者找来中英平行语料、中文维基、社区互动、新闻数据、科学文献等语料再进行预训练，作者还在持续收集更多中文语料数据，语料数据都开源，且还开源了33B和65B的大模型

# 重要羊驼大模型简介

## LLaMA

[代码仓库](https://github.com/facebookresearch/llama)

|   是否开源代码    | 是否开源训练数据 | 是否开源模型参数 |     训练数据大小      |     模型参数大小     |     训练设备      |   训练时长   |
|:-----------:|:--------:|:--------:|:---------------:|:--------------:|:-------------:|:--------:|
| 是（训练代码没有开源） |    是     |    是     | 1T、1T、1.4T、1.4T | 7B、13B、33B、65B | 65B用2048张A100 | 65B用时21天 |

- 13B版在很多测试上优于GPT3（175B）
- 65B版与许多当时的最优模型相比都具有竞争力，比如Chinchilla-70B和PaLM-540B
- 核心思想是说***同样预算***的情况下，用小模型+大量数据，可以得到比大模型+少量数据更好的效果，虽然训练时间更长

## OpenChatKit

[代码仓库](https://github.com/togethercomputer/OpenChatKit)

|   是否开源代码    | 是否开源训练数据 | 是否开源模型参数 |     训练数据大小      | 模型参数大小 |   训练设备    | 训练时长 |
|:-----------:|:--------:|:--------:|:---------------:|:------:|:---------:|:----:|
| 是 |    是     |    是     | 43M | 7B、20B | 16 * A100 |  未知  |

- 有两个版本，微调自[GPT-NeoX-20B](https://github.com/EleutherAI/gpt-neox)的GPT-NeoXT-Chat-Base-20B和微调自[Pythia-6.9B-deduped](https://github.com/EleutherAI/pythia)的Pythia-Chat-Base-7B
- 微调训练数据是[OIG](https://huggingface.co/datasets/laion/OIG)，由[Together](https://www.together.xyz/), [LAION](https://laion.ai/), 和[Ontocord.ai](https://www.ontocord.ai/)共同制作，作者还在持续收集数据，该数据还在持续更新中

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

| 是否开源代码 | 是否开源训练数据 | 是否开源模型参数 | 训练数据大小 | 模型参数大小 |   训练设备   | 训练时长 |
|:------:|:--------:|:--------:|:------:|:------:|:--------:|:----:|
|   是    |    是     |    是     |  70k   | 7B、13B | 8 * A100 |  1天  |

- Vicuna也是一个在LLaMA的基础上微调过来的模型，微调方式改进自Alpaca，主要有以下不同：1.Alpaca输入的上下文长度是512，改为2048；2.调整训练损失为多轮对话，并单独计算每个聊天机器人输出的微调损失；3.有使用压缩显存的计算和减少训练成本的技术，见[博客](https://vicuna.lmsys.org/)
- 对话数据来源于[ShareGPT.com](https://sharegpt.com/)，这是一个gpt用户分享自己聊天对话的网站，又一个不错的数据源
- 作者提到他们的7B模型大概花费140美元的训练费，13B是300美元
- 作者提到自己的13B模型已经达到了chatgpt 90%的能力水平，且和Alpaca对比生成的结果内容更丰富，结构更准确，但是和其他‘小’大模型一样，推理能力和数学能力不太行
- 提到一个OpenAI的API [moderation](https://platform.openai.com/docs/guides/moderation/overview)可以用来过滤到用户的不恰当输入
- 提供了一个在线与大语言模型对话的[demo](https://chat.lmsys.org/)，里面有Vicuna、Koala、Dolly、ChatGLM、Alpaca、LLaMA这几个模型

## ChatYuan

[代码仓库](https://github.com/clue-ai/ChatYuan)

| 是否开源代码 | 是否开源训练数据 | 是否开源模型参数 | 训练数据大小 | 模型参数大小 | 训练设备 | 训练时长 |
|:------:|:--------:|:--------:|:------:|:------:|:----:|:----:|
|   是    |    否     |    是     |   未知   |   未知   |  未知  |  未知  |

- ChatYuan-large-v2应该是微调自此团队的预训练模型[PromptCLUE](https://github.com/clue-ai/PromptCLUE)，该预训练模型开源了代码、数据和参数（base版），数据使用1.5万亿中文token，亿级中文任务数据上完成训练，训练任务超过150+
- 该项目开源了他的微调代码，和微调输入的数据格式，使用起来比较方便，说的技术路线改进自ChatYuan-large-v1，没有找到ChatYuan-large-v1的相关信息
- 看了一下demo效果，感觉很不错

## Dolly

[代码仓库](https://github.com/databrickslabs/dolly)

| 是否开源代码 | 是否开源训练数据 | 是否开源模型参数 | 训练数据大小 | 模型参数大小 |   训练设备   | 训练时长 |
|:------:|:--------:|:--------:|:------:|:------:|:--------:|:----:|
|   是    |    是     |    是     |  15k   |  12B   | 8 * A100 |  未知  |

- Dolly是[Pythia-12B](https://github.com/EleutherAI/pythia)经过指令微调得到的
- 数据都是指令数据，数据源主要是wiki和Databricks的数千名员工生成的（可能数据质量高）
- 在句法复杂的提示、编程问题、数学运算、事实错误、日期和时间、开放式问题回答、幻觉、列举特定长度的列表、文体模仿、幽默感等方面表现不佳

## Chinese-LLaMA-Alpaca

[代码仓库](https://github.com/ymcui/Chinese-LLaMA-Alpaca)

| 是否开源代码 | 是否开源训练数据 | 是否开源模型参数 |      训练数据大小      | 模型参数大小 |  训练设备   | 训练时长 |
|:-------:|:--------:|:--------:|:----------------:|:--------------:|:-------:|:----:|
|    是（训练代码没有开源）     |    是     |    是     | 预训练20G，指令精调2M、3M |  7B、13B  | 16张A100 |  未知  |

- 在LLaMA的基础上再做预训练（Pre-training）并使用Alpaca指令精调（Instruction Fine-tuning）得来，特点是***扩充了中文词表***，同样使用LoRA技术
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

## Koala

[代码仓库](https://github.com/young-geng/EasyLM)

[数据处理代码仓库](https://github.com/young-geng/koala_data_pipeline)

| 是否开源代码 |  是否开源训练数据   | 是否开源模型参数 | 训练数据大小 |            模型参数大小             |   训练设备   | 训练时长 |
|:------:|:-----------:|:--------:|:------:|:-----------------------------:|:--------:|:----:|
|  是  | 是（用的都是开源数据） |    是     |   未知   | 7B、13B | 8 * A100 | 6小时  |

- 首先Koala是基于EasyLM框架实现的（一个集成大语言模型的预训练、微调、验证功能的框架，且支持数百个显卡加速），Koala相当于是利用该框架和一些数据微调LLaMA得出来的模型
- 数据来自多种渠道，详见[博客的Datasets and Training章节](https://bair.berkeley.edu/blog/2023/04/03/koala/)

## GPT-4-LLM

[代码仓库](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)

| 是否开源代码 | 是否开源训练数据 | 是否开源模型参数 | 训练数据大小 | 模型参数大小 |   训练设备   | 训练时长 |
|:------:|:--------:|:--------:|:------:|:------:|:--------:|:----:|
|   否    |    是     |    否     |   未知   | 未知 | 未知 |  未知  |

- 微软研究院发布的小羊驼模型，目前只开源了数据，数据是中英双语的，持续关注中
- 除了使用self-instruct tuning训练了一个小羊驼，还生成了一些比较数据（来自GPT3.5、4、OPT1.3B）来训练了一个打分模型（reward models），用这个打分模型去量化GPT4和小羊驼的差距

## DeepSpeed-Chat

[代码仓库](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)

| 是否开源代码 | 是否开源训练数据 | 是否开源模型参数 | 训练数据大小 | 模型参数大小 |   训练设备   | 训练时长 |
|:------:|:--------:|:--------:|:------:|:------:|:--------:|:----:|
|   是    |    否     |    否     |   未知   | 未知 | 未知 |  未知  |

- 微软发布的微调框架，用[DeepSpeed](https://github.com/microsoft/DeepSpeed)实现了[论文InstructGPT](https://arxiv.org/abs/2203.02155)中的监督微调、奖励函数模型微调、强化学习人类反馈（Reinforcement Learning Human Feedback, RLHF），作者说微软研发的DeepSpeed可以显著加速训练（可达15倍速度），并且还可以显著加速推理，做到高吞吐量和低延迟
- 目前支持以下预训练模型：

| model family                                                   | size range  |
|----------------------------------------------------------------|-------------|
| [opt](https://huggingface.co/models?other=opt)                 | 0.1B - 66B  |
| [bloom](https://huggingface.co/models?other=bloom)             | 0.3B - 176B |
| [gpt\_neox](https://huggingface.co/models?other=gpt_neox)      | 1.3B - 20B  |
| [gptj](https://huggingface.co/models?other=gptj)               | 1.4B - 6B   |
| [gpt\_neo](https://huggingface.co/models?other=gpt_neo)        | 0.1B - 2.7B |
| [gpt2](https://huggingface.co/models?other=gpt2)               | 0.3B - 1.5B |
| [codegen](https://huggingface.co/Salesforce/codegen-16B-multi) | 0.35b - 16B |

正在开发对LLaMA的支持

- 以下是他加速RLHF训练的例子：

| GPU SKUs      | OPT-1.3B      | OPT-6.7B       | OPT-13.2B       | OPT-30B       | OPT-66B           | Bloom-175B      |
|---------------|---------------|----------------|-----------------|---------------|-------------------|-----------------|
| 1x V100 32G   | 1.8 days      |                |                 |               |                   |                 |
| 1x A6000 48G  | 1.1 days      | 5.6 days       |                 |               |                   |                 |
| 1x A100 40G   | 15.4 hrs      | 3.4 days       |                 |               |                   |                 |
| 1x A100 80G   | 11.7 hrs      | 1.7 days       | 4.9 days        |               |                   |                 |
| 8x A100 40G   | 2 hrs         | 5.7 hrs        | 10.8 hrs        | 1.85 days     |                   |                 |
| 8x A100 80G   | 1.4 hrs($45)  | 4.1 hrs ($132) | 9 hrs ($290)    | 18 hrs ($580) | 2.1 days ($1620)  |                 |
| 64x A100 80G  | 31 minutes    | 51 minutes     | 1.25 hrs ($320) | 4 hrs ($1024) | 7.5 hrs ($1920)    | 20 hrs ($5120) |

以上RLHF训练使用了135M tokens的数据，由6个开源数据集组成[rm-static](https://huggingface.co/datasets/Dahoas/rm-static)、[full-hh-rlhf](https://huggingface.co/datasets/Dahoas/full-hh-rlhf)、[synthetic-instruct-gptj-pairwise](https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise)、[rlhf-reward-datasets](https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets)、[webgpt_comparisons](https://huggingface.co/datasets/openai/webgpt_comparisons)、[SHP](https://huggingface.co/datasets/stanfordnlp/SHP)

# 数据集整理

xxx

# TODO

- [ ] 整理收集归纳中英文语料数据集，明确每个数据集的最佳用途（适合作预训练或任务微调或指令微调或RLHF等）
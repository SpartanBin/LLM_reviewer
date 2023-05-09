# 目的

- 2023-04-04: 为了建立自己领域的大语言模型（或者说为了加强大语言模型在自己领域的表现），而在做技术调研

# 先写结论

- 2023-04-18: 其实众多微调方法或框架使用技术都类似，且有一半小羊驼模型都微调自[LLaMA](#llama)，决定模型质量的因素主要是数据量、数据质量、算力成本，如果要制作自己专业领域的羊驼模型，个人认为应以LLaMA作为预训练模型，收集尽可能多的中英文语料（假设你的模型要部署到中文生产环境），对LLaMA做再训练，这一步如果效果不好，可以考虑用[ChatYuan](#chatyuan)替代，然后用[Task Tuning](#lmflow)和专业领域数据进行微调，最后收集指令微调数据，并进行指令微调，[LMFlow](#lmflow)和[DeepSpeed-Chat](#deepspeed-chat)都可以成为很好的微调框架
- 2023-04-26: 建立自己领域的大语言模型，我们要做'Task Tuning'或许是因为我们的知识库太大，类型太庞杂，不太容易全部放进'一个prompt'中，因此要让LLM利用我们的知识，简单暴力的方法或许是微调，现在还有一种方案，是在prompt engineering上做功夫，如使用LlamaIndex或是LangChains，使得LLM与自己领域的知识库相关联，获得知识增强型的回答

# 一些关于大语言模型的名词和实验现象或是其他重要技术

- Scaling (Training compute, FLOPs): 代表模型的规模，等于 α * model_size * training_tokens，α为系数，model_size为模型参数量，training_tokens为数据量（1000个tokens差不多为750个词语）
- [Scaling laws](https://arxiv.org/abs/2001.08361): 如果你希望提升模型的表现，就必须增大Scaling (模型大小，数据大小，计算资源大小)，[open ai 的gpt-4技术报告](https://arxiv.org/abs/2303.08774)显示，模型的表现与训练时间的关系是可预测的，也就是说，在模型刚开始训练时，就可以知道模型最终训练结束的表现
- 有137种能力是大Scaling才有的，而小Scaling没有，见[博客](https://bounded-regret.ghost.io/emergent-deception-optimization/)，这种能力称为Emergent abilities，这种能力打破了Scaling laws
- few-shot prompting（few-shot prompting是指给模型的输入中加入一些与当前模型要做的任务相似的例子）在小Scaling上没有什么用，但是在大Scaling上有用
- RLHF对小Scaling是有害的，但对大Scaling是有益的，见[论文](https://arxiv.org/abs/2204.05862)
- [The Inverse Scaling Prize](https://github.com/inverse-scaling/prize)提出了11种任务，这11种任务对于一般的大语言模型来说，都是随着Scaling增大，效果反而会变差的。[Inverse scaling can become U-shaped](https://arxiv.org/abs/2211.02011)发现如果使用[chain-of-thought prompting](https://arxiv.org/abs/2201.11903)（简单来说就是在给模型的提示中加入推理，让它‘分步思考’）可以使部分任务从随着Scaling变大一直变差，变为表现成U型，意思是先变差后变好
- 对于哈尔滨工业大学的[ChatGPT 调研报告](https://github.com/DeepTecher/awesome-ChatGPT-resource-zh/blob/main/pdfs/230311-%E5%93%88%E5%B0%94%E6%BB%A8%E5%B7%A5%E4%B8%9A%E5%A4%A7%E5%AD%A6-ChatGPT%E8%B0%83%E7%A0%94%E6%8A%A5%E5%91%8A.pdf)内容小总结: 1. 报告里提到ChatGPT在Scaling达到一定尺度的情况下，效果出现激增，打破了Scaling laws的规律(线性增强)，认为是因为加入了很多代码作为训练语料，因代码语言的特性可以强化模型的推理(reason)能力，[论文](https://arxiv.org/abs/2211.09110)发现训练数据中含有代码的模型具有很强的语言推理能力，代码预训练与思维链Chain-of-Thought（COT）表现息息相关，在预训练时使用代码数据成为越来越多研究者的共识；2. GPT式的decoder模型的预训练方法相当于是对一句话中下个词语的预测(分类)，这种方式可以方便生成任意长度文本，这种方式类似Word2Vec在做词向量编码，可能其能够把不同语言的相同语义投影到高维空间中的相同位置；3. 指令精调是提示学习(Prompt Learning)的加强版，其作用在于让模型学习人类对话交互的模式（与人类行为‘对齐’），且能让他有能力泛化到没见过的分布中；4. RLHF是让模型了解人类的‘意图’，让模型的回答是人类希望看到的；5. 部署大模型的方式是模型并行、数据并行的混用；6. 提到Nvidia的[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)和Microsoft的[DeepSpeed](https://github.com/microsoft/DeepSpeed)，应该都可以作为很好的部署框架；7. 在第四章提到巨量预训练数据集！！！；8. 提到构建指令精调数据集需要设计指令模板（针对不同任务），且覆盖不同类型的数据，逻辑推理任务标注时可以用COT，能够提高表现；9. 第五章提到评价方式，在需要评估自己的大模型时可以回来仔细参考
- [LlamaIndex](https://github.com/jerryjliu/llama_index): LlamaIndex提供了一个数据连接给LLM，包含多种数据源格式（API, PDF, docs, SQL等），为非结构化的数据源提供索引，方便用户请求并结合LLM获得数据增强的输出结果，将数据源结构化，解决LLM token输入长度限制和文本分割的问题，详见[文档](https://gpt-index.readthedocs.io/en/latest/)
- [LangChains](https://github.com/hwchase17/langchain): LangChains有多种功能，1.特定领域的检索增强（特定领域问答增强），也是为数据建立索引，2.优化对话记忆，建立聊天机器人，3.建立一个代理，让LLM去使用工具（包括arxiv、google search、wiki等），除此之外一些其他的应用案例，详见[文档](https://python.langchain.com/en/latest/)
- 对于Facebook的[LLMsPracticalGuide](https://arxiv.org/abs/2304.13712v2)及其[相关资源列表](https://github.com/Mooler0410/LLMsPracticalGuide)小结: 该综述主要是在比较LLM和‘传统微调’小模型在大部分自然语言任务上的优劣势，结果表明‘传统微调’小模型在某些任务上依然是合适且具有优势的，不过在模仿人类、聊天机器人、生成、知识密集型应用等任务上，大语言模型确实具有显著优势，个人认为，从对比结果的角度，或许可以把综述里的LLM视为GPT4等千亿万亿级参数模型，而把‘传统微调’小模型视为7B、13B等小羊驼模型。一些小细节（以下用小模型代指‘传统微调’小模型）: 1. LLM在只需要使用上下文知识的任务（比如阅读理解）中表现不佳；2. 在能够使用知识库检索的情况下，小模型是更好的选择；3. 实际应用中，out-of-distribution (OOD)情况较少，推荐小模型；4. 除了LoRA外，还提到了两种Parameter-Efficient Tuning技术[Prefix Tuning](https://arxiv.org/abs/2101.00190)和[P-Tuning](https://arxiv.org/abs/2110.07602)；5. 资源列表里给了很多预训练和fine tune用到的数据，给力!!!还列出了众多自然语言测试任务，可以做了解和参考

# 重要羊驼大模型发布时间表

- 2023-02-24 Meta AI发布[LLaMA](#llama)
- 2023-03-10 TOGETHER发布[OpenChatKit](#openchatkit)
- 2023-03-13 清华数据挖掘研究团队（THUDM）发布[ChatGLM-6B](#chatglm-6b)
- 2023-03-13 Stanford发布[Alpaca](#alpaca)
- 2023-03-14 Eric J. Wang（人肉了下，发现人在Stanford，什么身份不清楚）发布[Alpaca-LoRA](#alpaca-lora)
- 2023-03-19 LM-SYS（from UC Berkeley、CMU, Stanford, and UC San Diego）发布[FastChat](#fastchat)
- 2023-03-20 Nomic AI发布[GPT4All](#gpt4all)
- 2023-03-23 元语智能发布[ChatYuan](#chatyuan)
- 2023-03-25 Databricks发布[Dolly](#dolly)
- 2023-03-27 哈工大科大讯飞联合实验室（HFL）发布 [Chinese-LLaMA-Alpaca](#chinese-llama-alpaca)
- 2023-03-27 香港科技大学统计机器学习实验室（HKUST）发布 [LMFlow](#lmflow)
- 2023-04-03 Berkeley Artificial Intelligence Research Lab (BAIR) of UC Berkeley发布[Koala](#koala)
- 2023-04-06 微软研究院（Microsoft Research）发布[GPT-4-LLM](#gpt-4-llm)
- 2023-04-08 港中文发布[LLMZoo](#llmzoo)
- 2023-04-12 微软发布[DeepSpeed-Chat](#deepspeed-chat)
- 2023-04-19 Stability AI发布[StableLM](#stablelm)
- 2023-04-21 复旦发布[MOSS](#moss)

# 其他的一些羊驼大模型发布时间表和信息

- 2023-03-17 链家科技（Ke Technologies）发布[BELLE](https://github.com/LianjiaTech/BELLE)，目前已发布了ChatBELLE App一个支持MacOS的APP，已发布7B模型
- 2023-03-23 Facico（Facico是他github的名字，可以确定的是他是个中国人，其他没人肉到任何信息）发布[Chinese-Vicuna](#chinese-vicuna)，使用[BELLE](https://github.com/LianjiaTech/BELLE)和[Guanaco](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)数据作为训练数据，开源7B、13B模型，提到之后会增加多轮对话数据
- 2023-03-25 李煜东（人肉了下，发现是深圳大学博士生）发布[Chinese-ChatLLaMA](https://github.com/ydli-ai/Chinese-ChatLLaMA)，作者找来中英平行语料、中文维基、社区互动、新闻数据、科学文献等语料再进行预训练，作者还在持续收集更多中文语料数据，语料数据都开源，且还开源了33B和65B的大模型
- 2023-04-20 [Lamini](https://lamini.ai/)(一个公司，CEO是斯坦福的)发布[lamini](https://github.com/lamini-ai/lamini)，购买了他们的许可后，可以调用他们的api训练模型，不用使用自己的设备，且他们提供了一个数据生成器，可以生成用来进行指令精调的数据

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

## FastChat

[代码仓库](https://github.com/lm-sys/FastChat)

| 是否开源代码 | 是否开源训练数据 | 是否开源模型参数 | 训练数据大小 |             模型参数大小              |   训练设备   | 训练时长 |
|:------:|:--------:|:--------:|:------:|:-------------------------------:|:--------:|:----:|
|   是    |    否     |    是     |  70k   | Vicuna: 7B、13B, FastChat-T5: 3B | 8 * A100 |  1天  |

- 目前开源了2种模型，Vicuna先开源，随后开源FastChat-T5
- Vicuna也是一个在LLaMA的基础上微调过来的模型，微调方式改进自Alpaca，主要有以下不同：1.Alpaca输入的上下文长度是512，改为2048；2.调整训练损失为多轮对话，并单独计算每个聊天机器人输出的微调损失；3.有使用压缩显存的计算和减少训练成本的技术，见[博客](https://vicuna.lmsys.org/)
- 对话数据处理自[ShareGPT.com](https://sharegpt.com/)（这是一个gpt用户分享自己聊天对话的网站，又一个不错的数据源），不过并未分享自己处理后的数据，但是给了如何处理的代码
- 作者提到他们的7B模型大概花费140美元的训练费，13B是300美元
- 作者提到自己的13B模型已经达到了chatgpt 90%的能力水平，且和Alpaca对比生成的结果内容更丰富，结构更准确，但是和其他‘小’大模型一样，推理能力和数学能力不太行
- 提到一个OpenAI的API [moderation](https://platform.openai.com/docs/guides/moderation/overview)可以用来过滤到用户的不恰当输入
- 提供了一个在线与大语言模型对话的[demo](https://chat.lmsys.org/)，里面有Vicuna、Koala、Dolly、ChatGLM、Alpaca、LLaMA这几个模型
- 给出了一个对话平台，可以往其中添加对话机器人，目前支持多个聊天机器人，详见仓库
- 提供了一个验证的功能，可以用gpt4来评估模型的能力

## GPT4All

[代码仓库](https://github.com/nomic-ai/gpt4all)

| 是否开源代码 | 是否开源训练数据 | 是否开源模型参数 | 训练数据大小 | 模型参数大小 |   训练设备   | 训练时长 |
|:------:|:--------:|:--------:|:------:|:------:|:--------:|:----:|
|   是    |    是     |    是     |  437k  | 多个版本，请看[报告](https://static.nomic.ai/gpt4all/2023_GPT4All-J_Technical_Report_2.pdf) | 8 * A100 |  未知  |

- 微调了多个模型，最新的是微调自GPT-J的，所有都有完整的实验记录
- 称自己的训练数据都是干净的高质量，包括代码、故事和对话
- 提供了多个编译好的可执行程序，可直接在个人电脑上安装并使用
- 在框架中融入了langchain

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

## LLMZoo

[代码仓库](https://github.com/FreedomIntelligence/LLMZoo)

| 是否开源代码 | 是否开源训练数据 | 是否开源模型参数 | 训练数据大小 |            模型参数大小            |   训练设备   | 训练时长 |
|:------:|:--------:|:--------:|:------:|:----------------------------:|:--------:|:----:|
|   是    |    是     |    是     |   未知   | Phoenix: 7B, Chimera: 7B、13B | 未知 |  未知  |

- Phoenix微调自[BLOOMZ](https://huggingface.co/bigscience/bloom)，[论文](https://arxiv.org/abs/2211.05100)，是一个包含46种语言和13种程序语言的开源预训练模型
- Chimera微调自LLaMA
- 微调数据主要是指令数据和对话，共1.58GB
- 提供了Phoenix、Chimera与其他LLM的对比结果: 

| Evaluation by GPT4, Chinese                     | Ratio   |
|-------------------------------------------------|---------|
| Phoenix-inst-chat-7b vs. **ChatGPT**            | 85.2\%  |
| Phoenix-inst-chat-7b vs. **ChatGLM-6b**         | 94.6\%  |
| Phoenix-inst-chat-7b vs. **Baidu-Wenxin**       | 96.8\%  |
| **Phoenix-inst-chat-7b** vs. MOSS-moon-003-sft  | 109.7\% |
| **Phoenix-inst-chat-7b** vs. BELLE-7b-2m        | 122.7\% |
| **Phoenix-inst-chat-7b** vs. Chinese-Alpaca-7b  | 135.3\% |
| **Phoenix-inst-chat-7b** vs. Chinese-Alpaca-13b | 125.2\% |

85.2\%代表Phoenix-inst-chat-7b达到了ChatGPT百分之85.2的表现

| Evaluation by human, Chinese       | win | tie | lose  |
|------------------------------------|:---:|:---:|:-----:|
| Phoenix vs. **ChatGPT**            | 12  |  35 |  53   |
| Phoenix vs. **ChatGLM-6b**         | 36  |  11 |  53   |
| Phoenix vs. **Baidu-Wenxin**       | 29  |  25 |  46   |
| **Phoenix** vs. BELLE-7b-2m        | 55  |  31 |  14   |
| **Phoenix** vs. Chinese-Alpaca-13b | 56  |  31 |  13   |

| Evaluation by GPT4, Chinese            | Ratio      |
|----------------------------------------|------------|
| Chimera-chat-7b vs.  **ChatGPT**       | 85.2\%     |
| Chimera-chat-13b vs.  **ChatGPT**      | 92.6\%     |
| Chimera-inst-chat-13b vs.  **ChatGPT** | **96.6\%** |

效果很好，感觉这个模型和ChatGLM-6b都可以作为backbone

## DeepSpeed-Chat

[代码仓库](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)

| 是否开源代码 | 是否开源训练数据 | 是否开源模型参数 | 训练数据大小 | 模型参数大小 |   训练设备   | 训练时长 |
|:------:|:--------:|:--------:|:------:|:------:|:--------:|:----:|
|   是    |    否     |    否     |   未知   | 未知 | 未知 |  未知  |

- 微软发布的微调框架，用[DeepSpeed](https://github.com/microsoft/DeepSpeed)实现了[论文InstructGPT](https://arxiv.org/abs/2203.02155)中的监督微调、奖励函数模型微调、强化学习人类反馈（Reinforcement Learning Human Feedback, RLHF），作者说微软研发的DeepSpeed可以显著加速训练（可达15倍速度），并且还可以显著加速推理，做到高吞吐量和低延迟
- 目前支持以下预训练模型: 

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

## StableLM

[代码仓库](https://github.com/Stability-AI/StableLM)

| 是否开源代码 | 是否开源训练数据 | 是否开源模型参数  | 训练数据大小 |  模型参数大小   |   训练设备   | 训练时长 |
|:------:|:--------:|:---------:|:------:|:---------:|:--------:|:----:|
|   否    |    否     | 是 |  800B  | 3B、7B、13B | 未知 |  未知  |

- 13B的模型是微调自Vicuna-13B的，这个好像没有算在StableLM系列模型里
- StableLM暂时发布了3B、7B版本，还剩15B、30B、65B、175B没发布
- 训练数据构建自[The Pile](https://pile.eleuther.ai/)，数据量是The Pile的3倍，输入长度是4096 tokens
- 除此之外还使用了5个微调数据集: Alpaca的，gpt4all的，[RyokoAI的ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K)，Dolly的，[Anthropic的HH](https://github.com/anthropics/hh-rlhf)

## MOSS

[代码仓库](https://github.com/OpenLMLab/MOSS)

|   是否开源代码    | 是否开源训练数据  | 是否开源模型参数  | 训练数据大小 | 模型参数大小 |   训练设备   | 训练时长 |
|:-----------:|:---------:|:---------:|:------:|:------:|:--------:|:----:|
| 是（未开源预训练代码） | 是（还未全部开源） | 是（还未全部开源） |   未知   |  16B   | 未知 |  未知  |

- MOSS目前开源了3个版本，moss-moon-003-base、moss-moon-003-sft和moss-moon-003-sft-plugin，另还有3个版本将在之后发布，持续关注
- moss-moon-003-base是预训练模型，预训练自[CodeGen](https://github.com/salesforce/CodeGen)，[论文](https://arxiv.org/abs/2203.13474)，是在将近70B的中英文语料上预训练得到的，另外CodeGen是一个专门用来通过语言生成代码的模型，可能其代码语料充足，可能其推理能力很强
- moss-moon-003-sft在moss-moon-003-base基础上经过110万多轮对话数据微调得到
- moss-moon-003-sft-plugin这个比较有特色，是在moss-moon-003-base基础上经过110万多轮对话数据和30万插件增强的多轮对话数据上微调得到，在对话能力基础上还有使用搜索引擎、文生图、计算器、解方程等四种插件的能力
- 微调数据都还未完全开源，持续关注，数据包含：moss-002-sft-data，说的是moss-002用的，moss-002并未开源，moss-003-sft-data，刚刚提到的110万对话数据，moss-003-sft-plugin-data刚刚提到的插件增强对话数据

# 数据集整理

xxx

# TODO

- [ ] 整理收集归纳中英文语料数据集，明确每个数据集的最佳用途（适合作预训练或任务微调或指令微调或RLHF等）
# llm-learning
个人的大模型学习仓库。

### 一、大模型基础<br>
>[DataWhale-面向开发者的大模型手册(吴恩达课程)](https://github.com/datawhalechina/llm-cookbook)<br>

### 二、大模型应用开发<br>
#### RAG
[DataWhale官网](https://linklearner.com/)
>⭐[DataWhale-动手学大模型应用开发](https://github.com/datawhalechina/llm-universe/tree/main)<br>
>⭐[DataWhale-知识库助⼿项⽬](https://github.com/logan-zou/Chat_with_Datawhale_langchain)<br>
> [ChatWithDatawhale](https://github.com/sanbuphy/ChatWithDatawhale)
> [大模型白盒子构建指南-从零手搓](https://github.com/datawhalechina/tiny-universe)
>[A-Guide-to-Retrieval-Augmented-LLM](https://github.com/Wang-Shuo/A-Guide-to-Retrieval-Augmented-LLM)<br>
>[Awesome-LLM-RAG](https://github.com/jxzhangjhu/Awesome-LLM-RAG)

这一部分主要结合实际项目进行学习，技术点：
+ 检索增强生成 ( Retrieval Augmented Generation, RAG)
+ 提示工程（Prompt Enginnering）
+ 开发框架：LangChain

#### Agent
[吴恩达Translation-Agent](https://github.com/andrewyng/translation-agent/tree/main)

### 三、大模型部署和微调<br>
[DataWhale-开源大模型食用指南](https://github.com/datawhalechina/self-llmn)<br>
[DataWhale-大模型基础](https://github.com/datawhalechina/so-large-lm/tree/main)<br>

### 四、 工具和框架
[Kimi：一款面向普通用户（C端）的智能助手，旨在提供高效的信息查询和对话服务。](https://kimi.moonshot.cn/)
[ChatGPT(科学上网)](https://chat.openai.com/)
[天工API](https://www.tiangong.cn/)
[Prompt Engineering Guide 提示工程指南](https://www.promptingguide.ai/zh)
[LlamaIndex 中文文档](https://llama-index.readthedocs.io/zh/latest/getting_started/installation.html)

### 五、 LLM+Rec
[LLM4Rec-Awesome-Papers](https://github.com/WLiK/LLM4Rec-Awesome-Papers)


# 体系化的教程
https://github.com/liguodongiot/llm-action


https://m.okjike.com/originalPosts/658d5f52c7c69d5a9fdb573d?s=eyJ1IjoiNWY2YjZjMjMxZmVhMjcwMDE3NGYxZmU5IiwiZCI6MX0%3D
👩‍💻回顾今年，AI大模型开发者关注的技术要点可以归纳为以下几个主题，汇总了每个主题涵盖的一些开发者关心的问题，以及相关的技术文章和论文，分享出来

▶️大型语言模型（LLM）微调
▶️垂直领域模型微调
▶️LLM+外挂知识库
▶️LLM+Agent
▶️长文本微调
▶️多模态大型语言模型
▶️国内外高性能开源基座
▶️OpenAI官方发布的一些技术文档

1️⃣大型语言模型（LLM）微调
✅大语言模型从入门到精通
🔗大模型技术基础教材：intro-llm.github.io
🔗大模型技术实战：wangwei1237.github.io
✅微调指令数据集构造
🔗通过self instruct的方式让GPT-4生成大量的指令和回复的数据对
🔗开源指令集汇总：github.com
✅低资源下，微调大模型选择的技术路线
🔗参数高效微调方法（PEFT，如lora、prefix tuning等）：www.zhihu.com
🔗Huggface开源的高效微调大模型的库：huggingface.co
🔗QLoRA和全量参数微调Llama/Baichuan等：github.com
✅微调、推理大模型所需的显存计算
🔗大模型显存估计开源工具：huggingface.co
🔗大语言模型LLM推理及训练显存计算方法：www.cnblogs.com
✅微调、推理、量化常见使用的开源框架
🔗常见微调框架：llama-factory、deepspeed、metronlm、unsloth
🔗常见推理加速框架：vllm、mlc-llm、Medusa
🔗常见量化框架：exllamav2、bitsandbytes
✅大语言模型幻觉相关的论文：
🔗幻觉定义、解决思路github.com
✅符尧老师关于数据工程、大模型评测文章：
🔗包含预训练阶段如何找到「最优的混合比例+数据格式+数据课程」来使学习速度最大化等
yaofu.notion.site
🔗关于大模型评测：yaofu.notion.site

2️⃣垂直领域模型微调
✅领域主流模型：
教育（如educat）、医疗（如ChatGLM-Med）、金融（如FinLLM）、心理（MindChat）、法律（ChatLaw）、科学（starwhisper）等
✅开源、高质量的预训练语料
🔗悟道data.baai.ac.cn
✅领域：专用数据集配比如何
🔗Chathome数据配比，介于1:5~1:10之间

3️⃣LLM+外挂知识库
✅知识库构建流程
🔗从 RAG 到 Self-RAG zhuanlan.zhihu.com
✅实现rag的开源项目，
🔗langchain、llamaindex baoyu.io
✅大模型外挂知识库（RAG）优化方案
🔗www.zhihu.com

4️⃣LLM+Agent
✅OpenAI应用研究主管的万字长文
🔗Agent = LLM+ 记忆 + 规划技能 + 工具使用：juejin.cn
✅Agent当前的研究重心
🔗如何选择基础模型、prompt设计上有哪些参考的示例：ReACT（react-lm.github.io）、ReWOO（arxiv.org）
✅Agent有哪些常见的主流开源框架
Autogen、AutoGPT、BabyAGI等

5⃣️长文本微调
✅长文外推能力的定义
🔗苏剑林老师：spaces.ac.cn
✅主流模型使用的外推技术
🔗旋转位置编码RoPE zhuanlan.zhihu.com
✅长文微调的流程和训练代码
🔗单卡高效扩展LLaMA2-13B上下文： github.com
✅长文本压测
🔗Kimi Chat 公布“大海捞针”长文本压测结果 mp.weixin.qq.com
✅100k上下文的工程与数据基础方案
🔗From 符尧 100k 可以通过暴力工程实现，不需要 fancy 的架构改变 yaofu.notion.site

6️⃣多模态大型语言模型
✅多模态和多模态大模型(LMM)
🔗全面介绍多模态系统，包括LMM baoyu.io
✅多模态有哪些主流的开源模型
fuyu-8b、llava、mPLUG-Owl2、Qwen-VL
✅多模态大型语言模型微调
🔗数据集构造、微调、评测 zhuanlan.zhihu.com

7⃣️国内外有哪些优质开源基座
✅llama1/2、phi-1/phi-1.5/phi-2、Mistral 7B、Orca2
✅qwen（7/14/72B）、baichuan1/2、yi（6/34B）
✅intenlm、tigerbot1/2、skywork

8️⃣OpenAI官方发布的一些技术文档
✅【中文版】OpenAI官方提示工程指南
baoyu.io
✅OpenAI 微调文档
platform.openai.com
✅OpenAI 安全对齐研究、超级对齐计划
openai.com
openai.com

作者：动物世界
链接：https://zhuanlan.zhihu.com/p/15754393607
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


1.NLP基础知识
1.1 文本处理基础
Tokenizer分词技术。
掌握常见的分词算法原理：例如，BPE、WordPiece、SentencePiece、Unigram。学完之后可以对比每个算法的优劣，改进之处等。
Embedding技术。
学习相关的Embedding模型算法，如One-hot、Word2Vec、FastText、Glove等，增加自己对词嵌入的理解。

1.2 核心架构组件
注意力机制。掌握Self-Attention、Multi-Head Attention、Cross-Attention、Mask-Attention的原理与实现。并能够计算Transformer中注意力计算的复杂度。
位置编码。掌握各种位置编码的原理，如绝对位置编码、可学习的位置编码、旋转位置编码RoPE，总结各种位置编码的优劣。RoPE可能要求手写。
归一化技术。掌握Layer Norm、Batch Norm、RMSNorm的原理与实现。总结对比它们的优劣，以及Pre Norm、Post Norm的差异。
残差连接。掌握残差连接的数学原理，为什么能让网络做的更深等。

1.3 基础网络组件
MLP（多层感知机）。掌握Transformer中前馈神经网络的结构设计，不同维度投影的作用。（为什么大模型的世界知识存储在这一部分？面字节被问到了）
激活函数。掌握ReLU及其变体、GELU的优势、SwiGLU在大模型中的应用、激活函数选择的考虑因素等。

1.4 损失函数
交叉熵损失。需要掌握原理与代码实现、在大语言模型训练中的应用。

预训练技术
2.1 数据处理
数据获取方法。掌握公开数据集的使用、数据质量评估指标、了解常见网络爬虫技术。
数据清洗技术。熟悉常见的文本去重算法原理、如MinHash，了解常见训练数据配比策略，如代码、数学、通用知识问答等各种占比多少比较合理？（面百度被问到了）
2.2 预训练流程
训练策略。这一部分需要了解很多大模型预训练中超参数的设置，以及每个超参数的用途。网上可以直接搜素到，用到了去看一下背后的原理就行。
预训练优化。掌握常见的优化技术，如梯度累积、混合精度训练、模型并行与数据并行、如何保证训练稳定性等。

2.3 结果评估
评估指标。掌握常见的评测指标，如困惑度（Perplexity）等。了解常见大模型评测数据集，如MMLU、IF-EVAL、MATH等。
增量预训练。了解继续预训练相关技术、以及如何解决灾难性遗忘等。

3.后训练技术
3.1 监督微调（SFT）
基础微调技术。掌握全参数微调原理，了解学习率设置策略，早停策略等。（SFT与预训练过程中loss计算有什么不同？面腾讯被问到了）
高效参数微调。熟悉掌握常见的高效参数微调算法原理。如LoRA、QLoRA、Prefix Tuning、P-Tuning、P-TuningV2、Adapter Tuning。以及不同方法的性能对比等。

3.2 人类偏好对齐
RLHF技术。了解RLHF数据构建过程、熟悉奖励模型训练过程、掌握PPO、DPO算法原理，了解更前沿的对齐算法。

4.推理优化
4.1 框架应用
DeepSpeed。掌握ZeRO1、ZeRO2、ZeRO3优化策略。了解offload、infinity策略。
Megatron-LM。掌握Megatron-LM模型并行策略、如张量并行、流水线并行等。

4.2 性能优化算法
注意力优化。掌握常见优化算法的原理，如FlashAttention、FlashAttentionV2，vLLM中pageAttention的原理等。
KV Cache技术。了解什么是KV Cache，为什么需要它，以及比较前沿的KV Cache算法。

5.常见大模型架构
5.1 经典架构
大模型架构。了解常见的大模型架构如GPT系列、LLaMA系列、GLM系列、Qwen系列、DeepSeek系列等。对比他们之间的差异，以及每个系列模型演变过程。

5.2 创新架构
Mixture of Experts。了解混合专家模型架构，与Dense架构有啥优劣。
Mamba、RWKV。了解Mamba、RWKV等前沿架构，它们的创新之处。与Transformer架构的优劣对比。

6.大模型应用
6.1 检索增强生成（RAG）
检索技术。检索算法（如HNSW等）、向量数据库选择、Embedding模型微调、文档切分算法、文本相似度计算方法、Query理解、意图识别、混合检索等。
增强策略。上下文组织方法、提示词工程、重排算法、如何利用专业领域知识针对性微调底座大模型等。

6.2 Agent开发
框架与工具。了解ReAct范式，相关工具使用等。

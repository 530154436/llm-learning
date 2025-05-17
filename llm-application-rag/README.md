
天池-基于LLM智能问答系统学习赛
https://tianchi.aliyun.com/competition/entrance/532172/information
https://github.com/Tongyi-EconML/FinQwen?spm=a2c22.12281976.0.0.3ff94ea4Wg1dNy

[Coggle 30 Days of ML（24年1/2月）：动手学RAG](http://discussion.coggle.club/t/topic/30/1)
[RAG比赛-基于运营商文本数据的知识库检索](https://www.datafountain.cn/competitions/1045/datasets)

[awesome-papers-for-rag](https://github.com/gomate-community/awesome-papers-for-rag/tree/main)


[实现本地 RAG 服务：整合 Open WebUI、Ollama 和 Qwen2.5](https://cuterwrite.xlog.page/integrate-open-webui-ollama-qwen25-local-rag)
使用以下关键工具：
Open WebUI : 提供用户与模型交互的 web 界面。
Ollama : 用于管理 embedding 和大语言模型的模型推理任务。其中 Ollama 中的 bge-m3 模型将用于文档检索，Qwen2.5 将负责回答生成。
Qwen2.5 : 模型部分使用阿里推出的 Qwen 2.5 系列，为检索增强生成服务提供自然语言生成。
为了实现 RAG 服务，我们需要以下步骤：

部署 Open WebUI 作为用户交互界面。
配置 Ollama 以高效调度 Qwen2.5 系列模型。
使用 Ollama 配置的名为 bge-m3 的 embedding 模型实现检索向量化处理。

综述
https://github.com/lizhe2004/Awesome-LLM-RAG-Application


框架
https://github.com/gomate-community/TrustRAG
https://github.com/langgenius/dify
https://github.com/infiniflow/ragflow/tree/main
https://github.com/labring/FastGPT/blob/main/README.md

嵌入向量
SentenceTransformer

[大模型RAG的迭代路径](https://mp.weixin.qq.com/s/kTZc1UpAzpSNanRx82ZRtg)
[面向大语言模型的检索增强生成技术：综述 [译]](https://baoyu.io/translations/ai-paper/2312.10997-retrieval-augmented-generation-for-large-language-models-a-survey)
[一文读懂「RAG，Retrieval-Augmented Generation」检索增](https://download.csdn.net/blog/column/12545383/135714213)
[模块化RAG：RAG新范式，像乐高一样搭建 万字长文](https://www.53ai.com/news/RAG/2024080440218.html)
[GoMate是一款配置化模块化的Retrieval-Augmented Generation (RAG) 框架](https://github.com/gomate-community/GoMate)
[来自工业界的知识库 RAG 服务(五)，模块化知识库 GoMate 实现方案详解](https://github.com/gomate-community/GoMate/tree/main)
[来自工业界的开源知识库 RAG 项目最全细节对比](https://hustyichi.github.io/2024/07/08/compare/)
[FastGPT-基于 LLM 大语言模型的知识库问答系统](https://github.com/labring/FastGPT)

文档解析
[pymupdf](https://products.documentprocessing.com/zh/parser/python/pymupdf/)
[pdfplumber](https://github.com/jsvine/pdfplumber/)
[pdfplumber说明文档翻译](https://blog.csdn.net/hbh112233abc/article/details/125521584)
[【预处理】大模型下开源文档解析工具总结及技术思考](https://mp.weixin.qq.com/s?__biz=Mzg4NjI0NDg0Ng==&mid=2247484415&idx=1&sn=6b2a075e77c3355344d2d40d5d84e45c&chksm=cf9dd77ef8ea5e68f048a3d6f5caca04fe87885efe00ca7d347f6fc9a86cc43ab40a11fa0990#rd)
[Python处理PDF的第三方库对比](https://dothinking.github.io/2021-01-02-Python%E5%A4%84%E7%90%86PDF%E7%9A%84%E7%AC%AC%E4%B8%89%E6%96%B9%E5%BA%93%E5%AF%B9%E6%AF%94/)


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


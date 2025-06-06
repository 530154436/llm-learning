
## 一、预训练模型
### 1.1 Transformer计算注意力分数时为什么不直接使用输入向量？
Transformer在计算注意力分数时引入Query（Q）、Key（K）、Value（V）三个变换矩阵而非直接使用输入向量，主要基于以下原因：

1. **动态表征的必要性**
   - 输入向量（如词嵌入）是静态表示，仅包含词本身的语义和位置信息，无法根据不同上下文动态调整，存在“一词多义”的缺陷。 如"apple"在"水果"和"科技公司"语境中需要不同表征。
   - 通过Q、K、V三个可学习的线性变换矩阵，模型能够根据具体任务需求动态调整不同维度的权重，从而突出某些重要的语义特征。这样，每个token的表征会随着上下文的变化而动态演化，更精准地捕捉语义信息。
 
2. **数学优化考量**
   - 若直接使用原始向量进行点积，会存在三大问题：<br>
     维度灾难：在高维空间中，向量点积的结果可能会变得非常大，这会导致数值计算的不稳定。<br>
     分布偏移：不同层输出的向量分布不一致，这使得模型在学习过程中难以收敛，增加了训练的难度。<br>
     梯度不稳定：未缩放的相似度在反向传播过程中会产生问题，过大或过小的相似度会导致梯度消失或梯度爆炸，影响模型的训练效果。<br>
   - 引入Q、K、V变换矩阵后，可以对向量进行合理的变换和缩放，有效缓解上述问题，提高模型的稳定性和训练效率。

3. **支持多头注意力**
   - 必须通过不同变换矩阵才能实现：<br>
     多子空间并行计算：不同的Q、K、V矩阵可以将输入向量投影到多个不同的子空间中，在这些子空间中并行计算注意力分数，从而提高计算效率和模型的并行性。<br>
     捕获多样化依赖关系：不同的子空间可以关注输入序列中的不同方面，如语法、语义等，使得模型能够捕获多样化的依赖关系，提升模型的表达能力。

4. 架构优势分析 

| 设计维度  | 直接使用输入向量                     | Q/K/V变换机制               | 改进效果                   |
|-------|------------------------------|-------------------------|------------------------|
| 计算效率  | O(1)参数访问，计算简单但表达能力有限         | O(d_model×d_k)参数        | 牺牲少量效率换取性能             |
| 多义性处理 | 只能提供固定单一的表征，无法处理一词多义情况       | 能够根据上下文动态调整表征           | 据相关实验表明，语义理解能力可提升30%以上 |
| 长程依赖  | 主要依赖原始位置码来处理序列信息，对长序列的处理能力有限 | 通过注意力权重自适应地捕捉序列中的长程依赖关系 | 突破RNN的梯度消失限制           |
| 可解释性  | 难以可视化                        | 可以通过分析注意力热图来理解模型的决策过程   | 支持模型诊断，增强了模型的可解释性      |
 

### 1.2 loss出现 NAN
```
epoch: 2, Current lr : [9.992641648270787, 9.992641648270787], train_loss: nan, val_loss: nan
```
原因：学习率太高。

>参考：[从人脑到Transformer：轻松理解注意力机制中的QKV](https://zhuanlan.zhihu.com/p/688660519)

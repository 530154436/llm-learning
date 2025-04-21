

### 缩放点积注意力机制

> ![缩放点积注意力机制](./assets/image-20241024010439683.png)

给定查询矩阵 $Q$、键矩阵 $K$ 和值矩阵 $V$, 其注意力输出的数学表达式如下：

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

- **$Q$（Query）**: 用于查询的向量矩阵。
- **$K$（Key）**: 表示键的向量矩阵，用于与查询匹配。
- **$V$（Value）**: 值矩阵，注意力权重最终会作用在该矩阵上。
- **$d_k$**: 键或查询向量的维度。

> 理解 Q、K、V 的关键在于代码，它们实际上是通过线性变换从输入序列生成的，“故事”的延伸更多是锦上添花。

#### 公式解释

1. **点积计算（Dot Produce）**
   
   将查询矩阵 $Q$ 与键矩阵的转置 $K^\top$ 做点积，计算每个查询向量与所有键向量之间的相似度：
   
   $`\text{Scores} = Q K^\top`$
   
   - **每一行**表示某个查询与所有键之间的相似度（匹配分数）。
   - **每一列**表示某个键与所有查询之间的相似度（匹配分数）。
   
2. **缩放（Scaling）**
   
   > We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $\sqrt{d_k}$ .
   
   当 $d_k$ 较大时，点积的数值可能会过大，导致 Softmax 过后的梯度变得极小，因此除以 $\sqrt{d_k}$ 缩放点积结果的数值范围：
   
   $`\text{Scaled Scores} = \frac{Q K^\top}{\sqrt{d_k}}`$
   
   缩放后（Scaled Dot-Product）也称为注意力分数（**attention scores**）。
   
3. **Softmax 归一化**
   
   使用 Softmax 函数将缩放后的分数转换为概率分布：
   
   $`\text{Attention Weights} = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right)`$
   
   > **注意**：Softmax 是在每一行上进行的，这意味着每个查询的匹配分数将归一化为概率，总和为 1。
   
4. **加权求和（Weighted Sum）**
   
   最后，使用归一化后的注意力权重对值矩阵 $V$ 进行加权求和，得到每个查询位置的最终输出：
   $`\text{Output} = \text{Attention Weights} \times V`$

#### 代码实现

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力计算。
    
    参数:
        Q: 查询矩阵 (batch_size, seq_len_q, embed_size)
        K: 键矩阵 (batch_size, seq_len_k, embed_size)
        V: 值矩阵 (batch_size, seq_len_v, embed_size)
        mask: 掩码矩阵，用于屏蔽不应该关注的位置 (可选)

    返回:
        output: 注意力加权后的输出矩阵
        attention_weights: 注意力权重矩阵
    """
    embed_size = Q.size(-1)  # embed_size
    
    # 计算点积并进行缩放
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(embed_size)

    # 如果提供了掩码矩阵，则将掩码对应位置的分数设为 -inf
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 对缩放后的分数应用 Softmax 函数，得到注意力权重
    attention_weights = F.softmax(scores, dim=-1)

    # 加权求和，计算输出
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

**解释**

1. **缩放点积计算**

   使用 `torch.matmul(Q, K.transpose(-2, -1))` 计算查询与键之间的点积相似度，然后结果除以 $\sqrt{d_k}$ 进行缩放。

2. **掩码处理（Masked Attention）**

   如果提供了掩码矩阵（`mask`），则将掩码为 0 的位置的分数设为 $-\infty$（-inf）。这样在 Softmax 归一化时，这些位置的概率会变为 0，不参与输出计算：

   ```python
   if mask is not None:
       scores = scores.masked_fill(mask == 0, float('-inf'))
   ```

   > Softmax 函数的数学定义为：
   > 
   > $`\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}`$
   > 
   > 当某个分数为 $-\infty$ 时, $e^{-\infty} = 0$, 因此该位置的权重为 0。
   
3. **Softmax 归一化**

   Softmax 将缩放后的分数转换为概率分布（对行），表示每个查询向量与键向量之间的匹配程度：

   ```python
   attention_weights = F.softmax(scores, dim=-1)
   ```

4. **加权求和（Weighted Sum）**

   使用注意力权重对值矩阵 $V$ 进行加权求和，生成最终的输出：

   ```python
   output = torch.matmul(attention_weights, V)
   ```

#### Q: 为什么需要 Mask 机制？

- **填充掩码（Padding Mask）**

  在处理不等长的输入序列时，需要使用填充符（padding）补齐短序列。在计算注意力时，填充部分不应对结果产生影响（q 与填充部分的 k 匹配程度应该为 0），因此需要使用填充掩码忽略这些位置。

- **未来掩码（Look-ahead Mask）**

  > ![Mask](./assets/image-20241028152056813.png)
  
  在训练自回归模型（如 Transformer 中的解码器）时，为了防止模型“偷看”未来的词，需要用掩码屏蔽未来的位置，确保模型只能利用已知的上下文进行预测。

> [!note]
>
> 常见注意力机制除了缩放点积注意力，还有**加性注意力**（Additive Attention）注意力机制。

“那么 Q、K、V 到底是怎么来的？论文架构图中的三种 Attention 是完全不同的架构吗？”

让我们**带着疑惑往下阅读**，先不谈多头，理清楚 Masked，Self 和 Cross 注意力到底是什么。

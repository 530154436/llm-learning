### 3.4 Unigram
Unigram Language Model (ULM)模型是Kudo提出的。当时主要是为了解决机器翻译中分词的问题。作者使用一种叫做marginalized likelihood的方法来建模翻译问题，考虑到不同分词结果对最终翻译结果的影响，引入了分词概率。

与WordPiece一样，Unigram Language Model(ULM)同样使用语言模型来挑选子词。不同之处在于，BPE和WordPiece算法的词表大小都是从小到大变化，属于增量法。而Unigram Language Model则是`减量法`,即先初始化一个大词表，根据评估准则不断丢弃词表，直到满足限定条件。ULM算法考虑了句子的不同分词可能，因而能够输出带概率的多个子词分段。

> **参考文献：**
> - [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/pdf/1804.10959)


对于句子 $S$ , $\(\vec x=(x_{1},x_{2},...,x_{m})\)$ 为句子的一个分词结果，由 $m$ 个子词组成。所以，当前分词下句子 $S$ 的似然值可以表示为：

$$
P(\vec x)=\prod_{i=1}^m{P(x_{i})}
$$

对于句子 $S$ ，挑选似然值最大的作为分词结果，则可以表示为

$$
x^{*}=arg max_{x \in U(x)} P(\vec x)
$$

其中 $U(x)$ 包含了句子的所有分词结果。在实际应用中，词表大小有上万个，直接罗列所有可能的分词组合不具有操作性。针对这个问题，可通过维特比算法得到 $x^*$ 来解决。

那怎么求解每个子词的概率 $P(x_{i})$ 呢？ULM通过EM算法来估计。假设当前词表 $V$ , 则 $M$ 步最大化的对象是如下似然函数：

$$
L=\sum_{s=1}^{|D|}log(P(X^{(s)}))=\sum_{s=1}^{|D|}log(\sum_{x \in U(X^{(s)})}P(x))
$$

其中， $|D|$ 是语料库中语料数量。上述公式的一个直观理解是，将语料库中所有句子的所有分词组合形成的概率相加。
但是，初始时，词表 $V$ 并不存在。因而，ULM算法采用不断迭代的方法来构造词表以及求解分词概率：

1. 初始时，**建立一个足够大的词表**。<br>
   一般，可用语料中的所有字符加上常见的子字符串初始化词表，也可以通过BPE算法初始化。
2. 针对当前词表，用**EM算法**求解每个子词在语料上的概率。
3. 对于每个子词，计算当该子词被从词表中移除时，总的loss降低了多少，记为该子词的**loss**。
4. 将子词按照loss大小进行排序，**丢弃一定比例loss最小的子词**(比如20%)，保留下来的子词生成新的词表。<br>
   这里需要注意的是，单字符不能被丢弃，这是为了避免OOV情况。
5. 重复步骤2到4，直到词表大小减少到设定范围。

可以看出，ULM会保留那些以较高频率出现在很多句子的分词结果中的子词，因为这些子词如果被丢弃，其损失会很大。有以下优点：
+ 使用的训练算法可以利用所有可能的分词结果，这是通过data sampling算法实现的。
+ 提出一种基于语言模型的分词算法，这种语言模型可以给多种分词结果赋予概率，从而可以学到其中的噪声。

## 参考引用
[1] [预训练分词Subword](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/subword.html#unigram-language-model-ulm)<br>
[2] [transformers-WordPiece tokenization 算法](https://huggingface.co/learn/llm-course/zh-CN/chapter6/6?fw=pt)<br>

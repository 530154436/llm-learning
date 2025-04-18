<nav>
<a href="#unigram">Unigram</a><br/>
<a href="#参考引用">参考引用</a><br/>
</nav>

## Unigram
Unigram Language Model (ULM)模型是Kudo提出的。当时主要是为了解决机器翻译中分词的问题。作者使用一种叫做marginalized likelihood的方法来建模翻译问题，考虑到不同分词结果对最终翻译结果的影响，引入了分词概率。

与WordPiece一样，Unigram Language Model(ULM)同样使用语言模型来挑选子词。不同之处在于，BPE和WordPiece算法的词表大小都是从小到大变化，属于增量法。而Unigram Language Model则是**减量法**,即先初始化一个大词表，根据评估准则不断丢弃词表，直到满足限定条件。ULM算法考虑了句子的不同分词可能，因而能够输出带概率的多个子词分段。

> **参考文献：**
> - [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/pdf/1804.10959)


对于句子 $S$ , $\vec x=(x_{1},x_{2},...,x_{m})$ 为句子的一个分词结果，由 $m$ 个子词组成。并且假设每个子词都与其之前的子词独立。所以，当前分词下句子 $S$ 的似然值可以表示为：

$$
P(\vec x)=\prod_{i=1}^m{P(x_{i})}
$$

对于句子 $S$ ，挑选似然值最大的作为分词结果，则可以表示为

$$
x^{*}=arg max_{x \in U(x)} P(\vec x)
$$

其中 $U(x)$ 包含了句子的所有分词结果。在实际应用中，词表大小有上万个，直接罗列所有可能的分词组合不具有操作性。针对这个问题，可通过**维特比算法**得到 $x^*$ 来解决。

那怎么求解每个子词的概率 $P(x_{i})$ （模型参数）呢？ULM通过EM算法来估计。假设当前词表 $V$ , 则 $M$ 步最大化的对象是如下似然函数：

$$
L=\prod_{s=1}^{|D|}P(X^{(s)})=\prod_{s=1}^{|D|}\prod_{x \in U(X^{(s)})}P(x)
$$

其中， $|D|$ 是语料库中语料数量。上述公式的一个直观理解是，将语料库中所有句子的所有分词组合形成的概率相加。
转换为负对数形式后，最小化损失（loss）函数：

$$
-L = -\sum_{s=1}^{|D|}log(P(X^{(s)})) = -\sum_{s=1}^{|D|}log(\sum_{x \in U(X^{(s)})}P(x))
$$

但是，初始时，词表 $V$ 并不存在。因而，ULM算法采用不断迭代的方法（EM算法）来构造词表以及求解子词概率，步骤如下：

1. **初始化词汇表**（初始化分布参数）<br>
   构建初始词汇表：通常包含语料库中的所有单个字符（如字母、汉字）和高频的短子词组合。也可以通过BPE算法初始化。（这里针对每个词（假设长度为n），均从中取长度[2,n-1]的子串。）<br>
   初始化子词概率：为每个子词分配一个初始概率值，通常基于训练数据中的频次统计或均匀分布。<br>
2. **迭代直到收敛**<br>
   + 1）计算期望（E），即计算损失：<br>
   基于当前词表、子词概率（模型参数），对每个句子用`维特比算法`计算出使句子loss最小的分词组合，从而得到整个语料库的loss。<br>
   计算每个子词从词表中移除时语料库的loss减少量，记为该子词的**loss**。<br>
   + 2）最大化（M），即最小化loss：<br>
   将子词按照loss大小进行排序，**丢弃一定比例loss最小的子词**(比如20%)，保留下来的子词生成新的词表，同时更新子词概率。<br>
   这里需要注意的是，单字符不能被丢弃，这是为了避免OOV情况。
   + 3）M 步上找到的词表、子词概率（参数估计值）用于下一个 E 步计算中，这个过程不断交替进行。

示例1 维特比算法分词：
```
假设语料库预分词后得到的单词及其频次的集合：
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)

取这个语料库中所有的子串作为初始词汇库：
["h", "u", "g", "hu", "ug", "p", "pu", "n", "un", "b", "bu", "s", "hug", "gs", "ugs"]

词汇库中所有可能出现子词的频率（所有频率之和为 210）：
("h", 15) ("u", 36) ("g", 20) ("hu", 15) ("ug", 20) ("p", 17) ("pu", 17) ("n", 16)
("un", 16) ("b", 4) ("bu", 4) ("s", 5) ("hug", 15) ("gs", 5) ("ugs", 5)

为了对一个给定的单词进行分词，我们会查看所有可能的分词组合，并根据 Unigram 模型计算出每种可能的概率。
由于所有的分词都被视为独立的，因此这个单词分词的概率就是每个子词概率的乘积。以 "pug" 为例，我们得到的各种可能分词方式的概率如下：
P(["p","u","g"]) = P("p") × P("u") × P("g") = (17/210) *  (36/210) * (20/210) = 0.001321
P(["pu","g"]) = P("pu") × P("g") = (17/210) * (20/210) = 0.007709
P(["p","ug"]) = P("p") × P("ug") = (17/210) * (20/210) = 0.007709
因此， "pug" 将被分词为 ["p", "ug"] 或 ["pu", "g"]。

在这个例子中，找出所有可能的分词方式并计算其概率是容易的，但在语料库比较大的情况下有些困难。有一个经典的算法可以用来计算这个概率，叫做 `Viterbi 算法` 。
通过创建一个图来表示一个给定单词的所有可能分词情况，其中：如果从字符 `a` 到字符 `b` 的子词存在于词汇表中，则在图中存在一条从 `a` 到 `b` 的边。每条边代表从 `a` 到 `b` 进行切分的概率。

为了在该图中找到得分最高的路径（即最优分词），`Viterbi算法`会执行以下步骤：
1. **初始化**：为句子中的每个位置确定最佳得分分割点。
2. **遍历与计算**：从句子的开始到结束，对于每个位置，遍历所有以当前位置结束的子词，并结合这些子词起始位置的最佳得分来计算最高得分。
3. **回溯路径**：一旦到达句子的末尾，只需回溯之前记录的路径，即可得到最终的最优路径，即得分最高的分词组合。
以 "unhug" 为例，对于每个位置，最佳切分子词的分数如下：
Character 0 (u): ["u"] (score 0.171429)
Character 1 (n): ["un"] (score 0.076191)
Character 2 (h): ["un","h"] (score 0.005442)
Character 3 (u): ["un","hu"] (score 0.005442)
Character 4 (g): ["un","hug"] (score 0.005442)
```

示例2 EM算法E步和M步：
```
语料库中的每个词都有一个分数，损失（loss）值是这些分数的负对数似然——即所有词的语料库中所有词的 -log(P(word)) 总和
每个单词的分词及其相应的得分如下：
"hug": ["hug"] (score 0.071428)
"pug": ["pu", "g"] (score 0.007710)
"pun": ["pu", "n"] (score 0.006168)
"bun": ["bu", "n"] (score 0.001451)
"hugs": ["hug", "s"] (score 0.001701)

因此，损失值（loss）是：
10 * (-log(0.071428)) + 5 * (-log(0.007710)) + 12 * (-log(0.006168)) + 4 * (-log(0.001451)) + 5 * (-log(0.001701)) = 169.8

需要计算移除每个 token 对损失值的影响。
例如，“pug”可以被分词为 ["pu", "g"] ，也可以被分词为 ["p", "ug"] ，获得的分数是相同的。因此，去除词汇表中的 "pu" 损失值还会是一样的。
但是，去除 "hug" 之后，损失会变得更糟，因为 "hug" 和 "hugs" 的 tokenization 会变成：
"hug": ["hu", "g"] (score 0.006802)
"hugs": ["hu", "gs"] (score 0.001701)
这些变化将导致损失增加：
- 10 * (-log(0.071428)) + 10 * (-log(0.006802)) = 23.5
因此， "pu" tokens 可能会从词汇表中移除，但 "hug" 则不会。
```

代码实现：
```python
import copy
from collections import defaultdict
from typing import Dict, Tuple, List, Iterable, Union


def prepare_corpus(model: str = "openai-community/gpt2", use_cache: bool = True):
    """
    预分词：
    openai-community/gpt2
    google-bert/bert-base-cased
    xlnet-base-cased:
    """

    # 已预分词的结果
    pre_tokens = [
          ['▁This', '▁is', '▁the', '▁Hugging', '▁Face', '▁Course.'],
          ['▁This', '▁chapter', '▁is', '▁about', '▁tokenization.'],
          ['▁This', '▁section', '▁shows', '▁several', '▁tokenizer', '▁algorithms.'],
          ['▁Hopefully,', '▁you', '▁will', '▁be', '▁able', '▁to', '▁understand', '▁how', '▁they', '▁are', '▁trained', '▁and', '▁generate', '▁tokens.']
      ]
    for tokens in pre_tokens:
        yield tokens

        
def init_vocab(corpus: Iterable, max_vocab_size: int = 300) -> Tuple[Dict[str, int], Dict[str, list]]:
    """
    计算初始词表
    """
    word_freqs = defaultdict(int)       # 单词-频率
    char_freqs = defaultdict(int)       # 字符-频率
    subwords_freqs = defaultdict(int)   # 子词-频率

    # 计算语料库中每个单词的频率
    for sentence in corpus:
        for word in sentence:
            word_freqs[word] += 1

    # 计算语料库中每个字符、子词的频率
    for word, freq in word_freqs.items():
        for i in range(len(word)):
            char_freqs[word[i]] += freq
            # 循环遍历长度至少为2的子词
            for j in range(i + 2, len(word) + 1):
                subwords_freqs[word[i:j]] += freq

    # 按频率对子词排序，用最优的子词对字符进行分组，获得大小为 300 的初始词汇表。
    sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)
    token_freqs = list(char_freqs.items()) + sorted_subwords[:max_vocab_size - len(char_freqs)]
    token_freqs = {token: freq for token, freq in token_freqs}

    print(f"初始词表({len(token_freqs)}): ", token_freqs)
    return word_freqs, token_freqs


def viterbi(word: str, model: Dict[str, float]) -> Tuple[List[str], int]:
    """
    Viterbi算法（动态规划）
    核心思路：
        将问题转化为一个路径搜索问题，把字符串的每个位置看作一个节点，从字符串的起始位置开始，逐步尝试不同的子词切分，记录到达每个位置的最小得分和对应的切分方式。
    具体步骤如下：
        初始化一个数组 dp，用于记录到达每个位置的最小得分，以及一个数组 path 用于记录对应的切分方式。
        遍历字符串的每个位置，对于每个位置，尝试所有可能的子词切分，更新 dp 数组和 path 数组。
        最后，根据 path 数组回溯得到最优切分方式。
    :param word: 字符串
    :param model:  子词对应的概率
    :return:
    """
    n = len(word)
    dp = [float("inf")] * (n + 1)
    path = [-1] * (n + 1)
    dp[0] = 0

    # 动态规划求解
    for i in range(1, n + 1):
        for j in range(i):
            token = word[j: i]
            if token in model:
                score = dp[j] + model[token]
                if score < dp[i]:
                    dp[i] = score
                    path[i] = j
    if path[-1] == -1:
        return ["<unk>"], None

    # 回溯得到最优切分方式
    optimal_split = []
    i = n
    while i > 0:
        prev = path[i]
        optimal_split.append(word[prev:i])
        i = prev
    optimal_split.reverse()
    return optimal_split, dp[n]


def compute_loss(model: Dict[str, float], word_freqs: Dict[str, int]) -> float:
    """
    计算语料库上的分词损失
    """
    loss = 0
    for word, freq in word_freqs.items():
        _, word_loss = viterbi(word, model)
        loss += freq * word_loss
    return loss


def compute_scores(model: Dict[str, float], word_freqs: Dict[str, int]):
    """
    计算每个子词的分数：移除该子词后模型对数似然的下降量。
    """
    scores = {}
    model_loss = compute_loss(model, word_freqs)
    for token, score in model.items():
        # 我们将保留长度为 1 的 tokens
        if len(token) == 1:
            continue
        model_without_token = copy.deepcopy(model)
        _ = model_without_token.pop(token)
        scores[token] = compute_loss(model_without_token, word_freqs) - model_loss
    return scores


def tokenize(sentence: Iterable, model: Dict[str, float]):
    encoded_words = [viterbi(word, model)[0] for word in sentence]
    return sum(encoded_words, [])


def main(vocab_size: int = 100, percent_to_remove: float = 0.1):
    # 构建初始词汇表
    sentences = prepare_corpus("xlnet-base-cased")
    word_freqs, token_freqs = init_vocab(sentences)

    # 训练: EM算法迭代优化概率
    # 初始化子词概率
    total_sum = sum([freq for token, freq in token_freqs.items()])
    model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}
    iteration = 1
    while len(model) > vocab_size:
        scores = compute_scores(model, word_freqs)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])  # loss减少量=移除后loss-移除前loss
        # 删除分数最低（loss减少量最小）的percent_to_remov tokens 。
        remove_sub_words = []
        for i in range(int(len(model) * percent_to_remove)):
            remove_sub_word = sorted_scores[i][0]
            token_freqs.pop(remove_sub_word)
            remove_sub_words.append(remove_sub_word)

        print(f"第{iteration}迭代：删除子词（{len(remove_sub_words)}）={remove_sub_words}")
        # 更新子词概率
        total_sum = sum([freq for token, freq in token_freqs.items()])
        model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}

        iteration += 1
    print(f"最终词表({len(model)}): {list(model.keys())}")

    # 预测
    # ['▁This', '▁is', '▁the', '▁Hugging', '▁Face', '▁Course.']
    sentence = ['▁This', '▁is', '▁the', '▁Hugging', '▁Face', '▁course.']
    predict = tokenize(sentence=sentence, model=model)
    # ['▁This', '▁is', '▁the', '▁Hugging', '▁Face', '▁', 'c', 'ou', 'r', 's', 'e', '.']
    print(f"预测输出: {predict}")
```
输出结果：
```
初始词表(300):  {'▁': 62, 'T': 6, 'h': 18, 'i': 26, 's': 26, 't': 28, ..., 'severa': 2, 'several': 2}
第1迭代：删除子词（30）=['is', 'to', '▁T', '▁Th', '▁Thi', 'Th', 'Thi', 'This', 'hi', 'his', 'se', '▁tok', ..]
第2迭代：删除子词（27）=['tokeniz', 'okeni', 'okeniz', 'keni', 'keniz', 'eni', 'eniz', 'ni', 'niz', 'iz', ...]
第3迭代：删除子词（24）=['Hu', 'Hug', 'Hugg', 'Huggi', 'Huggin', 'Hugging', 'ug', 'ugg', 'uggi', 'uggin', ..]
第4迭代：删除子词（21）=['Fac', 'Face', 'ac', 'ace', 'ce', '▁C', '▁Co', '▁Cou', '▁Cour', '▁Cours', '▁Course', ..]
第5迭代：删除子词（19）=['ur', 'urs', 'urse', 'urse.', 'rse', 'rse.', 'se.', 'e.', '▁c', '▁ch', '▁cha', '▁chap', ..]
第6迭代：删除子词（17）=['chapter', 'ha', 'hap', 'hapt', 'hapte', 'hapter', 'ap', 'apt', 'apte', 'apter', 'pt', ..]
第7迭代：删除子词（16）=['abou', 'about', 'bo', 'bou', 'bout', 'out', 'ut', '▁tokeniza', '▁tokenizat', '▁tokenizati', ..]
第8迭代：删除子词（14）=['tokenization', 'tokenization.', 'okeniza', 'okenizat', 'okenizati', 'okenizatio', 'okenization', ..]
第9迭代：删除子词（13）=['eniza', 'enizat', 'enizati', 'enizatio', 'enization', 'enization.', 'niza', 'nizat', 'nizati', ..]
第10迭代：删除子词（11）=['izat', 'izati', 'izatio', 'ization', 'ization.', 'za', 'zat', 'zati', 'zatio', 'zation', 'zation.']
第11迭代：删除子词（10）=['ati', 'atio', 'ation', 'ation.', 'tion.', 'ion.', 'on.', 'n.', '▁sec', '▁sect']
第12迭代：删除子词（9）=['▁secti', '▁sectio', 'sec', 'sect', 'secti', 'sectio', 'section', 'ec', 'ect']
第13迭代：删除子词（8）=['ecti', 'ectio', 'ection', 'ct', 'cti', 'ctio', 'ction', '▁sh']
第14迭代：删除子词（8）=['▁sho', '▁show', 'sh', 'sho', 'show', 'shows', 'hows', 'ows']
第15迭代：删除子词（7）=['ws', '▁sev', '▁seve', '▁sever', '▁severa', 'sev', 'seve']
最终词表(66): ['▁', 'T', 'h', 'i', 's', 't', 'e', 'H', 'u', 'g', 'n', 'F', 'a', 'c', 'C', 'o', 'r', '.', 'p', 'b', 
             'k', 'z', 'w', 'v', 'l', 'm', 'f', 'y', ',', 'd', '▁t', 'er', '▁a', '▁to', 'en', '▁This', 'th', 'ou', 
             '▁token', 'ra', 'nd', '▁is', '▁the', '▁H', 'in', 'te', '▁ab', '▁tokeniz', 'how', 'era', 'al', 's.', 
             'll', 'and', '▁Hugging', '▁Face', '▁Course.', '▁chapter', '▁about', '▁tokenization.', '▁section', 
             '▁shows', '▁several', 'sever', 'severa', 'several']
预测输出: ['▁This', '▁is', '▁the', '▁Hugging', '▁Face', '▁', 'c', 'ou', 'r', 's', 'e', '.']

Process finished with exit code 0

```
## 参考引用
[1] [预训练分词Subword](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/subword.html#unigram-language-model-ulm)<br>
[2] [transformers-Unigram tokenization 算法](https://huggingface.co/learn/llm-course/zh-CN/chapter6/7?fw=pt)<br>
[3] [搜狗百科-em算法](https://baike.sogou.com/v9130286.htm?ch=frombaikevr&fromTitle=em%E7%AE%97%E6%B3%95)<br>
[4] [文本挖掘的分词原理](https://www.cnblogs.com/pinard/p/6677078.html)<br>

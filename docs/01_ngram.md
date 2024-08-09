# 1.ngram

## 1.统计语言模型

### 1.1 介绍

通俗的说，语言模型就是用来计算一个句子的概率的**概率模型**，它通常被描述为字符串`s`的概率分布$P(s)$。这里的字符串`s`由`l`个“基元”组成，基元可以是字、词或者短语等。于是句子`s`的概率可以表示为所有基元的联合概率分布：

$$
p(s)=p\left(w_{1}, w_{2}, w_{3}, \ldots, w_{l}\right)
$$

利用bayes公式将其转变为：

$$
\begin{aligned} p(s) & =p\left(w_{1}\right) p\left(w_{2} \mid w_{1}\right) p\left(w_{3} \mid w_{1}, w_{2}\right) \ldots p\left(w_{l} \mid w 1, w_{2}, \ldots, w_{l-1}\right) \\ & =\prod_{i=1}^{l} p\left(w_{i} \mid w_{1} \ldots w_{i-1}\right)\end{aligned}
$$

而上式中的条件概率$p(w_1),p(w_2|w_1),p(w_3,|w_1,w_2)...$就是语言模型的参数。当这些模型都计算出的时候，对于任何一个给定的句子`s`都可以通过将对应的条件概率相乘的方法很快得到句子的概率了。不同的模型计算这些参数的值的方式不同，常用的方法有**n-gram，决策树，最大熵模型神经网络**等等。

### 1.2 模型缺点

上述模型看上去非常简单，但是通过简单估算可以发现其中存在的问题。对于一个大小为`N`的词典D来说，一个长度为`l`的句子`s`，这其中需要的参数就是$N^l$个。在这种计算方法中，第`i`个词是由前`i-1`个词(历史词)推出的，随着历史词个数的增长，参数个数按指数级增长。这就会导致两个问题：

- **参数空间过大**，根本无法从训练语料中得到所有的参数，因为大部分的情形在训练语料中根本就没出现
- **数据稀疏严重**，对于没有在语料中出现的组合，根据最大似然估计就会得到这个参数为0，这将导致整个句子的概率为0

## 2.N元语法(N-gram)模型

### 2.1 马尔可夫假设

为了解决之前讲的参数空间过大的问题，引入马尔科夫假设：**一个词出现的概率只和它前面出现的一个或有限的几个词有关**：

$$
p(s)=\prod_{i=1}^{l} p\left(w_{i} \mid w_{1} \ldots w_{i-1}=\prod_{i=1}^{l} p\left(w_{i} \mid w_{i-k}, w_{i-(k-1)}, \ldots, w_{i-1}\right)\right.
$$

可以看到通过这样的方式，模型的参数就得到了大大减少。

### 2.2 N元语法

满足上面的条件的模型就被成为n元语法，其中的n对应考虑的不同历史词个数。通常情况下，n不能取太大的数，否则的话，参数个数过多的问题还是会存在，常用`n=1，2，3`。

`n=1`时的模型被称为一元语言模型(Uni-gram)，它表示的是句子中每个词之间是条件无关的，也就是说所需要考虑的历史词为0，即

$$
p\left(w_{1}, w_{2}, \ldots, w_{l}\right)=p\left(w_{1}\right) p\left(w_{2}\right) \ldots p\left(w_{l}\right)
$$

对于`n≥2`的情况，每个词与其前`n-1`个词有关，即

$$
p\left(w_{1}, w_{2} m . ., w_{l}\right)=\prod_{i=1}^{l+1} p\left(w_{i} \mid w_{i-n+1}^{i-1}\right)
$$

然后根据最大似然估计，这些条件概率的计算就是在语料中统计这些词组合所出现的概率，即：

$$
p\left(w_{k} \mid w_{k-n+1}^{k-1}\right)=\frac{\operatorname{count}\left(w_{k-n+1}^{k}\right)}{\operatorname{count}\left(w_{k-n+1}^{k-1}\right)}
$$

从中可以看出，n的增大会增加模型的复杂度。虽然理论上n越大，模型的效果越好。但是当n很大的时候，需要的训练语料也会很大，而且当n大到一定程度时，其对于模型效果的提升越来越少，**通常使用n=3就已经满足要求了**。

### 2.3 OOV与平滑/回退

当语料库有限，大概率会在实际预测的时候遇到没见过的词或短语，这就是**未登录词（OOV）**，这样就会造成概率计算的公式中，分子或分母为0，毕竟它们都只是频率。分子为0的话，整个句子的概率是连乘出来的结果是0；分母是0的话，数学上就根本没法计算了。

即使是使用n=2的情况，也不能保证语料空间能够涵盖真实的语言模型空间。如果单纯的考虑词频比例作为参数进行计算的话就会碰到两个问题：

- 当$\operatorname{count}\left(w_{k-n+1}^{k}\right)=0$时，是不是就能说明概率$p\left(w_{k} \mid w_{1}^{k-1}\right)$就是0呢？
- 当$\operatorname{count}\left(w_{k-n+1}^{k}\right)=\operatorname{count}\left(w_{k-n+1}^{k-1}\right)$，能否认为$p\left(w_{k} \mid w_{1}^{k-1}\right)$为1呢？

显然上面的两种情况都是不能，考虑0或1这两个极端情况都会导致整个句子的概率过大或过小，这都是不希望看到的。然而，无论语料库有多么大，都无法回避这个问题。**平滑化方法就是处理这种问题的**。

计算n-gram的一个常用工具是[SRILM](http://www.speech.sri.com/projects/srilm/download.html "SRILM")，它是基于C++的，而且里面也集成了很多平滑方法，使用起来非常方便。

#### （1）**平滑（smoothing）**

为每个w对应的Count增加一个很小的值，目的是使所有的 N-gram 概率之和为 1、使所有的 N-gram 概率都不为 0。常见平滑方法：

**Laplace Smoothing**

**Add-one**：即强制让所有的n-gram至少出现一次，只需要在分子和分母上分别做加法即可。这个方法的弊端是，大部分n-gram都是没有出现过的，很容易为他们分配过多的概率空间。

$$
p\left(w_{n} \mid w_{n-1}\right)=\frac{C\left(w_{n-1} w_{n}\right)+1}{C\left(w_{n-1}\right)+|V|}
$$

**Add-K**：在Add-one的基础上做了一点小改动，原本是加1，现在加上一个小于1的常数K。但是缺点是这个常数仍然需要人工确定，对于不同的语料库K可能不同。

$$
p\left(w_{n} \mid w_{n-1}\right)=\frac{C\left(w_{n-1} w_{n}\right)+k}{C\left(w_{n-1}\right)+k|V|}
$$

其他平滑方法：

- Good-Turing smoothing
- Jelinek-Mercer smoothing (interpolation)
- Catz smoothing
- Witten-Bell smoothing
- Absolute discounting
- Kneser-Ney smoothing

#### （2）回退 **（Katz backoff）**

从N-gram回退到(N-1)-gram，例如Count(the,dog)\~=Count(dog)。

### 2.4 N-gram小节

n-gram模型简单来说就是在给定的语料库中统计各个词串出现的次数，并进行适当的平滑处理。之后将这些值存储起来，在计算出现句子的概率时，找到对应的概率参数，相乘即得到总的概率值。

总结下基于统计的 n-gram 语言模型的优缺点：

优点：

- 采用极大似然估计，参数易训练；
- 完全包含了前 n-1 个词的全部信息；
- 可解释性强，直观易理解。

缺点：

- 缺乏长期依赖，只能建模到前 n-1 个词；
- 随着 n 的增大，参数空间呈指数增长；
- 数据稀疏，难免会出现OOV的问题；
- 单纯的基于统计频次，泛化能力差。

## 3.代码解读

### 3.1 详细代码

整体来说，代码比较简单，具体实现了n-gram模型（`NgramModel`）和回退 n-gram 模型（`BackoffNgramModel`，未使用），再训练时也采用`smoothing`的方式处理OOV。详细代码见 `code/01_ngram/ngram.py`，有详细中文注释，部分代码如下：

- n-gram 模型

```python
class NgramModel:
    """ n-gram 模型类。
    """
    def __init__(self, vocab_size, seq_len, smoothing=0.0):
        """初始化 n-gram 模型。

        Args:
            vocab_size (int): 词汇表大小。
            seq_len (int): 序列长度。
            smoothing (float): 平滑系数，默认为 0.0。
        """
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        # 计数数组，形状为 (vocab_size,) * seq_len
        self.counts = np.zeros((vocab_size,) * seq_len, dtype=np.uint32)
        # 用于存储均匀分布的缓冲区，避免每次创建新的数组
        self.uniform = np.ones(self.vocab_size, dtype=np.float32) / self.vocab_size

    def train(self, tape):
        """ 训练模型。

        Args:
            tape (list): 形状为 (seq_len,) 的训练数据序列。
        """
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len
        self.counts[tuple(tape)] += 1

    def get_counts(self, tape):
        """ 获取给定上下文的计数。

        Args:
            tape (list): 形状为 (seq_len - 1,) 的上下文序列。

        Returns:
            np.ndarray: 给定上下文的计数数组。
        """
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len - 1
        return self.counts[tuple(tape)]

    def __call__(self, tape):
        """ 计算下一个 token 的条件概率分布。

        Args:
            tape (list): 形状为 (seq_len - 1,) 的上下文序列。

        Returns:
            np.ndarray: 下一个标记的条件概率分布。
        """

        assert isinstance(tape, list)
        assert len(tape) == self.seq_len - 1
        # 获取计数并应用平滑处理
        counts = self.counts[tuple(tape)].astype(np.float32)
        # 添加平滑处理（"fake counts"）到所有计数
        counts += self.smoothing 
        counts_sum = counts.sum()
        # 如果计数总和大于 0，则计算概率；否则返回均匀分布
        probs = counts / counts_sum if counts_sum > 0 else self.uniform
        return probs
```

### 3.2 代码运行

#### （1）数据集

数据集是2018年来自[ssa.gov](https://www.ssa.gov/oact/babynames/ "ssa.gov")的32,032个名字，这些名字在测试部分分为1000个名字，在val部分分为1000个名字，其余的在训练部分，所有这些名字都在`datasets/data`文件夹中。 因此，n-gram模型本质上将尝试学习这些名称中字符的统计信息，然后通过从模型中采样来生成新的名称。&#x20;

#### （2）代码运行

注意：在 code 目录下运行

```bash
python 01_ngram/ngram.py
```

脚本首先“训练”一个小的字符级Tokenizer(所有26个小写英文字母和换行字符的词汇表大小为27)，然后使用验证分割对n-gram模型进行一个小的网格搜索，该模型具有n-gram阶' n '和平滑因子的各种超参数设置。&#x20;

默认设置的最优值是“n=4，平滑=0.1”。 然后取这个最佳模型，从中抽取200个字符，最后报告测试损失和困惑。 下面是完整的输出，它应该只需要几秒钟就能产生:&#x20;

```bash
(llm) PS llm101n-zh\code> python .\01_ngram\ngram.py
seq_len 3 | smoothing 0.03 | train_loss 2.1843 | val_loss 2.2443
seq_len 3 | smoothing 0.10 | train_loss 2.1870 | val_loss 2.2401
seq_len 3 | smoothing 0.30 | train_loss 2.1935 | val_loss 2.2404
seq_len 3 | smoothing 1.00 | train_loss 2.2117 | val_loss 2.2521
seq_len 4 | smoothing 0.03 | train_loss 1.8703 | val_loss 2.1376
seq_len 4 | smoothing 0.10 | train_loss 1.9028 | val_loss 2.1118
seq_len 4 | smoothing 0.30 | train_loss 1.9677 | val_loss 2.1269
seq_len 4 | smoothing 1.00 | train_loss 2.1006 | val_loss 2.2114
seq_len 5 | smoothing 0.03 | train_loss 1.4955 | val_loss 2.3540
seq_len 5 | smoothing 0.10 | train_loss 1.6335 | val_loss 2.2814
seq_len 5 | smoothing 0.30 | train_loss 1.8610 | val_loss 2.3210
seq_len 5 | smoothing 1.00 | train_loss 2.2132 | val_loss 2.4903
best hyperparameters: {'seq_len': 4, 'smoothing': 0.1}
felton
jasiel
chaseth
nebjnvfobzadon
brittan
shir
esczsvn
freyanty
aubren
malanokhanni
jemxebcghhzhnsurias
lam
coock
braeya
leiazie
ilamil
vleck
plgiavae
ahzai
sire
azari
ril
aqqhvtsmerysteena
jena
jo
test_loss 2.106370, test_perplexity 8.218358
wrote datasets/dev\ngram_probs.npy to disk (for visualization)
```

`4-gram`模型抽样了一些相对合理的名称，如`felton`和`jasiel`，但也抽样了一些更奇怪的名称，如`nebjnvfobzadon`，但不能对一个`4-gram`字符级语言模型期望太高。 最后，测试困惑度报告为`~8.2`，因此模型对测试集中的每个字符都感到困惑，就好像它从8.2个等可能字符中随机选择一样。&#x20;

#### （3）可视化

代码还将n-gram概率写入磁盘到`datasets/dev`文件夹中，然后可以使用dev/visualize\_probs.ipynb文件可视化。

```python
probs = np.load("ngram_probs.npy")
probs.shape
```

> (27, 27, 27, 27)

```python
assert probs.shape == (27, 27, 27, 27)
reshaped = probs.reshape(27**2, 27**2)
plt.figure(figsize=(6, 6))
plt.imshow(reshaped, cmap='hot', interpolation='nearest')
plt.axis('off')
```

> (-0.5, 728.5, 728.5, -0.5)

![](image/image_1rJ6pKjJoI.png)

## 4参考资料

1. [opensource.niutrans.com/mtbook/section2-3.html](https://opensource.niutrans.com/mtbook/section2-3.html "opensource.niutrans.com/mtbook/section2-3.html")
2. [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/3.pdf "Speech and Language Processing")
3. [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo "The spelled-out intro to language modeling: building makemore")

"""
n-gram Language Model

Good reference:
Speech and Language Processing. Daniel Jurafsky & James H. Martin.
https://web.stanford.edu/~jurafsky/slp3/3.pdf

Example run:
python ngram.py
"""

import os
import itertools
import numpy as np

# -----------------------------------------------------------------------------
class RNG:
    """ 随机数生成
        模拟Python中随机数生成，完全确定，能够认为控制，也可以在其他语言中使用
    """
    def __init__(self, seed):
        """初始化RNG实例。

        Args:
            seed (int): 生成随机数序列的初始种子。
        """
        self.state = seed

    def random_u32(self):
        """ 生成一个32位无符号整数的随机数。

        使用Xorshift算法:
        参考: https://en.wikipedia.org/wiki/Xorshift#xorshift*

        在Python中，
        - `0xFFFFFFFFFFFFFFFF` 相当于C中的无符号64位整数转换
        - `0xFFFFFFFF` 相当于C中的无符号32位整数转换。

        Returns:
            int: 32位无符号整数。
        """
        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    def random(self):
        """ 生成一个范围在[0, 1)之间的32位浮点随机数。

        Returns:
            float: 范围在[0, 1)之间的随机数。
        """
        return (self.random_u32() >> 8) / 16777216.0    # 16777216.0 等于 2^24

# -----------------------------------------------------------------------------
# sampling from the model

def sample_discrete(probs, coinf):
    """从离散分布中采样。

    Args:
        probs (list of float): 概率分布列表。
        coinf (float): 从[0, 1)范围内生成的随机数。

    Returns:
        int: 采样的索引。
    """
    cdf = 0.0   # 累积分布函数的当前值
    for i, prob in enumerate(probs):
        cdf += prob
        # 如果随机数小于累积分布函数的当前值，则返回对应的索引
        if coinf < cdf:
            return i
    # 如果由于舍入误差导致没有找到合适的索引，则返回最后一个索引
    return len(probs) - 1

# -----------------------------------------------------------------------------
# models: n-gram 模型，以及一个回退模型，可以使用多个不同阶的 n-gram 模型

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

# currently unused, just for illustration
class BackoffNgramModel:
    """回退 n-gram 模型类。

    这个模型可以用于组合多个不同阶的 n-gram 模型。
    在训练过程中，它会用相同的数据更新所有模型。
    在推理过程中，它会使用具有当前上下文数据的最高阶模型。

    Note: 目前未使用，仅作说明
    """
    def __init__(self, vocab_size, seq_len, smoothing=0.0, counts_threshold=0):
        """初始化回退 n-gram 模型。

        Args:
            vocab_size (int): 词汇表大小。
            seq_len (int): 序列长度。
            smoothing (float): 平滑系数，默认为 0.0。
            counts_threshold (int): 计数阈值，默认为 0。
        """
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.counts_threshold = counts_threshold
        # 创建不同阶的 NgramModel 对象
        self.models = {i: NgramModel(vocab_size, i, smoothing) for i in range(1, seq_len + 1)}

    def train(self, tape):
        """训练模型。

        Args:
            tape (list): 形状为 (seq_len,) 的训练数据序列。
        """
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len
        # 用相同的数据更新所有模型
        for i in range(1, self.seq_len + 1):
            self.models[i].train(tape[-i:])

    def __call__(self, tape):
        """ 计算下一个 token 的条件概率分布。

        Args:
            tape (list): 形状为 (seq_len - 1,) 的上下文序列。

        Returns:
            np.ndarray: 下一个标记的条件概率分布。

        Raises:
            ValueError: 如果找不到适用于当前上下文的模型，则抛出异常。
        """
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len - 1
        # 寻找具有当前上下文数据的最高阶模型
        for i in reversed(range(1, self.seq_len + 1)):
            tape_i = tape[-i+1:] if i > 1 else []
            counts = self.models[i].get_counts(tape_i)
            if counts.sum() > self.counts_threshold:
                return self.models[i](tape_i)
        # 不应该到达这里，因为一元模型应该始终有数据
        raise ValueError("no model found for the current context")

# -----------------------------------------------------------------------------
# data iteration and evaluation utils

# small utility function to iterate tokens with a fixed-sized window
def dataloader(tokens, window_size):
    for i in range(len(tokens) - window_size + 1):
        yield tokens[i:i+window_size]

def eval_split(model, tokens):
    # evaluate a given model on a given sequence of tokens (splits, usually)
    sum_loss = 0.0
    count = 0
    for tape in dataloader(tokens, model.seq_len):
        x = tape[:-1] # the context
        y = tape[-1]  # the target
        probs = model(x)
        prob = probs[y]
        sum_loss += -np.log(prob)
        count += 1
    mean_loss = sum_loss / count if count > 0 else 0.0
    return mean_loss

# -----------------------------------------------------------------------------

# "训练" 分词器，使其能够映射字符到分词
train_text = open('datasets/data/train.txt', 'r').read()
assert all(c == '\n' or ('a' <= c <= 'z') for c in train_text)
uchars = sorted(list(set(train_text))) # 输入文本中出现的字符
vocab_size = len(uchars)
# 建立字符到分词的映射表
char_to_token = {c: i for i, c in enumerate(uchars)}
# 建立分词到字符的映射表
token_to_char = {i: c for i, c in enumerate(uchars)}
# 将\n指定为文本结束token  <|endoftext|>
EOT_TOKEN = char_to_token['\n'] 

# 预先分词所有数据集
test_tokens = [char_to_token[c] for c in open('datasets/data/test.txt', 'r').read()]
val_tokens = [char_to_token[c] for c in open('datasets/data/val.txt', 'r').read()]
train_tokens = [char_to_token[c] for c in open('datasets/data/train.txt', 'r').read()]

# 使用网格搜索在验证集上进行超参数搜索
seq_lens = [3, 4, 5]    # 序列长度候选值
smoothings = [0.03, 0.1, 0.3, 1.0]  # 平滑因子候选值
best_loss = float('inf')    # 初始化损失为无穷大
best_kwargs = {}    # 最佳超参数字典

# 对于每一个序列长度和平滑因子的组合
for seq_len, smoothing in itertools.product(seq_lens, smoothings):
    # 训练N-gram模型
    model = NgramModel(vocab_size, seq_len, smoothing)
    for tape in dataloader(train_tokens, seq_len):
        model.train(tape)
    # 评估训练集和验证集上的损失
    train_loss = eval_split(model, train_tokens)
    val_loss = eval_split(model, val_tokens)
    print("seq_len %d | smoothing %.2f | train_loss %.4f | val_loss %.4f"
          % (seq_len, smoothing, train_loss, val_loss))
    # 更新最佳超参数
    if val_loss < best_loss:
        best_loss = val_loss
        best_kwargs = {'seq_len': seq_len, 'smoothing': smoothing}

# 用最佳超参数重新训练模型使
seq_len = best_kwargs['seq_len']
print("best hyperparameters:", best_kwargs)
model = NgramModel(vocab_size, **best_kwargs)
for tape in dataloader(train_tokens, seq_len):
    model.train(tape)

# 从模型中采样生成文本
sample_rng = RNG(1337)
tape = [EOT_TOKEN] * (seq_len - 1)
for _ in range(200):
    # 获取概率分布
    probs = model(tape)
    # 采样下一个分词
    coinf = sample_rng.random()
    probs_list = probs.tolist()
    next_token = sample_discrete(probs_list, coinf)
    # otherwise update the token tape, print token and continue
    # 更新 token 序列，输出 token 并继续
    next_char = token_to_char[next_token]
    # 更新 token 序列
    tape.append(next_token)
    if len(tape) > seq_len - 1:
        tape = tape[1:]
    print(next_char, end='')
print() # newline

# 最后，在测试集评估
test_loss = eval_split(model, test_tokens)
test_perplexity = np.exp(test_loss)
print("test_loss %f, test_perplexity %f" % (test_loss, test_perplexity))

# 获取最终计数，归一化为概率，并写入文件以便可视化
counts = model.counts + model.smoothing
probs = counts / counts.sum(axis=-1, keepdims=True)
vis_path = os.path.join("datasets/dev", "ngram_probs.npy")
np.save(vis_path, probs)
print(f"wrote {vis_path} to disk (for visualization)")

"""
Implements a simple n-gram language model in PyTorch.
Acts as the correctness reference for all the other versions.
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from common import RNG, StepTimer

# -----------------------------------------------------------------------------
# PyTorch 实现的 MLP n-gram 模型：首先是不使用 nn.Module 的版本

class MLPRaw:
    """
    使用之前的 n 个 token，将它们通过查找表编码，
    将向量连接起来，并通过 MLP 预测下一个 token。

    参考文献：
    Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
    """

    def __init__(self, vocab_size, context_length, embedding_size, hidden_size, rng):
        """
        初始化 MLPRaw 模型的参数。

        参数：
        vocab_size (int): 词汇表的大小
        context_length (int): 上下文长度，即用于预测的 token 数量
        embedding_size (int): 嵌入层的维度
        hidden_size (int): MLP 隐藏层的大小
        rng (RNG): 随机数生成器，用于权重初始化
        """

        v, t, e, h = vocab_size, context_length, embedding_size, hidden_size
        self.embedding_size = embedding_size
        # 初始化词嵌入表
        self.wte = torch.tensor(rng.randn(v * e, mu=0, sigma=1.0)).view(v, e)
        # 初始化第一层全连接层的权重和偏置
        # 这个 scale 用来确保初始化的权重值不会过大或过小。
        scale = 1 / math.sqrt(e * t)
        # weights相当于wx+b函数中的w
        self.fc1_weights = torch.tensor(rng.rand(t * e * h, -scale, scale)).view(h, t * e).T
        # bias相当于wx+b函数中的b
        self.fc1_bias = torch.tensor(rng.rand(h, -scale, scale))
        # 初始化第二层全连接层的权重和偏置
        scale = 1 / math.sqrt(h)
        self.fc2_weights = torch.tensor(rng.rand(v * h, -scale, scale)).view(v, h).T
        self.fc2_bias = torch.tensor(rng.rand(v, -scale, scale))
        # 明确告知 PyTorch 这些是参数，并需要梯度
        for p in self.parameters():
            p.requires_grad = True

    def parameters(self):
        # 返回模型的所有参数
        return [self.wte, self.fc1_weights, self.fc1_bias, self.fc2_weights, self.fc2_bias]

    def __call__(self, idx, targets=None):
        # 调用 forward 方法进行前向传播
        return self.forward(idx, targets)

    def forward(self, idx, targets=None):
        # idx 是输入的 token，(B, T) 形状的整数 tensor
        # targets 是目标 token，(B, ) 形状的整数 tensor
        B, T = idx.size()
        # 前向传播
        # 使用嵌入表编码所有的 token
        emb = self.wte[idx] # (B, T, embedding_size)
        # 将所有嵌入连接在一起
        emb = emb.view(B, -1) # (B, T * embedding_size)
        # 通过 MLP 进行前向传播
        hidden = torch.tanh(emb @ self.fc1_weights + self.fc1_bias)
        logits = hidden @ self.fc2_weights + self.fc2_bias
        # 如果提供了目标，也计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

class MLP(nn.Module):
    """
    MLP n-gram 模型的 PyTorch 实现，使用 nn.Module。

    包含一个嵌入层和一个由两层线性层组成的 MLP。
    """

    def __init__(self, vocab_size, context_length, embedding_size, hidden_size, rng):
        """
        初始化 MLP 模型的参数。

        参数：
        vocab_size (int): 词汇表的大小，即嵌入层的词汇数
        context_length (int): 上下文长度，即用于预测的 token 数量
        embedding_size (int): 嵌入层的维度
        hidden_size (int): MLP 隐藏层的大小
        rng (RNG): 随机数生成器，用于初始化权重
        """
        super().__init__()
        # 初始化 token 嵌入表
        self.wte = nn.Embedding(vocab_size, embedding_size)  # 词汇表大小 x 嵌入维度
        # 初始化 MLP 模型
        self.mlp = nn.Sequential(
            nn.Linear(context_length * embedding_size, hidden_size),  # 第一层线性变换
            nn.Tanh(),  # 激活函数
            nn.Linear(hidden_size, vocab_size)  # 第二层线性变换，输出词汇表大小
        )
        # 使用自定义随机数生成器初始化权重
        self.reinit(rng)

    @torch.no_grad()
    def reinit(self, rng):
        """
        使用自定义随机数生成器初始化模型的权重。

        参数：
        rng (RNG): 随机数生成器，用于初始化权重
        """
        def reinit_tensor_randn(w, mu, sigma):
            """
            用正态分布初始化张量。

            参数：
            w (torch.Tensor): 要初始化的张量
            mu (float): 正态分布的均值
            sigma (float): 正态分布的标准差
            """
            winit = torch.tensor(rng.randn(w.numel(), mu=mu, sigma=sigma))
            w.copy_(winit.view_as(w))

        def reinit_tensor_rand(w, a, b):
            """
            用均匀分布初始化张量。

            参数：
            w (torch.Tensor): 要初始化的张量
            a (float): 均匀分布的下界
            b (float): 均匀分布的上界
            """
            winit = torch.tensor(rng.rand(w.numel(), a=a, b=b))
            w.copy_(winit.view_as(w))

        # 匹配 PyTorch 默认初始化：
        # 嵌入层使用 N(0,1) 初始化
        reinit_tensor_randn(self.wte.weight, mu=0, sigma=1.0)
        # 线性层（W, b）使用 U(-K, K) 初始化，其中 K = 1/sqrt(fan_in)
        scale = (self.mlp[0].in_features)**-0.5
        reinit_tensor_rand(self.mlp[0].weight, -scale, scale)
        reinit_tensor_rand(self.mlp[0].bias, -scale, scale)
        scale = (self.mlp[2].in_features)**-0.5
        reinit_tensor_rand(self.mlp[2].weight, -scale, scale)
        reinit_tensor_rand(self.mlp[2].bias, -scale, scale)

    def forward(self, idx, targets=None):
        """
        执行前向传播。

        参数：
        idx (torch.Tensor): 输入 token 的张量，形状为 (B, T)，其中 B 是批大小，T 是上下文长度
        targets (torch.Tensor, optional): 目标 token 的张量，形状为 (B, )，默认为 None

        返回：
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - logits (torch.Tensor): 模型的输出，形状为 (B, V)，其中 V 是词汇表大小
            - loss (torch.Tensor, optional): 如果提供了目标 token，则返回损失，否则为 None
        """
        B, T = idx.size()
        # 获取嵌入表示
        emb = self.wte(idx)  # 形状为 (B, T, embedding_size)
        # 展平嵌入表示
        emb = emb.view(B, -1)  # 形状为 (B, T * embedding_size)
        # 通过 MLP 计算 logits
        logits = self.mlp(emb)
        loss = None
        # 如果提供了目标 token，则计算交叉熵损失
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss


# -----------------------------------------------------------------------------
# 简单的 DataLoader，用于迭代所有 n-grams

def dataloader(tokens, context_length, batch_size):
    """
    生成用于训练的批量数据。

    参数：
    tokens (list of int): 序列化的 token 列表，表示输入文本
    context_length (int): 上下文长度，即每个输入序列的 token 数量
    batch_size (int): 每个批次的样本数量

    返回：
    Generator[Tuple[torch.Tensor, torch.Tensor]]:
        - inputs (torch.Tensor): 输入张量，形状为 (B, T)，其中 B 是批大小，T 是上下文长度
        - targets (torch.Tensor): 目标张量，形状为 (B, )，其中 B 是批大小
    """

    n = len(tokens)
    inputs, targets = [], []
    pos = 0

    while True:
        # 使用滑动窗口从 token 列表中提取上下文长度 + 1 的窗口
        window = tokens[pos:pos + context_length + 1]
        inputs.append(window[:-1])  # 取上下文部分作为输入
        targets.append(window[-1])  # 取目标 token 作为输出

        # 一旦收集到一个批次，返回该批次
        if len(inputs) == batch_size:
            yield (torch.tensor(inputs), torch.tensor(targets))
            inputs, targets = [], []  # 清空当前批次的输入和目标

        # 移动位置，若到达序列末尾则从头开始
        pos += 1
        if pos + context_length >= n:
            pos = 0


# -----------------------------------------------------------------------------
# 评估函数

@torch.inference_mode()
def eval_split(model, tokens, max_batches=None):
    """
    计算模型在给定 token 上的损失。

    参数：
    model (nn.Module): 要评估的模型
    tokens (list of int): 序列化的 token 列表，表示输入文本
    max_batches (int, optional): 最大批次数。如果为 None，则评估整个数据集；否则，仅评估前 `max_batches` 个批次

    返回：
    float: 平均损失值
    """
    total_loss = 0
    num_batches = len(tokens) // batch_size  # 计算数据集中总的批次数

    # 如果指定了最大批次数，则取最小值以限制评估批次的数量
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)

    # 创建数据迭代器
    data_iter = dataloader(tokens, context_length, batch_size)

    # 遍历批次并计算总损失
    for _ in range(num_batches):
        inputs, targets = next(data_iter)  # 获取一个批次的数据
        logits, loss = model(inputs, targets)  # 计算模型的输出和损失
        total_loss += loss.item()  # 累加损失值

    # 计算平均损失
    mean_loss = total_loss / num_batches
    return mean_loss


# -----------------------------------------------------------------------------
# 从模型中采样

def softmax(logits):
    """
    计算给定 logits 的 softmax 概率分布。

    参数：
    logits (torch.Tensor): 输入的 logits，形状为 (V,) 的一维张量，其中 V 是词汇表大小

    返回：
    torch.Tensor: 计算得到的概率分布，形状为 (V,)
    """
    # 为了数值稳定性，减去 logits 中的最大值
    maxval = torch.max(logits)
    # 计算 exp(logits - maxval)
    exps = torch.exp(logits - maxval)
    # 计算概率分布
    probs = exps / torch.sum(exps)
    return probs

def sample_discrete(probs, coinf):
    """
    从离散概率分布中进行采样。

    参数：
    probs (torch.Tensor): 概率分布，形状为 (V,) 的一维张量，其中 V 是词汇表大小
    coinf (float): 介于 0 和 1 之间的浮点数，用作随机数

    返回：
    int: 采样得到的离散值的索引
    """
    # 累积分布函数
    cdf = 0.0
    # 遍历每个概率值
    for i, prob in enumerate(probs):
        cdf += prob
        # 如果随机数小于当前累积概率，则返回当前索引
        if coinf < cdf:
            return i
    # 处理舍入误差，返回最后一个索引
    return len(probs) - 1


# -----------------------------------------------------------------------------
# 开始训练！

# "训练" Tokenizer，使其能够在字符和令牌之间进行映射
train_text = open('../datasets/data/train.txt', 'r').read()
# 确保训练文本只包含换行符或小写字母
assert all(c == '\n' or ('a' <= c <= 'z') for c in train_text)
# 获取训练文本中的所有唯一字符，并按字母排序
uchars = sorted(list(set(train_text)))
vocab_size = len(uchars)  # 词汇表大小
print(vocab_size)
# 创建字符到令牌的映射
char_to_token = {c: i for i, c in enumerate(uchars)}
# 创建令牌到字符的映射
token_to_char = {i: c for i, c in enumerate(uchars)}
# 将换行符指定为结束标记 (EOT)
EOT_TOKEN = char_to_token['\n']
# 将所有文本预处理为令牌
test_tokens = [char_to_token[c] for c in open('../datasets/data/test.txt', 'r').read()]
val_tokens = [char_to_token[c] for c in open('../datasets/data/val.txt', 'r').read()]
train_tokens = [char_to_token[c] for c in open('../datasets/data/train.txt', 'r').read()]

# 创建模型
context_length = 3  # 如果用 3 个令牌预测第 4 个令牌，这是一个 4-gram 模型
embedding_size = 48  # 嵌入向量的维度
hidden_size = 512  # 隐藏层的大小
init_rng = RNG(1337)  # 初始化随机数生成器
# 两个模型类都会产生完全相同的结果。一个使用 nn.Module，另一个不使用
model = MLPRaw(vocab_size, context_length, embedding_size, hidden_size, init_rng)
# model = MLP(vocab_size, context_length, embedding_size, hidden_size, init_rng)

# 创建优化器
learning_rate = 7e-4  # 学习率
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# 训练循环
timer = StepTimer()  # 用于记录时间的计时器
batch_size = 128  # 批次大小
num_steps = 50000  # 训练步数
print(f'num_steps {num_steps}, num_epochs {num_steps * batch_size / len(train_tokens):.2f}')
train_data_iter = dataloader(train_tokens, context_length, batch_size)

for step in range(num_steps):
    # 余弦学习率调度，从最大学习率到 0
    lr = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # 每隔一段时间评估一次验证损失
    last_step = step == num_steps - 1
    if step % 200 == 0 or last_step:
        train_loss = eval_split(model, train_tokens, max_batches=20)
        val_loss = eval_split(model, val_tokens)
        print(f'step {step:6d} | train_loss {train_loss:.6f} | val_loss {val_loss:.6f} | lr {lr:e} | time/step {timer.get_dt()*1000:.4f}ms')

    # 训练步骤
    with timer:
        # 获取下一个批次的训练数据
        inputs, targets = next(train_data_iter)
        # 前向传播（计算损失）
        logits, loss = model(inputs, targets)
        # 反向传播（计算梯度）
        loss.backward()
        # 更新优化器（更新参数）
        optimizer.step()
        optimizer.zero_grad()

# 模型推断
# 硬编码一个提示文本，以便继续生成文本
sample_rng = RNG(42)  # 采样用的随机数生成器
prompt = "\nrichard"  # 提示文本
context = [char_to_token[c] for c in prompt]  # 将提示文本转换为令牌
assert len(context) >= context_length
context = context[-context_length:]  # 裁剪到上下文长度
print(prompt, end='', flush=True)
# 现在生成接下来的 200 个令牌
with torch.inference_mode():
    for _ in range(200):
        # 取最后的 context_length 个令牌，预测下一个令牌
        context_tensor = torch.tensor(context).unsqueeze(0)  # (1, T)
        logits, _ = model(context_tensor)  # (1, V)
        probs = softmax(logits[0])  # (V, )
        coinf = sample_rng.random()  # "硬币抛掷"，范围在 [0, 1) 之间的 float32
        next_token = sample_discrete(probs, coinf)  # 从离散分布中采样
        context = context[1:] + [next_token]  # 更新令牌上下文
        print(token_to_char[next_token], end='', flush=True)
print()  # 换行

# 最后报告测试损失
test_loss = eval_split(model, test_tokens)
print(f'test_loss {test_loss}')


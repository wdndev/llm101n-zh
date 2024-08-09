"""
最小 (byte-level) Byte Pair Encoding (BPE) 分词.

算法上遵循GPT分词器的实现:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

但是：
- 不处理正则表达式的分割模式。
- 不处理任何特殊标记。
"""

from .base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):
    """ 最基本 BPE 分词
    """

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        """
        训练分词器。

        Args:
            text (str): 训练文本。
            vocab_size (int): 词汇表大小。
            verbose (bool, optional): 是否打印详细信息。

        Raises:
            AssertionError: 如果vocab_size小于256，则抛出异常。
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256   # 需要合并的次数

        # 输入文本预处理
        text_bytes = text.encode("utf-8") # 原始字节
        ids = list(text_bytes) # 字节对应的整数列表，范围0..255

        # 迭代合并最常见的对来创建新token
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # 统计每一对连续整数出现的次数
            stats = get_stats(ids)
            # 找到出现次数最多的对
            pair = max(stats, key=stats.get)
            # 新建一个token： 分配下一个可用的 id
            idx = 256 + i
            # 将id中出现的所有pair替换为idx
            ids = merge(ids, pair, idx)
            # 保存合并记录
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # 打印信息
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # 保存变量
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def decode(self, ids):
        """
        解码整数列表为字符串。

        Args:
            ids (list of int): 整数列表。

        Returns:
            str: 解码后的字符串。
        """
        # 给定整数列表，返回Python字符串
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        """
        编码字符串为整数列表。

        Args:
            text (str): 输入字符串。

        Returns:
            list of int: 编码后的整数列表。
        """
        text_bytes = text.encode("utf-8") # 原始字节
        ids = list(text_bytes) # 字节对应的整数列表，范围0..255
        while len(ids) >= 2:
            # 找到合并索引最低的对
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            # 如果没有更多可以合并的对，pair将是任意的第一对
            # 检测终止情况
            if pair not in self.merges:
                break # 没有更多的合并对
            # 否则，合并最佳对（最低合并索引）
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

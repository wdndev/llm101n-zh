"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.
"""
import unicodedata

# -----------------------------------------------------------------------------
# 一些辅助函数，适用于BasicTokenizer和RegexTokenizer

def get_stats(ids, counts=None):
    """ 给定一个整数列表，返回连续对的计数字典。
    可选地更新现有的计数字典。
    Args:
        ids (list of int): 一组整数
        counts (dict, optional): 存储连续对计数的现有字典

    Returns:
        dict: 连续对及其计数的字典

    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = {} if counts is None else counts
    # 迭代连续元素
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """ 在整数列表(ids)中，用新整数标记(idx)替换所有连续出现的对(pair)。
    Args:
        ids (list of int): 整数列表
        pair (tuple of int): 要合并的连续对
        idx (int): 替换后的整数标记

    Returns:
        list of int: 替换后的新整数列表

    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # 如果不是最后一个位置且对匹配，则替换
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def replace_control_characters(s: str) -> str:
    """ 替换字符串中的控制字符以避免输出乱序（例如`\n`或其他更糟糕的情况）
    Args:
        s (str): 输入字符串

    Returns:
        str: 替换控制字符后的字符串
    
    Ref：
        https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        http://www.unicode.org/reports/tr44/#GC_Values_Table
    """
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    """ 打印一个记号，转义控制字符
    """
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

# -----------------------------------------------------------------------------
# Tokenizer 基类

class Tokenizer:
    """
    Tokenizer基类，提供基本的分词功能和保存/加载功能。
    """

    def __init__(self):
        # 默认情况下，词汇表大小为256（所有字节），没有合并规则，没有模式。
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        """ 据文本训练一个指定大小的词表
        Args:
            text (str): 训练文本
            vocab_size (int): 词汇表大小
            verbose (bool, optional): 是否打印调试信息
        """
        raise NotImplementedError

    def encode(self, text):
        """ 将字符串编码为整数列表
        """
        raise NotImplementedError

    def decode(self, ids):
        """ 将整数列表解码为字符串
        """
        raise NotImplementedError

    def _build_vocab(self):
        """
        构建词汇表。
        
        词汇表由合并规则确定性地构建。
        """
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Args:
            file_prefix (str): 文件前缀

        保存两个文件：file_prefix.vocab 和 file_prefix.model
        这受到sentencepiece模型保存方式的启发（但并不等同）：
        - model文件是关键的，用于load()。
        - vocab文件仅供人类检查。
        """
        # 写入 model 文件：供之后的load()使用
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # 写入版本、模式和合并规则
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # 写入特殊token，数量和token
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # 写入合并字典
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # 写入vocab文件: 方面查看
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # 注意：许多 token 是 utf-8 格式的，不能被 decode 为正常字符串
                # 使用 errors='replace' 来替换
                # 意味着不能再 load() 中使用。因为这种解码方式是有损耗的
                s = render_token(token)
                # 查找这个 token 的孩子 token
                if idx in inverted_merges:
                    # 如果这个 token 有孩子token，则合并显示
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # 否则直接打印（这应该是前256个token，即bytes）
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        """ 加载模型文件
        Args:
            model_file (str): 模型文件路径
        注意：仅针对模型文件。
        """
        assert model_file.endswith(".model")
        # 读取模型文件
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # 读取版本
            version = f.readline().strip()
            assert version == "minbpe v1"
            # 读取模式
            self.pattern = f.readline().strip()
            # 读取特殊标记
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # 读取合并
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()

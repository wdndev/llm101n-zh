import time
from math import log, cos, sin, pi

# -----------------------------------------------------------------------------
# 随机数生成

def box_muller_transform(u1, u2):
    """使用 Box-Muller 变换将均匀分布的随机数转换为标准正态分布的随机数。

    Args:
        u1 (float): 第一个均匀分布的随机数，取值范围在 [0, 1) 之间。
        u2 (float): 第二个均匀分布的随机数，取值范围在 [0, 1) 之间。

    Returns:
        tuple: 包含两个标准正态分布的随机变量 (z1, z2)。
    """
    z1 = (-2 * log(u1)) ** 0.5 * cos(2 * pi * u2)
    z2 = (-2 * log(u1)) ** 0.5 * sin(2 * pi * u2)
    return z1, z2

class RNG:
    """一个简单的伪随机数生成器，能够生成均匀分布和正态分布的随机数。

    Attributes:
        state (int): 用于确定随机数序列的种子。
    """

    def __init__(self, seed):
        """初始化 RNG 类。

        Args:
            seed (int): 用于确定随机数序列的种子。
        """
        self.state = seed

    def random_u32(self):
        """生成一个32位的无符号随机整数。

        Returns:
            int: 32位的无符号随机整数。
        """
        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    def random(self):
        """生成一个在 [0, 1) 范围内的随机浮点数。

        Returns:
            float: 在 [0, 1) 范围内的随机浮点数。
        """
        return (self.random_u32() >> 8) / 16777216.0

    def rand(self, n, a=0, b=1):
        """生成 n 个在 [a, b) 范围内的随机浮点数。

        Args:
            n (int): 要生成的随机数的个数。
            a (float, optional): 随机数的下界，默认为 0。
            b (float, optional): 随机数的上界，默认为 1。

        Returns:
            list: 包含 n 个在 [a, b) 范围内的随机浮点数的列表。
        """
        return [self.random() * (b - a) + a for _ in range(n)]

    def randn(self, n, mu=0, sigma=1):
        """生成 n 个服从正态分布的随机浮点数。

        Args:
            n (int): 要生成的随机数的个数。
            mu (float, optional): 正态分布的均值，默认为 0。
            sigma (float, optional): 正态分布的标准差，默认为 1。

        Returns:
            list: 包含 n 个服从均值为 mu，标准差为 sigma 的正态分布的随机浮点数的列表。
        """
        out = []
        for _ in range((n + 1) // 2):
            u1, u2 = self.random(), self.random()
            z1, z2 = box_muller_transform(u1, u2)
            out.extend([z1 * sigma + mu, z2 * sigma + mu])
        out = out[:n]  # 如果 n 是奇数，裁剪列表以符合要求
        return out

# -----------------------------------------------------------------------------
# StepTimer for timing code

class StepTimer:
    """用于计量代码执行时间的计时器，支持指数移动平均时间估计。

    Attributes:
        ema_alpha (float): EMA的衰减因子，控制EMA的平滑程度。
        ema_time (float): 当前的EMA时间估计。
        corrected_ema_time (float): 经过偏差修正后的EMA时间估计。
        start_time (float): 计时器开始的时间戳。
        step (int): 计时器的当前步骤数。
    """

    def __init__(self, ema_alpha=0.9):
        """初始化 StepTimer 类。

        Args:
            ema_alpha (float): EMA的衰减因子，默认为0.9。
        """
        self.ema_alpha = ema_alpha
        self.ema_time = 0
        self.corrected_ema_time = 0.0
        self.start_time = None
        self.step = 0

    def __enter__(self):
        """上下文管理器的进入方法，记录开始时间。

        Returns:
            StepTimer: 返回计时器本身。
        """
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器的退出方法，记录结束时间并更新EMA时间估计。

        Args:
            exc_type: 异常类型（如果有）。
            exc_val: 异常值（如果有）。
            exc_tb: 异常回溯信息（如果有）。
        """
        end_time = time.time()
        iteration_time = end_time - self.start_time
        self.ema_time = self.ema_alpha * self.ema_time + (1 - self.ema_alpha) * iteration_time
        self.step += 1
        # 偏差修正后的EMA时间估计
        self.corrected_ema_time = self.ema_time / (1 - self.ema_alpha ** self.step)

    def get_dt(self):
        """获取经过偏差修正后的EMA时间估计。

        Returns:
            float: 经过偏差修正后的EMA时间估计。
        """
        return self.corrected_ema_time

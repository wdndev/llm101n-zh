
class RNG:
    """ 随机数生成
        模拟Python中随机数生成，完全确定，能够认为控制，也可以在其他语言中使用
    """
    def __init__(self, seed):
        """初始化RNG实例

        Args:
            seed (int): 生成随机数序列的初始种子
        """
        self.state = seed

    def random_u32(self):
        """ 生成一个32位无符号整数的随机数

        使用 Xorshift 算法:
        参考: https://en.wikipedia.org/wiki/Xorshift#xorshift*

        在Python中，
        - `0xFFFFFFFFFFFFFFFF` 相当于C中的无符号64位整数转换
        - `0xFFFFFFFF` 相当于C中的无符号32位整数转换

        Returns:
            int: 32位无符号整数
        """
        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    def random(self):
        """ 生成一个范围在[0, 1)之间的32位浮点随机数

        Returns:
            float: 范围在[0, 1)之间的随机数
        """
        return (self.random_u32() >> 8) / 16777216.0    # 16777216.0 等于 2^24

    def uniform(self, a=0.0, b=1.0):
        """ 生成一个范围在[a, b)之间的32位浮点随机数

        Returns:
            float: 范围在[a, b)之间的随机数
        """
        return a + (b - a) * self.random()


def gen_data(random: RNG, n=100):
    """ 生成包含 3 个类的 n 个二维数据点的随机数据集

    Returns:
        list: 训练集、验证集和测试集
    """
    pts = []
    for _ in range(n):
        x = random.uniform(-2.0, 2.0)
        print(x)
        y = random.uniform(-2.0, 2.0)
        # label = 0 if x**2 + y**2 < 1 else 1 if x**2 + y**2 < 2 else 2
        # 非常简单的数据集
        label = 0 if x < 0 else 1 if y < 0 else 2
        pts.append(([x, y], label))

    # 训练集、验证集、测试集比例：80%、10%、10%
    tr = pts[:int(0.8 * n)]
    val = pts[int(0.8 * n):int(0.9 * n)]
    te = pts[int(0.9 * n):]
    return tr, val, te

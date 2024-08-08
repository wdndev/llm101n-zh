"""
定义一个简单的 autograd 引擎，并使用它来对平面中的点进行分类
使用简单的多层感知器 （MLP） 到 3 类（红色、绿色、蓝色）。
"""
import math
from matplotlib import pyplot as plt
from utils import RNG, gen_data

random = RNG(42)


# -----------------------------------------------------------------------------
# Value 类具体实现

class Value:
    """ 存储单个标量值及其梯度.
    """

    def __init__(self, data, _children=(), _op=''):
        """ 初始化Value实例

        Args:
            data (int): 存储的数值
            _children (tuple):
            _op (str): 进行的运算操作符
        """

        self.data = data
        self.grad = 0

        # 用于 Autograd 图形构造的内部变量
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # 用于生成此节点的运算，用于 GraphViz /调试/等

    def __add__(self, other):
        """ 对 Value 对象进行加法运算

        Returns:
            Value: 相加之后构造的 Value 对象
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # 记录加法运算求导后的梯度
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        """ 对 Value 对象进行乘法运算

        Returns:
            Value: 相乘之后构造的 Value 对象
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # 记录乘法运算求导后的梯度
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        """ 对 Value 对象进行 other 次幂运算

        Returns:
            Value: 幂运算之后封装结果的 Value 对象
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # 记录幂运算求导后的梯度
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        """ 对 Value 对象进行 max(0, x) 运算

        Returns:
            Value: relu 之后构造的 Value 对象
        """
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            # 记录对 relu 求导后的梯度
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        """ 对 Value 对象进行 tanh 运算

        Returns:
            Value: tanh 之后构造的 Value 对象
        """
        out = Value(math.tanh(self.data), (self,), 'tanh')

        def _backward():
            # 记录对 tanh 求导后的梯度
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        """ 对 Value 对象进行指数运算

        Returns:
            Value: 指数运算之后构造的 Value 对象
        """
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            # 记录对指数运算求导后的梯度
            self.grad += math.exp(self.data) * out.grad

        out._backward = _backward

        return out

    def log(self):
        """ 对 Value 对象进行自然对数运算

        Returns:
            Value: log 运算之后构造的 Value 对象
        """
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            # 记录对对数运算求导后的梯度
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward

        return out

    def backward(self):

        # 计算图中所有元素的拓扑排序
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # 一次运行一个变量，并应用链式法则来计算其梯度
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


# -----------------------------------------------------------------------------
# 多层感知器 （MLP） 网络

class Module:
    """ 神经网络模块定义
    """
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    """ 神经元定义
    """
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1) * nin ** -0.5) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'TanH' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    """ 神经网络层定义
    """
    def __init__(self, nin, nout, **kwargs):
        """

        """
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """ MLP 模型类
    """
    def __init__(self, nin, nouts):
        """ MLP 模型初始化

        Args:
            nin (int): 输入维度
            nouts (list): 模型不同隐藏层的维度
        """
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


# -----------------------------------------------------------------------------
# loss 函数：负对数似然 （NLL） 损失
# NLL 损失 = 当目标是独热向量时的交叉熵损失

def cross_entropy(logits, target):
    # 减去最大值以实现数值稳定性（避免溢出）
    max_val = max(val.data for val in logits)
    logits = [val - max_val for val in logits]
    # 1) 逐元素计算 e^x
    ex = [x.exp() for x in logits]
    # 2) 计算上述各项的总和
    denom = sum(ex)
    # 3) 按总和进行归一化以获得概率
    probs = [x / denom for x in ex]
    # 4) 记录目标位置的概率
    logp = (probs[target]).log()
    # 5) 负对数似然损失（取相反数，因此我们得到损失 - 越低越好）
    nll = -logp
    return nll


# -----------------------------------------------------------------------------
# 计算模型在给定数据集上的损失

def eval_split(model, split):
    # evaluate the loss of a split
    loss = Value(0)
    for x, y in split:
        logits = model([Value(x[0]), Value(x[1])])
        loss += cross_entropy(logits, y)
    loss = loss * (1.0 / len(split))  # normalize the loss
    return loss.data


# -----------------------------------------------------------------------------
# 开始训练

# 生成一个随机数据集，其中包含 3 个类的 100 个二维数据点
train_split, val_split, test_split = gen_data(random, n=100)

# 初始化模型：2D 输入、16 个神经元、3 个输出 （logits）
model = MLP(2, [16, 3])

# 使用 Adam 优化器
learning_rate = 1e-1
beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-4
for p in model.parameters():
    p.m = 0.0
    p.v = 0.0

train_losses, val_losses = [], []

# 开始训练
for step in range(100):

    # forward the network (get logits of all training datapoints)
    loss = Value(0)
    for x, y in train_split:
        logits = model([Value(x[0]), Value(x[1])])
        loss += cross_entropy(logits, y)
    loss = loss * (1.0 / len(train_split))  # normalize the loss

    # backward pass (deposit the gradients)
    loss.backward()

    # update with AdamW
    for p in model.parameters():
        p.m = beta1 * p.m + (1 - beta1) * p.grad
        p.v = beta2 * p.v + (1 - beta2) * p.grad ** 2
        m_hat = p.m / (1 - beta1 ** (step + 1))  # bias correction
        v_hat = p.v / (1 - beta2 ** (step + 1))
        p.data -= learning_rate * (m_hat / (v_hat ** 0.5 + 1e-8) + weight_decay * p.data)
    model.zero_grad()  # never forget to clear those gradients! happens to everyone

    val_loss = eval_split(model, val_split)
    train_losses.append(loss.data)
    val_losses.append(val_loss)
    print(f"step {step}, train loss {loss.data}")
    print(f"step {step}, val loss {val_loss:.6f}")

# 绘制训练和验证损失
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss', linestyle='--')
plt.xlabel('Iteration (Step)')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Iterations')
plt.legend()
plt.grid(True)
plt.show()

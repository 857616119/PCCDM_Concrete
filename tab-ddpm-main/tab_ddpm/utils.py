import torch
import numpy as np
import torch.nn.functional as F
from torch.profiler import record_function
from inspect import isfunction

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    计算两个高斯分布之间的KL散度。
    形状会自动广播，因此可以将批次与标量等进行比较。

    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "至少有一个参数必须是张量 (at least one argument must be a Tensor)"

    # 强制方差为张量。广播有助于将标量转换为张量，但不适用于torch.exp()
    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def approx_standard_normal_cdf(x):
    """
    标准正态分布累积分布函数的快速近似。
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    计算离散化到给定图像的高斯分布的对数似然。
    假设x原本是uint8值，缩放到范围[-1, 1]。

    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: 目标图像。假设原本是uint8值，缩放到范围[-1, 1]。
    :param means: 高斯均值张量。
    :param log_scales: 高斯对数标准差张量。
    :return: 与x形状相同的对数概率张量（以纳特为单位）。
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def sum_except_batch(x, num_dims=1):
    '''
    对除批次维度外的所有维度求和。
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, 批次维度数（默认=1）
    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def mean_flat(tensor):
    """
    对所有非批次维度取平均值。
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def ohe_to_categories(ohe, K):
    """
    将独热编码转换回分类索引。
    Convert one-hot encoding back to categorical indices.
    
    Args:
        ohe: 独热编码张量
        K: 每个分类特征的类别数数组
    Returns:
        分类索引张量
    """
    K = torch.from_numpy(K)
    indices = torch.cat([torch.zeros((1,)), K.cumsum(dim=0)], dim=0).int().tolist()
    res = []
    for i in range(len(indices) - 1):
        res.append(ohe[:, indices[i]:indices[i+1]].argmax(dim=1))
    return torch.stack(res, dim=1)

def log_1_min_a(a):
    """数值稳定地计算log(1 - exp(a))"""
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    """数值稳定地计算log(exp(a) + exp(b))"""
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def exists(x):
    """检查变量是否存在"""
    return x is not None

def extract(a, t, x_shape):
    """
    从张量a中提取时间步t对应的值并扩展到x_shape形状。
    用于扩散模型中的时间步参数提取。
    
    Args:
        a: 形状为[num_timesteps, ...]的张量
        t: 时间步索引张量
        x_shape: 目标形状
    Returns:
        提取并扩展后的张量
    """
    b, *_ = t.shape
    t = t.to(a.device)
    out = a.gather(-1, t)
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)

def default(val, d):
    """如果val存在则返回val，否则返回默认值d"""
    if exists(val):
        return val
    return d() if isfunction(d) else d

def log_categorical(log_x_start, log_prob):
    """计算分类分布的对数似然"""
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    """
    将分类索引转换为对数独热编码。
    用于多项扩散中处理分类变量。
    
    Args:
        x: 分类索引张量，形状为[batch_size, num_categorical_features]
        num_classes: 每个分类特征的类别数数组
    Returns:
        对数独热编码张量
    """
    onehots = []
    for i in range(len(num_classes)):
        onehots.append(F.one_hot(x[:, i], num_classes[i]))
 
    x_onehot = torch.cat(onehots, dim=1)
    log_onehot = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_onehot

def log_sum_exp_by_classes(x, slices):
    """
    对每个类别切片分别计算logsumexp。
    用于多项扩散中的归一化。
    
    Args:
        x: 输入张量
        slices: 类别切片列表
    Returns:
        每个切片内logsumexp的结果
    """
    device = x.device
    res = torch.zeros_like(x)
    for ixs in slices:
        res[:, ixs] = torch.logsumexp(x[:, ixs], dim=1, keepdim=True)

    assert x.size() == res.size()

    return res

@torch.jit.script
def log_sub_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """数值稳定地计算log(exp(a) - exp(b))"""
    m = torch.maximum(a, b)
    return torch.log(torch.exp(a - m) - torch.exp(b - m)) + m

@torch.jit.script
def sliced_logsumexp(x, slices):
    """
    使用累积对数求和技巧高效计算每个切片的logsumexp。
    优化版本，使用JIT编译提高性能。
    
    Args:
        x: 输入张量
        slices: 切片边界索引
    Returns:
        每个切片内logsumexp的结果
    """
    lse = torch.logcumsumexp(
        torch.nn.functional.pad(x, [1, 0, 0, 0], value=-float('inf')),
        dim=-1)

    slice_starts = slices[:-1]
    slice_ends = slices[1:]

    slice_lse = log_sub_exp(lse[:, slice_ends], lse[:, slice_starts])
    slice_lse_repeated = torch.repeat_interleave(
        slice_lse,
        slice_ends - slice_starts, 
        dim=-1
    )
    return slice_lse_repeated

def log_onehot_to_index(log_x):
    """将对数独热编码转换回分类索引"""
    return log_x.argmax(1)

class FoundNANsError(BaseException):
    """在采样过程中发现NaN值的异常"""
    def __init__(self, message='在采样过程中发现NaN值 (Found NANs during sampling.)'):
        super(FoundNANsError, self).__init__(message)
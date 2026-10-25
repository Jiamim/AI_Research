"""第 08 章：逐 token 追加 KIVI 风格缓存，核验尾部与分段 softmax。"""
import math
import torch
from experiments.demo_runtime import get_runtime, same, exact

# 第 0 步：auto优先CUDA；runner可用--device指定。所有计算张量在此设备创建。
rt = get_runtime()
device, dtype = rt.device, rt.dtype
from experiments.demo_torch_kernels import softmax
from experiments.demo_torch_kernels import TinyKIVI

# 第 1 步：预先生成 K/V，以追加顺序模拟 decode；只有一个头，无 RoPE。
rng = rt.rng(800)
keys = rng.normal(size=(9, 4))
values = rng.normal(size=(9, 4))
queries = rng.normal(size=(9, 4))
keys[:, 0] *= 8
cache = TinyKIVI(dim=4, group=2, residual=4, bits=2, device=device)

# 第 2 步：每次只追加当前 token；K 在尾部满4个时分时间组量化，
# V 保留最近4个，旧行按特征组量化。两个尾部长度不必相同。
for t in range(len(keys)):
    cache.append(keys[t], values[t])
    Kq, Vq = cache.reconstruct()
    assert Kq.shape == Vq.shape == (t+1, 4)
    z = queries[t]@Kq.T/math.sqrt(4)
    direct = softmax(z)@Vq

    # 第 3 步：按任意 token 边界分段，保留每段的最大值、分母和未归一化分子。
    # 不能直接平均两段各自的 softmax 输出，因为每段概率质量不同。
    statistics = []
    for start in range(0, t+1, 3):
        logits, V = z[start:start+3], Vq[start:start+3]
        maximum = logits.max()
        unnormalized = torch.exp(logits-maximum)
        statistics.append((maximum, unnormalized.sum(), unnormalized@V))
    global_max = max(m for m, _, _ in statistics)
    numerator = sum(torch.exp(m-global_max)*u for m, _, u in statistics)
    denominator = sum(torch.exp(m-global_max)*mass for m, mass, _ in statistics)
    merged = numerator/denominator
    assert same(merged, direct)

    # 第 4 步：比较原始缓存，区分“Attention计算正确”和“缓存量化无误差”。
    reference = softmax(queries[t]@keys[:t+1].T/2)@values[:t+1]
    print('长度 / K尾部 / V尾部 / 输出MSE:', t+1,
          len(cache.k_recent), len(cache.v_recent), torch.mean((merged-reference)**2))
    if t < 3:
        assert same(merged, reference)  # 尚未触发任何量化。

# 第 5 步：模拟追加失败，确保已存在缓存没有被半更新。
before = tuple(array.clone() for array in cache.reconstruct())
try:
    cache.append(torch.full((4,), math.nan, device=device, dtype=dtype), torch.ones(4, device=device, dtype=dtype))
except ValueError:
    pass
else:
    raise AssertionError('非有限输入必须被拒绝')
for actual, expected in zip(cache.reconstruct(), before):
    assert exact(actual, expected)
print('无效追加后 K/V 状态保持不变')
# TinyKIVI 的码存在 uint8 容器中，未物理打包成2 bit，不能用数组字节数推压缩率。

# 收尾：等待本demo设备上的操作完成，方便Colab定位CUDA异步报错。
rt.synchronize()

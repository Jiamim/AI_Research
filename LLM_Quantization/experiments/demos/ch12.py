"""第 12 章：四状态 trellis 编码，动态规划和穷举交叉核验。"""
import itertools
import torch
from experiments.demo_runtime import get_runtime, same, exact

# 第 0 步：auto优先CUDA；runner可用--device指定。所有计算张量在此设备创建。
rt = get_runtime()
device, dtype = rt.device, rt.dtype

# 第 1 步：每个状态对应一个标量码字；每步输入一个bit，状态保留最近两bit。
# 固定初始状态0，所以首步只有两个状态可达，不能逐元素任意选码字。
values = rt.tensor([-.8, -.1, .9, .2, -.4])
codebook = rt.tensor([-1., -.25, .25, 1.])
paths = {0: (0., [])}

# 第 2 步：每层只保留到达同一状态的最低代价路径。
# 后续可达集合只依赖当前状态，因此此前更高代价路径可以安全丢弃。
for t, value in enumerate(values):
    following = {}
    for state, (cost, bits) in paths.items():
        for bit in (0, 1):
            next_state = ((state << 1) | bit) & 3
            next_cost = cost+(value-codebook[next_state])**2
            if next_state not in following or next_cost < following[next_state][0]:
                following[next_state] = (float(next_cost), bits+[bit])
    paths = following
    print('位置 / 可达状态最低SSE:', t, {s: round(c, 5) for s, (c, _) in paths.items()})
best_cost, best_bits = min(paths.values(), key=lambda item: item[0])

# 第 3 步：解码只需要初始状态、bit流与共享码本，不需要原始 values。
def decode(bits):
    state = 0
    output = []
    for bit in bits:
        state = ((state << 1) | bit) & 3
        output.append(codebook[state])
    return torch.stack(output)
decoded = decode(best_bits)
assert same(torch.sum((decoded-values)**2), best_cost)

# 第 4 步：本例仅2^5条路径，穷举提供独立全局最优核验。
exhaustive = min(torch.sum((decode(bits)-values)**2)
                 for bits in itertools.product((0, 1), repeat=len(values)))
assert same(best_cost, exhaustive)

# 第 5 步：bit流真正打包；注明尾部补齐以及固定起点假设。
bits_tensor = rt.tensor(best_bits, dtype=torch.int64)
padded = torch.nn.functional.pad(bits_tensor, (0, (-len(best_bits)) % 8))
shifts = torch.arange(8, device=device)
packed = torch.sum(padded.reshape(-1, 8) << shifts, dim=1).to(torch.uint8)
restored_bits = ((packed.long()[:, None] >> shifts) & 1).flatten()[:len(values)]
assert exact(restored_bits, best_bits)
print('最佳bit流 / 重建 / SSE:', best_bits, decoded, best_cost)
print('逻辑bit数 / 实际payload字节:', len(best_bits), packed.nbytes)
# 还应记录长度、码本与块边界；这里只展示 QTIP 的路径思想，不含其实际码构造或内核。

# 收尾：等待本demo设备上的操作完成，方便Colab定位CUDA异步报错。
rt.synchronize()

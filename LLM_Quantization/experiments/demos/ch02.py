"""第 02 章：逐列补偿，观察每一步并用穷举核对最终离散结果。"""
import itertools
import math
import torch
from experiments.demo_runtime import get_runtime, same

# 第 0 步：auto优先CUDA；runner可用--device指定。所有计算张量在此设备创建。
rt = get_runtime()
device, dtype = rt.device, rt.dtype

# 第 1 步：构造相关输入，让权重误差之间存在可以抵消的交叉项。
X = rt.tensor([[1., .5], [0., math.sqrt(3)/2]])
w = rt.tensor([.6, .6])
grid = rt.tensor([0., 1.])
H = 2*X.T@X
M = torch.linalg.inv(H)  # 仅二维教学；大模型使用稳定分解和批量更新。
current = w.clone()     # 工作副本会补偿，原 w 留作误差参照。
quantized = torch.empty_like(w)

# 第 2 步：先算 RTN 基线。固定网格下它优化权重距离，不利用输入相关性。
rtn = grid[torch.argmin(abs(w[:, None]-grid), dim=1)]
loss = lambda candidate: float(torch.sum((X@(candidate-w))**2))
print('RTN:', rtn, '输出 SSE:', loss(rtn))

# 第 3 步：每次固定一个码字，只允许后面的自由权重继续补偿。
for i in range(w.numel()):
    chosen = grid[torch.argmin(abs(grid-current[i]))]
    epsilon = chosen-current[i]
    # M 的第 0 行/列始终对应当前尚未固定的第一个变量。
    correction = epsilon*M[:, 0]/M[0, 0]
    current[i:] += correction
    quantized[i] = chosen
    assert same(current[i], chosen)
    print(f'第{i+1}步: 码字={chosen}, 连续补偿={correction}, 工作权重={current}')
    # 消去已经固定的变量，得到剩余主子矩阵的逆；不能重复用最初的逆。
    if len(M) > 1:
        M = M[1:, 1:]-torch.outer(M[1:, 0], M[0, 1:])/M[0, 0]
        assert same(M, torch.linalg.inv(H[i+1:, i+1:]))

# 第 4 步：用输出误差与二次型两条独立路径计算同一个目标。
delta = quantized-w
assert same(loss(quantized), .5*delta@H@delta)
assert same(quantized, [1., 0.])
assert same(loss(quantized), .28)

# 第 5 步：这里只有四个离散组合，可以穷举确定本例的全局最优值。
# 大模型无法这样穷举；本例最优不构成 GPTQ 一般全局最优的证明。
combinations = rt.tensor(list(itertools.product((0, 1), repeat=w.numel())), dtype=torch.int64)
candidates = grid[combinations]
costs = ((candidates-w)@X.T).square().sum(dim=1)
print('全部候选及 SSE:', list(zip(candidates.tolist(), costs.tolist())))
assert same(loss(quantized), costs.min())
print('补偿结果:', quantized, '输出 SSE:', loss(quantized))

# 收尾：等待本demo设备上的操作完成，方便Colab定位CUDA异步报错。
rt.synchronize()

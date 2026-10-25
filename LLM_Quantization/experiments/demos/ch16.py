"""第 16 章：在质量和内存门槛下，计算不同使用量的成功任务成本。"""
import torch
from experiments.demo_runtime import get_runtime

# 第 0 步：把候选数据放到CPU/CUDA；成本公式、筛选和argmin在同一设备完成。
rt = get_runtime()
device, dtype = rt.device, rt.dtype

# 第 1 步：声明假想候选及单位。成本用同型号GPU秒，避免不同设备小时直接相加。
# 这些数据只解释决策逻辑，不是任何模型或硬件的实测。
candidates = [
    dict(name='FP16', offline=0., attempt=.20, success=.92, memory=14., p95=.8),
    dict(name='快速W4', offline=600., attempt=.16, success=.90, memory=5., p95=.7),
    dict(name='充分校准W4', offline=36000., attempt=.12, success=.915, memory=5.2, p95=.6),
    dict(name='过度压缩', offline=300., attempt=.10, success=.70, memory=3., p95=.5),
]
minimum_quality, max_memory, max_latency = .89, 16., 1.

# 第 2 步：先按硬门槛排除不可用方案，再比较成本；不允许用低质量换便宜。
data = rt.tensor([[c[k] for k in ('offline', 'attempt', 'success', 'memory', 'p95')] for c in candidates])
feasible_mask = (data[:, 2] >= minimum_quality) & (data[:, 3] <= max_memory) & (data[:, 4] <= max_latency)
feasible = data[feasible_mask]
names = [c['name'] for c, keep in zip(candidates, feasible_mask.tolist()) if keep]
assert len(feasible) == 3

# 第 3 步：固定尝试次数N，离线成本也算入总支出。
# 总成本/期望成功数=(offline+N*attempt)/(N*success)，不是单token价格。
for attempts in (1000, 100000, 1000000):
    costs = (feasible[:, 0]+attempts*feasible[:, 1])/(attempts*feasible[:, 2])
    best = int(costs.argmin())
    print('尝试次数:', attempts,
          '各方案GPU秒/期望成功数:', {name: round(float(v), 4) for name, v in zip(names, costs)},
          '选择:', names[best])

# 第 4 步：用方程求精细校准与基线的盈亏点，注意成功率不同也影响门槛。
base, optimized = feasible[0], feasible[2]
saving_per_success = base[1]/base[2]-optimized[1]/optimized[2]
assert saving_per_success > 0
break_even = (optimized[0]/optimized[2]-base[0]/base[2])/saving_per_success
print('相同尝试数量下成功成本的盈亏点:', round(float(break_even)), '次尝试')

# 第 5 步：这是按给定成功率的期望算账，不把所有任务都假设为可独立重试成功。
# 修改 success、请求长度或内存门槛，观察最优方案是否变化；最终需用真实成本替换。
rt.synchronize()

"""第 04 章：优化舍入变量，并跟踪 HQQ 的一个可解子问题。"""
import math
import torch
from experiments.demo_runtime import get_runtime

# 第 0 步：auto优先CUDA；runner可用--device指定。所有计算张量在此设备创建。
rt = get_runtime()
device, dtype = rt.device, rt.dtype

# 第 1 步：固定原始权重；只让舍入偏移变化，目标是联合输出误差。
w = rt.tensor([.49, .46])
X = rt.tensor([[1., 1.], [.9, 1.1]])
target = X@w
offset = torch.zeros_like(w)
best_loss = math.inf
best_codes = None

# 第 2 步：前向真实 round，反向手写 identity STE。
# loss=mean(error²)，因此梯度是 2 X^T error/n；这里省略裁剪外梯度。
for iteration in range(30):
    normalized = w+offset
    codes = torch.clamp(torch.round(normalized), 0, 1)
    error = X@codes-target
    loss = torch.mean(error**2)
    if loss < best_loss:
        best_loss, best_codes = float(loss), codes.clone()
    gradient = 2*X.T@error/len(X)
    mask = (normalized > 0) & (normalized < 1)
    offset -= .02*torch.sign(gradient)*mask
    if iteration in (0, 1, 5, 29):
        print('舍入迭代 / 当前码 / 当前损失:', iteration, codes, loss)
assert best_loss < torch.mean((X@torch.round(w)-target)**2)
print('保留的最佳码与损失:', best_codes, best_loss)

# 第 3 步：另用一组权重演示 HQQ p=1、固定尺度的分裂更新。
# 这里 zero 是浮点 offset，不能当作整数仿射 zero point 导出。
weights = rt.tensor([-.3, -.1, .06, .5])
scale, zero, beta = .1, 5., 10.
for iteration in range(5):
    # 作者路径每轮从原 weights 重新选码，随后对残差做软阈值。
    q = torch.clamp(torch.round(weights/scale+zero), 0, 15)
    residual = weights-scale*(q-zero)
    auxiliary = torch.sign(residual)*torch.clamp_min(abs(residual)-1/beta, 0)
    # 固定 q 和 auxiliary 时，下面的均值是二次子问题的解析驻点。
    zero = torch.mean(q-(weights-auxiliary)/scale)
    stationary_residual = weights-scale*(q-zero)-auxiliary
    assert abs(stationary_residual.sum()) < 1e-12
    print('HQQ迭代 / offset / 权重MSE:', iteration, zero,
          torch.mean((weights-scale*(q-zero))**2))
    beta *= 1.2

# 第 4 步：本例只证明固定变量子问题的条件，不断言整个启发式迭代单调最优。
# 修改 X 能改变最佳舍入；HQQ 这条路径不使用 X，所以不能感知该任务变化。

# 收尾：等待本demo设备上的操作完成，方便Colab定位CUDA异步报错。
rt.synchronize()

"""第 07 章：手写 QAT 的 STE 更新，与冻结量化基座的 LoRA 训练对照。"""
import torch
from experiments.demo_runtime import get_runtime, same, exact

# 第 0 步：auto优先CUDA；runner可用--device指定。所有计算张量在此设备创建。
rt = get_runtime()
device, dtype = rt.device, rt.dtype
from experiments.demo_torch_kernels import group_qdq

# 第 1 步：构造一个线性教师，校准/测试输入分离；不用下载真实模型。
rng = rt.rng(700)
X = rng.normal(size=(96, 4))
test = rng.normal(size=(64, 4))
W = rt.tensor([[.33, -.26, .12, .43], [-.14, .38, -.32, .09]])
# 相关输入使权重误差能互相补偿；若输入各向同性，初始最近舍入可能已经最优。
X[:, 1] = X[:, 0]+.1*X[:, 1]
test[:, 1] = test[:, 0]+.1*test[:, 1]
teacher = X@W.T

# 第 2 步：固定步长和范围，明确 STE 所对应的前向量化器。
step = .2
quantize = lambda value: step*torch.clamp(torch.round(value/step), -3, 3)
master = W.clone()  # 主权重保留小更新，前向另生成 qdq 权重。
best_q = quantize(master)
best_loss = float(torch.mean((X@best_q.T-teacher)**2))
initial_loss = best_loss
crossed_grid = False
for iteration in range(120):
    qdq = quantize(master)
    error = X@qdq.T-teacher
    # loss 是所有 token×输出元素的平均，梯度必须除以 error.numel()。
    gradient = 2*error.T@X/error.numel()
    ste_mask = abs(master/step) < 3
    master -= .04*gradient*ste_mask
    candidate = quantize(master)
    crossed_grid = crossed_grid or not exact(candidate, quantize(W))
    loss = float(torch.mean((X@candidate.T-teacher)**2))
    if loss < best_loss:
        best_loss, best_q = loss, candidate.clone()
assert best_loss <= initial_loss
assert crossed_grid  # 主权重积累的小更新确实改变过前向码字。
print('QAT初始 / 最佳校准MSE / 测试MSE:', initial_loss, best_loss,
      torch.mean((test@best_q.T-test@W.T)**2))

# 第 3 步：另建冻结 INT4 QDQ 基座，只优化 B@A 的低秩增量。
# 这是 QLoRA 的冻结基座结构演示；此处格式为 INT4，不是完整 NF4/分页实现。
base = group_qdq(W, bits=4, group_size=4)
original_base = base.clone()
A = rng.normal(size=(1, 4))*.1
B = torch.zeros((2, 1), device=device, dtype=dtype)
before = torch.mean((X@base.T-teacher)**2)
for _ in range(300):
    error = X@(base+B@A).T-teacher
    grad_weight = 2*error.T@X/error.numel()
    # 同一轮的两个梯度都用更新前的 A/B，避免边更新边改变另一梯度。
    grad_B = grad_weight@A.T
    grad_A = B.T@grad_weight
    B -= .2*grad_B
    A -= .2*grad_A
assert exact(base, original_base)
after = torch.mean((X@(base+B@A).T-teacher)**2)
assert after < before

# 第 4 步：浮点合并与双支路代数等价；再量化合并权重又会引入误差。
separate = test@base.T+(test@A.T)@B.T
merged = base+B@A
assert same(separate, test@merged.T)
requantized = group_qdq(merged, bits=4, group_size=4)
print('adapter前后校准MSE:', before, after)
print('独立adapter / 重新量化合并后测试MSE:',
      torch.mean((separate-test@W.T)**2), torch.mean((test@requantized.T-test@W.T)**2))
# 不对原始硬 round 做有限差分来验证 STE；STE 是这里明确选择的替代导数。

# 收尾：等待本demo设备上的操作完成，方便Colab定位CUDA异步报错。
rt.synchronize()

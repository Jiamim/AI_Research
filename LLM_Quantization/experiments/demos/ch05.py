"""第 05 章：配对旋转、变换 Hessian、独立校准和非线性反例。"""
import torch
from experiments.demo_runtime import get_runtime, same

# 第 0 步：auto优先CUDA；runner可用--device指定。所有计算张量在此设备创建。
rt = get_runtime()
device, dtype = rt.device, rt.dtype
from experiments.demo_torch_kernels import hadamard, group_qdq

# 第 1 步：构造离群输入，用两个不同随机流分出校准/测试数据。
rng = rt.rng(500)
W = rng.normal(size=(4, 8))
calibration = rng.normal(size=(48, 8))
test = rng.normal(size=(96, 8))
calibration[:, 0] *= 15
test[:, 0] *= 15
H = 2*calibration.T@calibration

# 第 2 步：原坐标与四个随机符号 Hadamard 都作为候选。
# R=diag(signs)@H8，两个因子都正交，因此 R 仍保持长度和内积。
rotations = [torch.eye(8, device=device, dtype=dtype)]
for _ in range(4):
    signs = rng.choice([-1., 1.], size=8)
    rotations.append(torch.diag(signs)@hadamard(8, device=device))

# 第 3 步：每个候选同时旋转输入与权重，先验证未量化时函数完全一致。
records = []
for index, R in enumerate(rotations):
    Xr, Wr = calibration@R, W@R
    assert same(R.T@R, torch.eye(8, device=device, dtype=dtype))
    assert same(Xr@Wr.T, calibration@W.T)
    H_rotated = R.T@H@R
    assert same(H_rotated, 2*Xr.T@Xr)
    # 量化器看到的是旋转后的权重；误差也必须放在同一个坐标系下计算。
    Wq = group_qdq(Wr, bits=3, group_size=4)
    E = Wq-Wr
    output_sse = torch.sum((Xr@E.T)**2)
    assert same(output_sse, .5*torch.trace(E@H_rotated@E.T))
    records.append((float(output_sse), Wq, R))
    print('候选 / 校准SSE / 激活峰值:', index, output_sse, torch.amax(abs(Xr)))

# 第 4 步：校准选择后再报告测试误差；包含恒等矩阵使校准目标不劣于基线。
best = min(range(len(records)), key=lambda i: records[i][0])
assert records[best][0] <= records[0][0]
for index in (0, best):
    _, Wq, R = records[index]
    print('候选 / 独立测试MSE:', index, torch.mean(((test@R)@Wq.T-test@W.T)**2))

# 第 5 步：检查旋转不能任意穿过非线性，解释整模型融合为什么要看计算图。
x = rt.tensor([[1., -2.]])
R = hadamard(2, device=device)
assert not same(torch.clamp_min(x@R, 0), torch.clamp_min(x, 0)@R)
print('ReLU前后随意移动旋转：不等价')
# 这里只量化权重，未复现 QuaRot 的全部激活/KV 位置或 SpinQuant 训练。

# 收尾：等待本demo设备上的操作完成，方便Colab定位CUDA异步报错。
rt.synchronize()

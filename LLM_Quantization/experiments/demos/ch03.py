"""第 03 章：在校准集搜索缩放，在独立输入上比较 AWQ/SmoothQuant 思路。"""
import torch
from experiments.demo_runtime import get_runtime, same

# 第 0 步：auto优先CUDA；runner可用--device指定。所有计算张量在此设备创建。
rt = get_runtime()
device, dtype = rt.device, rt.dtype
from experiments.demo_torch_kernels import group_qdq, symmetric_qdq

# 第 1 步：固定权重，分开生成校准输入与测试输入；测试不参与选择 alpha。
rng = rt.rng(300)
W = rng.normal(size=(4, 8))*.2
calibration = rng.normal(size=(64, 8))
test = rng.normal(size=(128, 8))
calibration[:, 0] *= 20
test[:, 0] *= 12  # 模拟幅度漂移，观察校准最优能否迁移。

# 第 2 步：两个方案都依赖 X/s 和 W*s 的成对变换。
# group_qdq 每行沿输入维切组；symmetric_qdq 的 axis=1 表示每 token 共享尺度。
def evaluate(X, scale, quantize_activation):
    transformed = X/scale
    weight_q = group_qdq(W*scale, bits=4, group_size=4)
    if quantize_activation:
        transformed = symmetric_qdq(transformed, bits=8, axis=1)[0]
    return float(torch.mean((transformed@weight_q.T-X@W.T)**2))

# 第 3 步：激活均值构造 AWQ 风格尺度族，最大幅度构造 SmoothQuant 尺度族。
# 两条路线分别在自己的目标下比较；这里不是完整论文校准流程。
mean_abs = torch.clamp_min(torch.mean(abs(calibration), dim=0), 1e-8)
act_max = torch.clamp_min(torch.amax(abs(calibration), dim=0), 1e-8)
weight_max = torch.clamp_min(torch.amax(abs(W), dim=0), 1e-8)
for name, quantize_activation in [('AWQ风格 W4A16', False), ('平滑 W4A8', True)]:
    candidates = [(None, rt.tensor([1.]*8))]  # 显式包含未缩放基线。
    for alpha in torch.linspace(0, 1, 11, device=device, dtype=dtype):
        scale = (act_max**alpha/weight_max**(1-alpha)
                 if quantize_activation else mean_abs**alpha)
        scale /= torch.sqrt(scale.min()*scale.max())
        assert same((calibration/scale)@(W*scale).T, calibration@W.T)
        candidates.append((float(alpha), scale))

    # 第 4 步：只依据 calibration 选候选；随后一次性报告 test 误差。
    errors = [evaluate(calibration, s, quantize_activation) for _, s in candidates]
    best = int(rt.tensor(errors).argmin())
    alpha, chosen_scale = candidates[best]
    assert errors[best] <= errors[0]
    print(name, '选择 alpha:', alpha, '校准 MSE:', errors[best])
    print('独立测试：未缩放 / 已选缩放 MSE:',
          evaluate(test, rt.tensor([1.]*8), quantize_activation),
          evaluate(test, chosen_scale, quantize_activation))

# 第 5 步：不能断言测试误差一定改善，也不能比较两方案的误差就推出速度排名。
# 本例未实现裁剪搜索、全模型分支融合、打包和低精度内核。

# 收尾：等待本demo设备上的操作完成，方便Colab定位CUDA异步报错。
rt.synchronize()

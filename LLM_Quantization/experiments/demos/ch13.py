"""第 13 章：在同一个 NVFP4 数值格式里搜索全局尺度，再检查激活残差。"""
import torch
from experiments.demo_runtime import get_runtime, same, exact

# 第 0 步：auto优先CUDA；runner可用--device指定。所有计算张量在此设备创建。
rt = get_runtime()
device, dtype = rt.device, rt.dtype
from experiments.demo_torch_kernels import micro_fp4

# 第 1 步：张量最后一维必须是16的倍数，匹配模拟器的 NVFP4 微块。
rng = rt.rng(1300)
W = rng.normal(size=(4, 32))
calibration = rng.normal(size=(48, 32))
test = rng.normal(size=(64, 32))
default_W, metadata = micro_fp4(W, kind='nv')
default_scale = metadata['tensor_scale']

# 第 2 步：枚举全局尺度；每个候选都重新计算 E4M3 块尺度和 E2M1 码字。
# 目标用 calibration 的层输出，不是只看权重 MSE；倍率1保留默认基线。
candidates = []
for multiplier in (.6, .8, 1., 1.2, 1.4):
    quantized, state = micro_fp4(W, kind='nv', tensor_scale=default_scale*multiplier)
    error = calibration@(quantized-W).T
    candidates.append((float(torch.mean(error**2)), multiplier, quantized, state))
best = min(candidates, key=lambda item: item[0])
assert best[0] <= next(row[0] for row in candidates if row[1] == 1.)
print('倍率与校准MSE:', [(m, loss) for loss, m, _, _ in candidates])
print('选择倍率 / 测试MSE:', best[1], torch.mean((test@(best[2]-W).T)**2))

# 第 3 步：仅从保存的三类数值重建，核验尺度方向与微块布局。
state = best[3]
reconstructed = (state['tensor_scale']*state['block_scale']*state['payload']).reshape(W.shape)
assert exact(reconstructed, best[2])
print('声明的NVFP4 payload+scale预算bit每权重:', 4+8/16+32/W.numel())
# 张量仍用PyTorch浮点容器；全局scale用高精度模拟，此预算不是张量实际nbytes。

# 第 4 步：激活也量化时，对比不补偿、精确残差、再量化残差三种输出。
Xq, _ = micro_fp4(test)
residual = test-Xq
Rq, _ = micro_fp4(residual)
Wq = best[2]
reference = test@W.T
outputs = {'不补偿': Xq@Wq.T, '精确残差': test@Wq.T, '量化残差': (Xq+Rq)@Wq.T}
for name, output in outputs.items():
    print(name, '测试输出MSE:', torch.mean((output-reference)**2))

# 第 5 步：完整误差包含两侧交叉项，因此不预设精确残差的输出误差最小。
E_R = Rq-residual
assert same(outputs['量化残差']-reference, test@(Wq-W).T+E_R@Wq.T)
# 本例为尺度搜索和残差诊断，未实现 SOAR/ARCQuant 的完整训练或 GPU 拼接执行。

# 收尾：等待本demo设备上的操作完成，方便Colab定位CUDA异步报错。
rt.synchronize()

"""科研进阶：隔离测试、同格式校准搜索与配对bootstrap，完成一个小型研究闭环。"""
import torch
from experiments.demo_runtime import get_runtime

# 第 0 步：auto优先CUDA；runner可用--device指定。所有计算张量在此设备创建。
rt = get_runtime()
device, dtype = rt.device, rt.dtype

# 第 1 步：先固定数据划分与候选，再看结果；三个集合不会复用同一行输入。
rng = rt.rng(2026)
W = rng.normal(size=(4, 8))
# 人为放大一个权重通道、却让校准集很少激活它，构造可解释的分布错配。
W[:, 0] *= 8
calibration = rng.normal(size=(48, 8))
development = rng.normal(size=(48, 8))
test = rng.normal(size=(128, 8))
calibration[:, 0] *= .01
development[:, 0] *= 2
test[:, 0] *= 2
base_scale = torch.amax(abs(W), dim=1, keepdim=True)/7
candidates = (.5, .75, 1., 1.25, 1.5)

# 第 2 步：所有候选都保存同样的4-bit整数码和每行FP16尺度，固定格式预算。
# 这里搜索统一裁剪倍率，不是DRQ或任何新算法；用于练习研究协议。
models = []
for factor in candidates:
    scales = (base_scale*factor).to(torch.float16)
    codes = torch.clamp(torch.round(W/scales.to(dtype)), -7, 7).to(torch.int8)
    decoded = codes*scales.to(dtype)
    # 将两种数据上的误差分别记录，不能让test决定候选。
    cal_loss = torch.mean((calibration@(decoded-W).T)**2)
    dev_loss = torch.mean((development@(decoded-W).T)**2)
    models.append((decoded, float(cal_loss), float(dev_loss)))
cal_best = min(range(len(models)), key=lambda i: models[i][1])
dev_best = min(range(len(models)), key=lambda i: models[i][2])
baseline = cal_best  # 对照“校准集选参”与“目标域开发集选参”，两者候选数量相同。
print('候选 / 校准MSE / 开发MSE:', [(f, m[1], m[2]) for f, m in zip(candidates, models)])
print('校准选择 / 开发选择的倍率:', candidates[cal_best], candidates[dev_best])
assert cal_best != dev_best  # 只验证这个预设分布错配例子，不作为一般保证。

# 第 3 步：选择冻结后才读取test；每行一个独立样本，保留逐样本差值。
base_errors = torch.mean((test@(models[baseline][0]-W).T)**2, dim=1)
chosen_errors = torch.mean((test@(models[dev_best][0]-W).T)**2, dim=1)
differences = chosen_errors-base_errors  # 负值代表这个MSE目标改善。

# 第 4 步：bootstrap保持同一输入上的基线/候选配对；只估计当前模型的样本波动。
resampling = rt.rng(2027).integers(0, len(test), size=(5000, len(test)))
interval = torch.quantile(differences[resampling].mean(dim=1), rt.tensor([.025, .975]))
assert models[dev_best][2] <= models[baseline][2]
assert torch.isfinite(interval).all()
print('独立测试平均MSE差 / 95%百分位区间:', differences.mean(), interval)

# 第 5 步：有限证据只对应此小模型的层误差，不支持真实LLM任务结论。
print('证据范围：合成线性层、同格式、开发集选参；未测生成质量或GPU速度。')
# 继续实践可换三组随机种子、构造新的测试域；不要反复看同一test后改候选。

# 收尾：等待本demo设备上的操作完成，方便Colab定位CUDA异步报错。
rt.synchronize()

"""第 11 章：综合排错，从错位 scale 定位首个不一致的阶段。"""
import torch
from experiments.demo_runtime import get_runtime, same, exact

# 第 0 步：auto优先CUDA；runner可用--device指定。所有计算张量在此设备创建。
rt = get_runtime()
device, dtype = rt.device, rt.dtype

# 第 1 步：构造两输出通道不同尺度的权重，保留浮点和 QDQ 两个参照。
W = rt.tensor([[.1, .2, -.3, .4], [1., -2., 3., -4.]])
scale = (torch.amax(abs(W), dim=1, keepdim=True)/7).to(torch.float16)
codes = torch.clamp(torch.round(W/scale.to(dtype)), -7, 7).to(torch.int8)
qdq = codes*scale.to(dtype)

# 第 2 步：模拟重载 bug：码字未变，scale 却误按相反输出通道排列。
# 编码相同不保证解码正确，这个错误可以通过逐阶段比较定位。
loaded_codes = codes.clone()
wrong_scale = torch.flip(scale, dims=(0,)).clone()
wrong_decoded = loaded_codes*wrong_scale.to(dtype)
print('编码一致:', exact(loaded_codes, codes))
print('错误解码一致:', exact(wrong_decoded, qdq))
assert exact(loaded_codes, codes)
assert not same(wrong_decoded, qdq)

# 第 3 步：用单位输入逐列探测线性层；这样每个输出直接对应一个权重元素。
probe = torch.eye(4, device=device, dtype=dtype)
expected = probe@qdq.T
actual = probe@wrong_decoded.T
bad_positions = torch.argwhere(abs(expected-actual) > 1e-8)
assert bad_positions.numel() > 0
print('不一致的[输入探针,输出通道]位置:', bad_positions.tolist())

# 第 4 步：修复 scale 映射；验证应先对照 QDQ，再对照原始浮点模型。
fixed = loaded_codes*scale.to(dtype)
assert exact(probe@fixed.T, expected)
print('修复后 QDQ 差异:', torch.amax(abs(fixed-qdq)))
print('仍然存在的量化权重MSE:', torch.mean((fixed-W)**2))

# 第 5 步：两种误差分开记录，避免把导出错误当作量化算法质量差。
stages = {'量化误差': float(torch.mean((qdq-W)**2)),
          '错误导出附加误差': float(torch.mean((wrong_decoded-qdq)**2)),
          '修复导出附加误差': float(torch.mean((fixed-qdq)**2))}
assert stages['修复导出附加误差'] == 0
print('阶段诊断:', stages)
# 练习：将 bug 改成输入列置换，观察为什么必须配对变换输入或恢复列映射。

# 收尾：等待本demo设备上的操作完成，方便Colab定位CUDA异步报错。
rt.synchronize()

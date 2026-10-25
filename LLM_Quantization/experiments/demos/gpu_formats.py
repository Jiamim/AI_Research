"""新架构格式：设备能力提示与MXFP4/6/8数值模拟，CPU/CUDA均可运行。"""
import torch
from experiments.demo_runtime import get_runtime, same
from experiments.demo_torch_kernels import fp_levels, nearest, positive_scale

# 第0步：显式选择设备；CUDA可用只证明通用张量可运行，不等于原生FP4可用。
rt = get_runtime()
device = rt.device
if device.type == 'cuda':
    capability = torch.cuda.get_device_capability(device)
    print('GPU / compute capability:', torch.cuda.get_device_name(device), capability)
    if capability in ((10, 0), (10, 3)):
        print('Blackwell数据中心：查tcgen05、后端版本和尺度布局；本例仍是模拟。')
    elif capability in ((12, 0), (12, 1)):
        print('Blackwell SM12x：查mma.sync及SM120/121专用内核，不复用SM100全部假设。')
    else:
        print('未在本例已核对的Blackwell集合中；不据此声称有原生FP4内核。')
else:
    print('CPU运行：验证数值关系，未检测原生GPU内核。')

# 第1步：同一输入分成32元素微块；第二块整体放大以检查独立尺度。
x = rt.rng(9020).normal(size=(2, 64))
x[:, 32:] *= 12
blocks = x.reshape(2, 2, 32)

# 第2步：构造OCP FP6 E3M2有限值，bias=3，所有指数编码均用于有限值。
# FP4/FP8复用已有表；静态码本送到设备后距离与选码均在设备上计算。
positive6 = [2.**-2*m/4 if e == 0 else 2.**(e-3)*(1+m/4)
             for e in range(8) for m in range(4)]
levels6 = rt.tensor(sorted(set(positive6+[-v for v in positive6])))
assert same(levels6.max(), 28.) and same(levels6[levels6 > 0].min(), .0625)
formats = [('MXFP4', 4, fp_levels('e2m1', x)),
           ('MXFP6 E3M2', 6, levels6), ('MXFP8 E4M3', 8, fp_levels('e4m3', x))]

# 第3步：保守ceil选择二次幂尺度；这不是厂商默认转换recipe的逐位复现。
for name, bits, levels in formats:
    maximum = blocks.abs().amax(dim=-1, keepdim=True)
    # 零块选scale=1；非零块的理想尺度必须可表示，再限制到E8M0有限指数。
    ideal = positive_scale(torch.where(maximum > 0, maximum/levels.max(), torch.ones_like(maximum)))
    scale = 2.**torch.ceil(torch.log2(ideal)).clamp(-127, 127)
    decoded = scale*nearest(blocks/scale, levels)
    assert same(torch.log2(scale), torch.round(torch.log2(scale)))
    assert torch.isfinite(decoded).all()
    print(name, 'MSE:', ((decoded-blocks)**2).mean(), '预算bit/value:', bits+8/32)

# 第4步：FP4最大代表值6，中间没有5；FP6的更多码字不等于任务必然更好。
assert not (fp_levels('e2m1', x) == 5).any()
print('完整部署还要计padding、保留层和工作区；这些数值MSE不能推断内核速度。')
rt.synchronize()

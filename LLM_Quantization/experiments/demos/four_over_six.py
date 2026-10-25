"""Four Over Six的候选比较与随机选优偏差；CPU/CUDA共同数值路径。"""
import torch
from experiments.demo_runtime import get_runtime, same
from experiments.demo_torch_kernels import fp_levels, nearest

# 第0步：用两个16元素微块，固定全局scale=1；这使实际尺度可直接手算。
rt = get_runtime()
x = rt.tensor([[10., 20., 30., 40.]*4, [15., 30., 120., 180.]*4])

# 第1步：对同一原输入计算两个合法E4M3尺度，再选FP4码字。
# 先转尺度再选码，避免错误地使用未保存的理想scale进行误差比较。
restorations, scales = [], []
for limit in (4., 6.):
    maximum = x.abs().amax(dim=1, keepdim=True)
    scale = nearest(maximum/limit, fp_levels('e4m3', x))
    scale = torch.where(maximum == 0, torch.ones_like(scale), scale.clamp(min=2.**-9))
    codes = nearest(x/scale, fp_levels('e2m1', x))
    restorations.append(scale*codes)
    scales.append(scale)
restorations = torch.stack(restorations)
losses = ((restorations-x)**2).mean(dim=-1)
choices = losses.argmin(dim=0)
decoded = restorations[choices, torch.arange(2, device=rt.device)]
assert choices.tolist() == [0, 1]
assert same(decoded, x)
print('4/6候选MSE（行是候选，列是块）:', losses)
print('所选范围上限:', rt.tensor([4., 6.])[choices])

# 第2步：局部最小MSE只针对这两个确定候选。对角尺度仅有二次幂时，
# 理想的s4/s6=1.5无法精确保持，这就是不能原样搬到E8M0的原因。
assert not same(torch.log2(rt.tensor(1.5)), torch.round(torch.log2(rt.tensor(1.5))))

# 第3步：精确枚举两种无偏估计器，但按当前误差选优会引入选择偏差。
# 目标1；A取0或2各一半，B取0.5(概率0.75)或2.5(概率0.25)，均值都是1。
a = rt.tensor([0., 2.])
b = rt.tensor([.5, 2.5])
pa, pb = rt.tensor([.5, .5]), rt.tensor([.75, .25])
assert same(a@pa, 1.) and same(b@pb, 1.)
selected = torch.where((a[:, None]-1).abs() <= (b[None, :]-1).abs(), a[:, None], b[None, :])
expectation = (selected*pa[:, None]*pb[None, :]).sum()
assert same(expectation, .625)
print('分别无偏，按最小误差选择后的期望:', expectation)

# 第4步：这说明统计机制，不冒充4/6完整随机量化器的误差大小。
# 本例FP查表平局取首值，实际CUDA转换的tie规则应按后端核验。
rt.synchronize()

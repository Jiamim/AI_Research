"""梯度舍入与MS-EDEN尺度子步骤：只证明明确的条件期望。"""
import torch
from experiments.demo_runtime import get_runtime, same
from experiments.demo_torch_kernels import fp_levels, nearest, hadamard

# 第0步：构造位于相邻FP4码字间的小梯度，固定网格以排除动态scale干扰。
rt = get_runtime()
x = rt.tensor([.13, .72, 1.22, 2.6])
low, high = rt.tensor([0., .5, 1., 2.]), rt.tensor([.5, 1., 1.5, 3.])
p = (x-low)/(high-low)
expected = (1-p)*low+p*high
variance = (1-p)*(low-x)**2+p*(high-x)**2
rtn = nearest(x, fp_levels('e2m1', x))
assert same(expected, x)
assert torch.all(variance >= (rtn-x)**2-1e-12)
print('SR期望 / 方差 / RTN误差:', expected, variance, rtn-x)

# 第1步：搭建一个16元素EDEN缩放子问题，得到确定的码字与幅度修正。
# 这不是Quartet II完整128维旋转和逐16元素修正的算法。
original = rt.rng(14030).normal(size=(16,))
R = hadamard(16, device=rt.device)
rotated = original@R
scale = rt.tensor(.25)
q = nearest(rotated/scale, fp_levels('e2m1', original))
reconstruction = scale*q
denominator = rotated@reconstruction
assert denominator > 0
correction = (rotated@rotated)/denominator
target_scale = correction*scale

# 第2步：固定码字后，对修正scale做FP8随机舍入，精确枚举两个结果。
# 目标必须在两个合法有限值内；若溢出/下溢，此无偏关系不再自动成立。
levels = fp_levels('e4m3', original)
lower = levels[levels <= target_scale].max()
upper = levels[levels >= target_scale].min()
probability = (target_scale-lower)/(upper-lower) if upper > lower else target_scale.new_zeros(())
expected_decoded = ((1-probability)*lower+probability*upper)*q
assert same(expected_decoded, correction*reconstruction)
assert same(rotated@(correction*reconstruction), rotated@rotated)
print('scale修正 / 上舍概率:', target_scale, probability)

# 第3步：径向内积匹配不保证每个坐标恢复，完整无偏还依赖旋转分布等条件。
back = expected_decoded@R.T
assert not same(back, original)
print('条件期望重建与原向量MSE:', ((back-original)**2).mean())

# 第4步：误差与偏差是两种指标；不能由这个条件期望断言有限维MS-EDEN严格无偏，
# 也不能由SR无偏断言它在所有训练预算中优于RTN。
rt.synchronize()

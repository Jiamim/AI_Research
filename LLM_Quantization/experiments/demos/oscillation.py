"""TetraJet-v2 OsciReset子步骤：识别网格边界振荡并观察一次重置。"""
import torch
from experiments.demo_runtime import get_runtime, same

# 第0步：三个权重轨迹分别表示边界振荡、平滑移动、完全静止。
rt = get_runtime()
trajectory = rt.tensor([[.49, .10, .20], [.51, .15, .20], [.49, .20, .20],
                        [.51, .25, .20], [.49, .30, .20], [.51, .35, .20]])
step = rt.tensor(1.)
codes = torch.round(trajectory/step)
quantized = codes*step

# 第1步：累计主权重距离和量化重建距离，按权重逐项计算OsciRisk。
distance_master = torch.diff(trajectory, dim=0).abs().sum(dim=0)
distance_quantized = torch.diff(quantized, dim=0).abs().sum(dim=0)
# 恒定轨迹定义risk=0；先替换分母避免在where未选分支仍出现0/0。
safe = torch.where(distance_master > 0, distance_master, torch.ones_like(distance_master))
risk = torch.where(distance_master > 0, distance_quantized/safe, torch.zeros_like(safe))
assert same(risk, [50., 0., 0.])

# 第2步：超过阈值的权重重置到当前码字中心；此时码字不变，主权重却改变。
master = trajectory[-1].clone()
reset_mask = risk >= 8
after = torch.where(reset_mask, quantized[-1], master)
assert not same(after, master)
assert same(torch.round(after/step), codes[-1])
print('risk / reset前 / reset后:', risk, master, after)

# 第3步：展示额外成本：若外层每128值一个FP32 scale，内层每16值一个FP8，
# payload+两级scale为4+8/16+32/128=4.75 bit/value；这不是每tensor一个全局scale。
bits = rt.tensor(4+8/16+32/128)
assert same(bits, 4.75)
print('双块尺度预算bit/value:', bits)

# 第4步：实际OsciReset周期跟踪部分参数，跟踪状态也占内存；一次reset并不保证
# 长期损失下降。OutControl的高精度通道与对应计算成本未在此标量示例中实现。
rt.synchronize()

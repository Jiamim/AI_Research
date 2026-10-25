"""第 14 章：用完整 MoE 输出误差分配专家位宽，并检查路由边界。"""
import itertools
import torch
from experiments.demo_runtime import get_runtime, same

# 第 0 步：auto优先CUDA；runner可用--device指定。所有计算张量在此设备创建。
rt = get_runtime()
device, dtype = rt.device, rt.dtype
from experiments.demo_torch_kernels import symmetric_qdq, softmax

# 第 1 步：3个线性专家、每token选2个；这个模型省略真实专家中的门控MLP。
rng = rt.rng(1400)
X = rng.normal(size=(48, 4))
experts = rng.normal(size=(3, 2, 4))
router = rng.normal(size=(3, 4)).to(torch.float16).to(dtype)
logits = X@router.T
top = torch.argsort(logits, dim=1)[:, -2:]
gate = torch.zeros_like(logits)
selected_logits = torch.gather(logits, 1, top)
gate.scatter_(1, top, softmax(selected_logits))
assert same(gate.sum(dim=1), 1., allow_scalar_broadcast=True)

# 第 2 步：对每个专家试2/4/8 bit；缓存输出以免穷举重复量化。
# 首先固定gate，只比较专家量化，防止把路由变化混进误差归因。
reference_experts = torch.stack([X@W.T for W in experts], dim=1)
reference = torch.sum(gate[:, :, None]*reference_experts, dim=1)
outputs = {}
for e in range(3):
    for b in (2, 4, 8):
        # 每行4个权重共用一个尺度；用实际FP16尺度重建，避免隐藏高精度元数据。
        _, codes, scale = symmetric_qdq(experts[e], bits=b, axis=1)
        decoded = codes*scale.to(torch.float16).to(dtype)
        outputs[e, b] = X@decoded.T

# 第 3 步：每专家8个权重、2个FP16尺度和1字节位宽标签。
# router固定FP16，三个候选共享同一份；这里无对齐开销，未实际打包。
router_bits = router.numel()*16
budget = 3*(8*4+2*16+8)+router_bits
records = []
for allocation in itertools.product((2, 4, 8), repeat=3):
    cost = sum(8*b+2*16+8 for b in allocation)+router_bits
    if cost > budget:
        continue
    actual = sum(gate[:, e, None]*outputs[e, b] for e, b in enumerate(allocation))
    loss = float(torch.mean((actual-reference)**2))
    records.append((loss, allocation, cost))
best = min(records)
uniform = next(row for row in records if row[1] == (4, 4, 4))
assert best[0] <= uniform[0] and best[2] <= budget
print('统一4bit / 最佳方案（MSE,各专家位宽,bit预算）:', uniform, best)

# 第 4 步：说明加权局部误差不是最终输出误差，跨专家项会抵消。
errors = rt.tensor([[1., 0.], [-1., 0.]])
weights = rt.tensor([.5, .5])
actual_error = torch.sum((weights@errors)**2)
proxy = weights@torch.sum(errors**2, dim=1)
assert actual_error == 0 and proxy == 1

# 第 5 步：路由分数靠近边界时很小扰动即可换专家；位宽优化后还须测完整路由。
near_tie = rt.tensor([1., .99, -1.])
perturbed = near_tie+rt.tensor([-.02, .02, 0.])
assert near_tie.argmax() != perturbed.argmax()
print('路由top1改变:', near_tie.argmax(), '→', perturbed.argmax())
# 这里没有独立测试、ILP或真实MoE内核，最佳只针对这批输入与有限候选。

# 收尾：等待本demo设备上的操作完成，方便Colab定位CUDA异步报错。
rt.synchronize()

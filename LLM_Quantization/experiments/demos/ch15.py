"""第 15 章：MLA 内容/位置误差路径，以及循环状态反复量化。"""
import math
import torch
from experiments.demo_runtime import get_runtime, same, exact

# 第 0 步：auto优先CUDA；runner可用--device指定。所有计算张量在此设备创建。
rt = get_runtime()
device, dtype = rt.device, rt.dtype
from experiments.demo_torch_kernels import symmetric_qdq, softmax

# 第 1 步：构造一个吸收投影后的单头 MLA；C 同时用于匹配与聚合。
# KR只进入匹配，B负责把潜变量映射到最终输出，不保存完整K/V。
rng = rt.rng(1500)
C = rng.normal(size=(8, 4))
KR = rng.normal(size=(8, 2))
query_content = rng.normal(size=(1, 4))
query_rope = rng.normal(size=(1, 2))
B = rng.normal(size=(4, 3))
temperature = 1/math.sqrt(6)
def attention(content, position):
    logits = temperature*(query_content@content.T+query_rope@position.T)
    probabilities = softmax(logits)
    return probabilities, probabilities@content@B
P, reference = attention(C, KR)

# 第 2 步：按每token分别量化两份缓存，保持布局、查询和投影不变。
Cq = symmetric_qdq(C, bits=3, axis=1)[0]
KRq = symmetric_qdq(KR, bits=3, axis=1)[0]
for name, content, position in [('仅内容', Cq, KR), ('仅位置', C, KRq), ('两者', Cq, KRq)]:
    Pq, output = attention(content, position)
    delta_P, E_C = Pq-P, content-C
    # 内容误差有直接聚合项和交叉项；位置误差只通过 delta_P 传播。
    expanded = delta_P@C@B+P@E_C@B+delta_P@E_C@B
    assert same(output-reference, expanded)
    print(name, 'logit概率MSE / 最终输出MSE:', torch.mean(delta_P**2), torch.mean((output-reference)**2))

# 第 3 步：切换到循环状态这个不同对象；每步都覆盖量化会反复吞掉小更新。
step = .1
exact = rt.tensor(0.)
stored, buffered, remainder = rt.tensor(0.), rt.tensor(0.), rt.tensor(0.)
for t in range(10):
    update = .03
    exact += update
    stored = step*torch.round((stored+update)/step)
    # 高精度余数记录上次未保存的信息，再与本次更新一起量化。
    full_value = buffered+remainder+update
    buffered = step*torch.round(full_value/step)
    remainder = full_value-buffered
    assert same(buffered+remainder, exact)
# 第 4 步：与精确累积量比较；不要仅验证每步量化误差很小就推断长期正确。
assert stored == 0 and same(exact, .3)
print('精确状态 / 反复量化 / 量化主体+余数:', exact, stored, buffered+remainder)
# 余数自身占存储。真实低秩余数的截断和读取不是免费的，本例不证明长期任务无损。

# 收尾：等待本demo设备上的操作完成，方便Colab定位CUDA异步报错。
rt.synchronize()

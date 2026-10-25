"""第 10 章：从逐样本损失和时间戳计算 PPL、配对差异与达标吞吐。"""
import torch
from experiments.demo_runtime import get_runtime, same, exact

# 第 0 步：auto优先CUDA；runner可用--device指定。所有计算张量在此设备创建。
rt = get_runtime()
device, dtype = rt.device, rt.dtype

# 第 1 步：模拟不同长度文本的 NLL 总和；这些是演示数据，不是真模型成绩。
token_counts = rt.tensor([2, 8, 5])
nll_sums = rt.tensor([2., 16., 7.5])
ppl = torch.exp(nll_sums.sum()/token_counts.sum())
wrong_ppl = torch.mean(torch.exp(nll_sums/token_counts))
assert not same(ppl, wrong_ppl)
print('正确的token加权PPL / 错误的窗口PPL均值:', ppl, wrong_ppl)

# 第 2 步：同一批题目的两模型结果成对相减，不独立打乱两个模型的题目。
base = rt.tensor([1, 1, 0, 1, 0, 0, 1, 1])
candidate = rt.tensor([1, 0, 1, 1, 1, 0, 1, 1])
differences = candidate-base
rng = rt.rng(1000)
indices = rng.integers(0, len(base), size=(5000, len(base)))
replicates = differences[indices].mean(dim=1)
interval = torch.quantile(replicates, rt.tensor([.025, .975]))
assert same(differences.mean(), .125)
print('准确率差 / 配对bootstrap区间:', differences.mean(), interval)
# 八道题只用于说明操作；真实数据应按独立题目、文档或会话单位重采样。

# 第 3 步：明确观测窗口内三个完整请求的首/末token时间和输出数量。
# 这里无失败和跨窗口请求；生产统计还必须记录超时、拒绝和部分输出。
arrival = rt.tensor([0., 0., .2])
first = rt.tensor([.1, .3, .35])
last = rt.tensor([.3, .9, .65])
output_tokens = rt.tensor([3, 4, 4])
window = 2.
assert torch.all(last <= window) and torch.all(output_tokens > 1)
ttft = first-arrival
tpot = (last-first)/(output_tokens-1)
good = (ttft <= .2) & (tpot <= .15)

# 第 4 步：goodput 先筛延迟门槛再除窗口；吞吐不等于每请求速度之和。
request_throughput = len(arrival)/window
goodput = good.sum()/window
token_throughput = output_tokens.sum()/window
assert exact(good, [True, False, True])
print('TTFT / TPOT:', ttft, tpot)
print('请求吞吐 / 延迟达标请求吞吐 / token吞吐:',
      request_throughput, goodput, token_throughput)
# 延迟达标并不表示答案正确；质量应使用独立门槛。

# 收尾：等待本demo设备上的操作完成，方便Colab定位CUDA异步报错。
rt.synchronize()

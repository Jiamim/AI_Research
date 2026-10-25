"""第 00 章：从模型配置计算权重与 KV 容量，不分配大型张量。"""
import torch
from experiments.demo_runtime import get_runtime

# 第 0 步：容量公式也用目标设备张量求和，Colab选择GPU后实际执行CUDA运算。
rt = get_runtime()
device, dtype = rt.device, rt.dtype

# 第 1 步：声明一个假想的 GQA 解码器，避免把参数量和缓存量混为一谈。
# 所有 Linear 无 bias，两个 RMSNorm；embedding 与输出头共享且保留 FP16。
layers, hidden, intermediate = 32, 4096, 11008
query_heads, kv_heads, head_dim, vocabulary = 32, 8, 128, 32000
assert hidden == query_heads*head_dim
kv_dim = kv_heads*head_dim
shapes = [(hidden, hidden), (kv_dim, hidden), (kv_dim, hidden),
          (hidden, hidden), (intermediate, hidden),
          (intermediate, hidden), (hidden, intermediate)]

# 第 2 步：按实际张量形状计数；KV 投影的行数比 Query 投影少。
shape_tensor = rt.tensor(shapes, dtype=torch.int64)
linear_parameters = layers*torch.prod(shape_tensor, dim=1).sum()
unquantized_parameters = rt.tensor(vocabulary*hidden + (2*layers+1)*hidden, dtype=torch.int64)
parameters = linear_parameters+unquantized_parameters
print('逻辑参数量（十亿）:', round(float(parameters)/1e9, 4))

# 第 3 步：计算每行分组的尾部空间、4-bit payload 和每组 FP16 scale。
# 这是明确布局下的预算，不包含内核重排副本、框架开销或未声明的零点。
group_size = 128
groups = layers*(shape_tensor[:, 0]*torch.div(shape_tensor[:, 1]+group_size-1, group_size, rounding_mode='floor')).sum()
payload_bytes = groups*group_size*4//8
scale_bytes = groups*2
weight_bytes = payload_bytes+scale_bytes+unquantized_parameters*2
fp16_bytes = parameters*2
print('FP16 / 分组 W4 权重 GiB:', fp16_bytes/2**30, weight_bytes/2**30)
print('W4 payload / scale / 保留权重 bytes:',
      payload_bytes, scale_bytes, unquantized_parameters*2)
assert weight_bytes > parameters*4/8  # 保留层和元数据不能消失。

# 第 4 步：独立扫描上下文与并发；这里让整个 KV 都是 FP16。
# K、V 各一份，所以系数为 2；并发或长度翻倍，payload 也翻倍。
batches = rt.tensor([1, 8], dtype=torch.int64)
lengths = rt.tensor([2048, 8192], dtype=torch.int64)
cache_bytes = 2*batches[:, None]*layers*lengths[None, :]*kv_heads*head_dim*2
print('并发1/8 × 长度2048/8192，KV GiB:', cache_bytes.to(dtype)/2**30)
print('权重+KV GiB:', (cache_bytes+weight_bytes).to(dtype)/2**30)
assert int(cache_bytes[0, 1]) == 2**30

# 第 5 步：改变 layers/kv_heads/group_size 后重算；不要据此承诺模型能装下。
# 真实部署还需加入激活、工作区、页碎片以及运行时峰值。
rt.synchronize()

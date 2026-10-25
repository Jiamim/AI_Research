"""第 06 章：训练小型加权码本，编码、检查空簇并计算真实成本。"""
import torch
from experiments.demo_runtime import get_runtime, exact

# 第 0 步：auto优先CUDA；runner可用--device指定。所有计算张量在此设备创建。
rt = get_runtime()
device, dtype = rt.device, rt.dtype

# 第 1 步：给每个权重附一个正的重要性；固定数据以便逐步手算。
# 重要性可以来自估计器，这里人为指定，不冒充任务 Hessian。
w = rt.tensor([-1., -.85, -.2, -.12, -.02, .08, .4, .9])
importance = rt.tensor([1., 1., 2., 8., 8., 2., 1., 1.])
centers = rt.tensor([-1., -.3, .2, 1.])
history = []

# 第 2 步：交替选最近码字和更新加权均值。
# 固定码本时，正的重要性不改变单元素的最近邻；共享中心更新才受其影响。
for iteration in range(12):
    indices = torch.argmin((w[:, None]-centers)**2, dim=1)
    for k in range(len(centers)):
        members = indices == k
        if members.any():
            centers[k] = torch.sum(importance[members]*w[members])/importance[members].sum()
        # 空簇保留旧中心，避免对空集合求均值产生 NaN。
    loss = torch.sum(importance*(w-centers[indices])**2)
    history.append(float(loss))
assert torch.all(torch.diff(rt.tensor(history)) <= 1e-12)

# 第 3 步：实际保存 FP16 码本后重新选码；保存精度会改变最近邻边界。
stored_centers = centers.to(torch.float16)
indices = torch.argmin((w[:, None]-stored_centers.to(dtype))**2, dim=1).to(torch.uint8)
decoded = stored_centers[indices.long()].to(dtype)
print('FP16码本 / 索引 / 重建:', stored_centers, indices, decoded)
print('加权损失:', torch.sum(importance*(w-decoded)**2))

# 第 4 步：4 个中心需要每索引 2 bit；四个索引低位在前装成一个字节。
# 位移和归约保留在所选设备；不足4个码补零，再按原长度裁掉。
padded = torch.nn.functional.pad(indices.long(), (0, (-indices.numel()) % 4))
shifts = torch.arange(4, device=device)*2
packed = torch.sum(padded.reshape(-1, 4) << shifts, dim=1).to(torch.uint8)
unpacked = ((packed.long()[:, None] >> shifts) & 3).flatten()[:len(indices)]
assert exact(unpacked, indices)

# 第 5 步：小矩阵的码本摊销很贵，不能仅用 2 bit 宣称实际压缩率。
bytes_used = packed.nbytes+stored_centers.nbytes
print('小例 payload+码本 bytes / 实际bit每权重:', bytes_used, bytes_used*8/len(w))
for parameter_count in (8, 1024):
    estimated_bytes = (parameter_count+3)//4+stored_centers.nbytes
    print('共享同一码本的权重数 / 预算bit每权重:',
          parameter_count, estimated_bytes*8/parameter_count)
# 这里没有行尺度、稀疏例外或文件头；它们在真实格式中仍需另计。

# 收尾：等待本demo设备上的操作完成，方便Colab定位CUDA异步报错。
rt.synchronize()

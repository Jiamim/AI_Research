"""第 01 章：分组 INT4 编码、物理打包、解码与输出误差。"""
import torch
from experiments.demo_runtime import get_runtime, same, exact

# 第 0 步：auto优先CUDA；runner可用--device指定。所有计算张量在此设备创建。
rt = get_runtime()
device, dtype = rt.device, rt.dtype

# 第 1 步：W 的形状是 [输出通道, 输入通道]，每行沿输入维切成 4 个值一组。
# 第二行含全零组，用来检查 scale=0 的退化情况。
W = rt.tensor([[-1., -.3, .2, .9, .03, .11, -.07, .15],
              [0., 0., 0., 0., 1.2, -.8, .4, -.1]])
group = 4
blocks = W.reshape(2, 2, group)
amax = torch.amax(abs(blocks), dim=-1, keepdim=True)

# 第 2 步：先把 scale 转成真正要存储的 FP16，再用该尺度选码。
# 非零组用 [-7,7]；全零组用有效尺度 1，避免除零。
scales = torch.where(amax == 0, torch.ones_like(amax), amax/7).to(torch.float16)
q = torch.clamp(torch.round(blocks/scales.to(dtype)), -7, 7).to(torch.int8)
qdq = (q*scales.to(dtype)).reshape(W.shape)

# 第 3 步：4-bit 补码 c=q mod 16；两个相邻编码组成一个字节。
# 仅 to(torch.int8) 仍然一值一字节，这一步才真的把 payload 减半。
codes = (q.reshape(-1).to(torch.int16) & 15).to(torch.uint8)
packed = codes[0::2] | (codes[1::2] << 4)
assert packed.nbytes == W.numel()//2

# 第 4 步：独立拆出高低 nibble；编码 >=8 表示负数，应减去 16。
unpacked = torch.empty(W.numel(), device=device, dtype=torch.int16)
unpacked[0::2] = packed & 15
unpacked[1::2] = packed >> 4
signed = torch.where(unpacked >= 8, unpacked-16, unpacked)
decoded = (signed.reshape(blocks.shape)*scales.to(dtype)).reshape(W.shape)
assert exact(signed, q.reshape(-1))
assert exact(decoded, qdq)
assert exact(decoded[1, :4], torch.zeros(4, device=device, dtype=dtype))

# 第 5 步：统计真实数组字节；这里不包含形状、格式标签或文件容器的开销。
bytes_used = packed.nbytes+scales.nbytes
print('整数码:', q.reshape(W.shape))
print('打包字节:', packed.tolist())
print('payload+scale bytes / bit每权重:', bytes_used, bytes_used*8/W.numel())

# 第 6 步：让一批输入通过原层与解码层，观察权重误差怎样传到输出。
X = rt.tensor([[1., 2., 0., -1., .2, .3, .4, .5], [1.]*8])
error = X@decoded.T-X@W.T
assert same(error, X@(decoded-W).T)
print('权重 MSE / 输出 MSE:', torch.mean((decoded-W)**2), torch.mean(error**2))
# 解码后用浮点矩阵乘验证数值，并没有执行 GPU INT4 GEMM。

# 收尾：等待本demo设备上的操作完成，方便Colab定位CUDA异步报错。
rt.synchronize()

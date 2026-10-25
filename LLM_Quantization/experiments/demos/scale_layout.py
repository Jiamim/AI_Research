"""尺度数值转换与字节布局分离：UE8M0、repack以及块缩放GEMM。"""
import torch
from experiments.demo_runtime import get_runtime, same, exact

# 第0步：有限UE8M0编码c映射为2^(c-127)，0xff保留为NaN，不能当普通指数。
rt = get_runtime()
scale = rt.tensor([.3, .7, 1.1, 3.1])
exponent = torch.ceil(torch.log2(scale)).to(torch.int64)
codes = exponent+127
assert torch.all((codes >= 0) & (codes < 255))
decoded = 2.**(codes.to(rt.dtype)-127)
assert not same(decoded, scale)  # FP32/64尺度转二次幂是重新量化，不是无损repack。

# 第1步：把四个尺度字节装入一个32-bit词的位模式，在int64容器中安全移位。
shifts = torch.arange(4, device=rt.device)*8
word = torch.sum(codes << shifts)
unpacked = (word >> shifts) & 255
assert exact(unpacked, codes)
assert same(2.**(unpacked.to(rt.dtype)-127), decoded)
print('尺度转换误差 / 打包词:', decoded-scale, word)

# 第2步：教学重排改变物理次序但不改变值；真实swizzle必须从后端源码读取。
order = rt.tensor([2, 0, 3, 1], dtype=torch.int64)
swizzled = codes[order]
restored_codes = swizzled[torch.argsort(order)]
assert exact(restored_codes, codes)

# 第3步：沿K切成两个块，每个部分和分别恢复尺度，再求总和。
A = rt.tensor([[1., 2., -1., .5], [.2, -.4, 1., 2.]])
B = rt.tensor([[1., .2, -.3, 2.], [.4, 1., 2., -.5], [-1., .5, 1., 1.]])
sa = rt.tensor([[.5, 2.], [1., .25]])
sb = rt.tensor([[2., .5], [.5, 1.], [1., 2.]])
full = (A*sa.repeat_interleave(2, dim=1))@(B*sb.repeat_interleave(2, dim=1)).T
parts = sum((A[:, 2*g:2*g+2]@B[:, 2*g:2*g+2].T)*sa[:, g, None]*sb[:, g][None, :]
            for g in range(2))
assert same(full, parts)
wrong = (A@B.T)*sa[:, :1]*sb[:, 0][None, :]
assert not same(wrong, full)
print('正确块缩放输出:', full)

# 第4步：只验证数值契约；这里不是DeepGEMM/FlashInfer的真实scale swizzle或原生GEMM。
rt.synchronize()

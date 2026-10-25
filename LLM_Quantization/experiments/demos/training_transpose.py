"""NVFP4训练二维尺度：前向与Dgrad读取相同权重的必要条件。"""
import torch
from experiments.demo_runtime import get_runtime, same
from experiments.demo_torch_kernels import micro_fp4, fp4_block_qdq, finite

# 第0步：32×32权重包含明显的大幅度行，放大行/列分组差异。
rt = get_runtime()
W = rt.rng(14021).normal(size=(32, 32))
W[0] *= 12
dy = rt.rng(14022).normal(size=(4, 32))

# 第1步：沿当前末轴做1×16量化，再对转置权重重新量化。
row_q, _ = micro_fp4(W)
column_q, _ = micro_fp4(W.T.contiguous())
assert not same(row_q.T, column_q)
print('1D两个方向的权重重建MSE:', ((row_q.T-column_q)**2).mean())

# 第2步：每16×16方形块共享一个实际FP8尺度，转置时块内最大值不变。
def square_qdq(weight):
    weight = finite(weight)
    if weight.ndim != 2 or any(size % 16 for size in weight.shape):
        raise ValueError('二维权重的行列均须是16的倍数')
    rows, columns = weight.shape
    blocks = weight.reshape(rows//16, 16, columns//16, 16).permute(0, 2, 1, 3)
    # 每个256值方块只计算一次尺度；复用带零块/溢出检查的数学子步骤。
    # 这里把方块展平是为了归约，不表示硬件原生消费256值尺度。
    decoded, _ = fp4_block_qdq(blocks.reshape(rows//16, columns//16, 256))
    restored = decoded.reshape(blocks.shape)
    return restored.permute(0, 2, 1, 3).reshape(weight.shape)
square = square_qdq(W)
square_transposed = square_qdq(W.T.contiguous())
assert same(square.T, square_transposed)
# 零张量和单个零块必须在除法前得到合法正尺度，不能靠0×错误码字凑出零。
assert same(square_qdq(torch.zeros_like(W)), torch.zeros_like(W))
with_zero_block = W.clone()
with_zero_block[:16, :16] = 0
assert same(square_qdq(with_zero_block)[:16, :16], torch.zeros_like(W[:16, :16]))

# 第3步：Y=XWᵀ的输入梯度应为dy@W；换成另一方向量化权重会改变梯度。
consistent_dgrad = dy@row_q
inconsistent_dgrad = dy@column_q.T
assert not same(consistent_dgrad, inconsistent_dgrad)
assert same(dy@square, dy@square_transposed.T)
print('Dgrad因权重不一致增加的MSE:', ((consistent_dgrad-inconsistent_dgrad)**2).mean())
print('1D / 2D权重MSE:', ((row_q-W)**2).mean(), ((square-W)**2).mean())

# 第4步：2D共享尺度是算法约束，实际Tensor Core仍消费1×16尺度布局，
# 常需要复制scale，不能仅把4.5 bit改成4+8/256宣称硬件存储收益。
print('每16×16块：1个逻辑尺度，喂给1×16布局可能需复制16次。')
rt.synchronize()

"""LLM.int8 专题：手算分流、阈值扫描和拆分batch的数值差异。"""
import torch
from experiments.demo_runtime import get_runtime, same, exact

# 第 0 步：auto优先CUDA；runner可用--device指定。所有计算张量在此设备创建。
rt = get_runtime()
device, dtype = rt.device, rt.dtype
from experiments.demo_torch_kernels import int8_matmul, split_matmul, error_metrics

# 第 1 步：构造4维输入，第4列有离群值；权重按[输出,输入]排列。
X = rt.tensor([[.02, -.04, .10, 30.], [.08, -.06, .02, -25.]])
W = rt.tensor([[.4, -.7, .2, .05], [-.3, .1, .8, -.04]])
reference = X@W.T
ordinary, state = int8_matmul(X, W)

# 第 2 步：检查每token激活尺度与每输出通道权重尺度形成外积。
# helper在目标设备用FP64精确承载小维度整数点积，再转INT32记录。
# 这是兼容CUDA的数值模拟，不是INT8 Tensor Core性能实现。
rebuilt = state['accumulator']*(state['sx']*state['sw'].T)
assert exact(ordinary, rebuilt)
print('普通INT8 NMSE:', error_metrics(reference, ordinary)['nmse'])

# 第 3 步：阈值只决定列集合；使用原始权重的FP64支路隔离量化机制。
# 真实bitsandbytes会使用不同缓存、布局和分支精度，本例不声称逐位一致。
for threshold in (.01, 6., 40.):
    mixed, info = split_matmul(X, W, threshold=threshold)
    assert same(mixed, info['y_regular']+info['y_outlier'])
    print('阈值 / 高精度列 / NMSE:', threshold,
          torch.nonzero(info['outliers']).flatten().tolist(), error_metrics(reference, mixed)['nmse'])
    if threshold == 6.:
        assert exact(info['outliers'], [False, False, False, True])
    if threshold == 40.:
        assert same(mixed, ordinary)

# 第 4 步：同一列仅在某个token超阈值，整batch检测仍会保护该列的全部token。
batched = rt.tensor([[7., .2, -.3, .1], [.2, .7, .4, -.1]])
together, info = split_matmul(batched, W, threshold=6.)
separate = torch.cat([split_matmul(row[None, :], W, threshold=6.)[0] for row in batched])
assert exact(info['outliers'], [True, False, False, False])
print('一起处理与分开处理的最大差异:', torch.amax(abs(together-separate)))
# 浮点线性层逐行等价，但这里离群集合和权重尺度会随batch变化，量化输出可能不同。

# 收尾：等待本demo设备上的操作完成，方便Colab定位CUDA异步报错。
rt.synchronize()

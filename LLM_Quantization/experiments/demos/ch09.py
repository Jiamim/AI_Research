"""第 09 章：把自定义 INT4 格式写入临时文件，重载后验证层输出。"""
import tempfile
from pathlib import Path
import math
import numpy as np  # 仅用于NPZ文件I/O，数值运算用PyTorch。
import torch
from experiments.demo_runtime import get_runtime, exact

# 第 0 步：auto优先CUDA；runner可用--device指定。所有计算张量在此设备创建。
rt = get_runtime()
device, dtype = rt.device, rt.dtype

# 第 1 步：声明格式契约；每行是一组，FP16 scale，补码 nibble 低位在前。
# 这个格式是教学协议，不是 GGUF、GPTQ 或 AWQ 的实际布局。
W = rt.tensor([[.3, -.7, 1., -.2], [.1, .6, -.3, .8]])
scale = (torch.amax(abs(W), dim=1, keepdim=True)/7).to(torch.float16)
codes = torch.clamp(torch.round(W/scale.to(dtype)), -7, 7).to(torch.int8)
raw = (codes.flatten().to(torch.int16) & 15).to(torch.uint8)
payload = raw[::2] | (raw[1::2] << 4)
reference_decoded = codes*scale.to(dtype)

# 第 2 步：临时目录随运行结束清理；不会修改项目里已有模型或结果。
with tempfile.TemporaryDirectory(prefix='int4_demo_') as directory:
    filename = Path(directory)/'linear.npz'
    # 只有文件I/O边界搬到CPU；读取后再明确搬回目标设备。
    np.savez(filename, payload=payload.detach().cpu().numpy(),
             scale=scale.detach().cpu().numpy(), shape=np.array(W.shape),
             format_version=np.array(1), group_axis=np.array(1))
    file_bytes = filename.stat().st_size

    # 第 3 步：禁用 pickle，只读取数值字段；先检查版本和形状再解码。
    with np.load(filename, allow_pickle=False) as archive:
        assert int(archive['format_version']) == 1 and int(archive['group_axis']) == 1
        shape = tuple(int(x) for x in archive['shape'])
        stored_scale = rt.tensor(archive['scale'].copy(), dtype=torch.float16)
        stored_payload = rt.tensor(archive['payload'].copy(), dtype=torch.uint8)
    assert shape == (2, 4) and stored_scale.shape == (2, 1)
    assert stored_payload.numel()*2 == math.prod(shape)

    # 第 4 步：重载过程只用文件字段，不借用原始权重或原始整数码。
    unsigned = torch.empty(stored_payload.numel()*2, device=device, dtype=torch.int16)
    unsigned[::2] = stored_payload & 15
    unsigned[1::2] = stored_payload >> 4
    signed = torch.where(unsigned >= 8, unsigned-16, unsigned)
    restored = signed.reshape(shape)*stored_scale.to(dtype)
    assert exact(restored, reference_decoded)

    # 第 5 步：同一个输入同时通过内存 QDQ 参考与文件重载结果。
    X = rt.tensor([[1., 2., -.5, .3], [.2, -.4, .8, 1.]])
    assert exact(X@restored.T, X@reference_decoded.T)
    print('payload+scale / NPZ实际文件 bytes:', payload.nbytes+scale.nbytes, file_bytes)
    print('重载层相对原始层的MSE:', torch.mean((X@restored.T-X@W.T)**2))

# 第 6 步：NPZ 的字段/容器开销对小文件占比很大；payload 与文件大小不是同一指标。
# 本例恢复浮点权重做矩阵乘，只验证保存契约，不能证明低比特推理加速。

# 收尾：等待本demo设备上的操作完成，方便Colab定位CUDA异步报错。
rt.synchronize()

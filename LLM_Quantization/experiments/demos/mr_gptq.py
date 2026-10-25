"""MR-GPTQ核心消融：微块旋转、固定FP4尺度、顺序补偿；非完整论文实现。"""
import torch
from experiments.demo_runtime import get_runtime, same
from experiments.demo_torch_kernels import hadamard, micro_fp4, fp_levels, nearest

# 第0步：所有大于标量的数据在指定设备；输入随机流一致，便于CPU/CUDA核对。
rt = get_runtime()
device = rt.device
W = rt.rng(13021).normal(size=(2, 32))*.2
X = rt.rng(13022).normal(size=(96, 32))
test = rt.rng(13023).normal(size=(64, 32))
X[:, 0] *= 8
test[:, 0] *= 5

# 第1步：固定共享网格逐列补偿。每一步只允许尚未固定的列连续变化。
def compensate(weight, hessian, scales):
    current = weight.clone()
    result = torch.empty_like(weight)
    inverse = torch.linalg.inv(hessian)
    levels = fp_levels('e2m1', weight)
    for column in range(weight.shape[1]):
        q = scales[:, column]*nearest(current[:, column]/scales[:, column], levels)
        epsilon = q-current[:, column]
        current[:, column:] += epsilon[:, None]*(inverse[0]/inverse[0, 0])[None, :]
        result[:, column] = q
        # 独立核验一：已固定的全部列必须等于保存码字，检查更新方向和索引。
        assert same(current[:, :column+1], result[:, :column+1])
        if inverse.shape[0] > 1:
            inverse = inverse[1:, 1:]-torch.outer(inverse[1:, 0], inverse[0, 1:])/inverse[0, 0]
            # 独立核验二：直接从原始H求自由变量的条件最优解，不复用逆更新公式。
            # H_ff(w_f-w0_f)=-H_fc(q_c-w0_c)，正定H保证连续子问题唯一最优。
            free_h = hessian[column+1:, column+1:]
            fixed_error = result[:, :column+1]-weight[:, :column+1]
            rhs = hessian[column+1:, :column+1]@fixed_error.T
            optimum = weight[:, column+1:]-torch.linalg.solve(free_h, rhs).T
            assert same(current[:, column+1:], optimum, atol=1e-10, rtol=1e-8)
            # 独立核验三：Schur消元必须等于剩余主子矩阵的逆。
            assert same(inverse, torch.linalg.inv(free_h), atol=1e-10, rtol=1e-8)
    return result

# 第2步：分别对照MXFP4/NVFP4，旋转大小匹配32/16的微块。
# 激活始终采用相同格式的RTN；权重RTN与补偿使用完全相同的初始尺度。
for kind, group in [('mx', 32), ('nv', 16)]:
    for rotated in (False, True):
        R = (torch.kron(torch.eye(32//group, device=device, dtype=rt.dtype),
                        hadamard(group, device=device)) if rotated
             else torch.eye(32, device=device, dtype=rt.dtype))
        Wr, Xr = W@R, X@R
        assert same(Xr@Wr.T, X@W.T)
        hessian = 2*Xr.T@Xr
        hessian += .01*hessian.diag().mean()*torch.eye(32, device=device, dtype=rt.dtype)
        baseline, state = micro_fp4(Wr, kind)
        scales = (state['tensor_scale']*state['block_scale']).expand_as(state['payload']).reshape(W.shape)
        corrected = compensate(Wr, hessian, scales)

        # 第3步：验证最终权重仍落在固定解码网格；补偿未增加存储变量。
        assert same(corrected/scales, nearest(corrected/scales, fp_levels('e2m1', W)))
        for name, weight in [('RTN', baseline), ('补偿', corrected)]:
            cal_q, _ = micro_fp4(Xr, kind)
            test_q, _ = micro_fp4(test@R, kind)
            print(kind, '旋转:', rotated, name, '校准/测试MSE:',
                  ((cal_q@weight.T-X@W.T)**2).mean(),
                  ((test_q@weight.T-test@W.T)**2).mean())

# 第4步：未加入尺度搜索与static act-order，也没匹配实际内核的swizzle。
# 不断言补偿/旋转在每个格式和数据集都获胜，特别是NVFP4小微块。
print('已逐步核对固定坐标、连续条件最优解与逆Hessian更新。')
rt.synchronize()

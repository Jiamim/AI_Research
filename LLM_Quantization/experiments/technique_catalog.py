"""Source for diagrams, code explanations and runnable snippets in main chapters.

Run build_technique_docs.py after editing; technique_lab.py executes these same
snippets and verifies the merged Markdown has not drifted from the source.
Full theory prose is maintained directly in the main chapter Markdown files.
"""

from textwrap import dedent

CHAPTERS = {
    "01": "均匀量化与误差", "02": "GPTQ与二阶误差补偿",
    "03": "离群值与缩放", "04": "可学习校准与HQQ",
    "05": "旋转与非相干化", "06": "码本_稀疏与混合精度",
    "07": "QAT_QLoRA与原生低比特", "08": "KV缓存与Attention量化",
    "09": "浮点格式与推理系统",
    "12": "新进展_变换与编码", "13": "新进展_FP4与误差修正",
    "14": "新进展_训练与MoE", "15": "新进展_KV与推理系统",
}
CARDS = []


def add(key, chapter, title, section, principle, steps, code, explanation, boundary, refs):
    CARDS.append(dict(id=key, chapter=chapter, title=title, section=section,
                      principle=principle, steps=steps, code=dedent(code).strip(),
                      explanation=explanation, boundary=boundary, refs=refs))


add("symmetric", "01", "对称均匀量化", "1.1",
    "用正步长 s 把实数映射到以零为中心的整数格点。s 决定分辨率，有限编码范围决定可表示的最大幅度。本例使用 [-7,7] 的 INT4 约定，零可以精确表示。",
    ["输入：-1、-0.3、0.2、0.9", "absmax=1，步长 s=1/7", "除以 s，舍入到 [-7,7]", "乘回 s，得到重建值"],
    '''
    import numpy as np
    x = np.array([-1., -.3, .2, .9])
    s = np.max(abs(x)) / 7
    q = np.clip(np.rint(x / s), -7, 7).astype(np.int8)
    restored = s * q
    assert np.array_equal(q, [-7, -2, 1, 6])
    assert np.max(abs(restored - x)) <= s / 2 + 1e-12
    print("scale, codes, restored:", s, q, restored)
    ''',
    ["先用最大绝对值确定步长，再把 x/s 转换成整数坐标。np.rint 在半整数处采用偶数舍入。",
     "q 是编码，restored 才是参与实数误差比较的重建值；乘回 scale 不会找回舍入丢掉的信息。",
     "第二个断言只适用于本例没有裁剪的范围；被裁剪的大值不满足半步长界。"],
    "此处 int8 是便于观察的容器，四个编码并没有自动打包成两个字节。全零组和显式 bound 校验见 quant_lab.py 的 symmetric_qdq。", "第 01 章基础推导")

add("affine", "01", "仿射量化与整数零点", "1.2",
    "通过整数零点 z 把编码范围平移，使零精确对应某个合法编码。范围必须先包含零；整数零点舍入可能使实际端点相对原 min/max 移动。",
    ["输入范围 [-1,2]", "s=1，零点 z=1", "q=round(x/s)+z", "重建 x_hat=s(q-z)"],
    '''
    import numpy as np
    x = np.array([-1., 0., .7, 2.])
    lo, hi = min(x.min(), 0), max(x.max(), 0)
    s = (hi - lo) / 3 if hi > lo else 1.
    z = int(np.clip(np.rint(-lo/s), 0, 3))
    q = np.clip(np.rint(x/s) + z, 0, 3).astype(int)
    restored = s * (q-z)
    assert z == 1 and restored[1] == 0
    assert np.array_equal(q, [0, 1, 2, 3])
    print("zero point, codes, restored:", z, q, restored)
    ''',
    ["lo/hi 将全正或全负数据的范围扩展到零，这是整数 zero point 的约定。",
     "z 表示数值零的编码；编码 0 在本例表示 -1。反量化必须先减 z 再乘 s。"],
    "浮点 offset 的 min-max 编码与这里的整数零点不同；存储成本还包含 scale 和 zero point。", "第 01 章基础推导")

add("rtn", "01", "RTN：最近格点舍入", "1.4",
    "固定独立网格后，权重平方误差可以按元素分解，因此每个元素选最近点就是最优解。这个结论不包含 scale 搜索，也不包含相关输入导致的输出误差交叉项。",
    ["固定网格 {0,1}", "每个权重独立选最近点", "得到权重平方误差最小值", "相关输入下还需检查输出误差"],
    '''
    import itertools
    import numpy as np
    w = np.array([.6, .6])
    candidates = np.array(list(itertools.product([0., 1.], repeat=2)))
    rtn = np.rint(w)
    weight_cost = np.sum((candidates-w)**2, axis=1)
    output_cost = (candidates.sum(axis=1)-w.sum())**2
    assert np.isclose(np.sum((rtn-w)**2), weight_cost.min())
    assert (rtn.sum()-w.sum())**2 > output_cost.min()
    print("RTN:", rtn, "best output candidate:", candidates[output_cost.argmin()])
    ''',
    ["枚举四个候选组合给出独立参照，验证最近舍入的权重误差最小。",
     "输出取两个权重之和，等价于输入 [1,1]。两个舍入误差同向相加，因此权重最优不等于输出最优。"],
    "枚举只适合这个二维反例；真实模型中的离散搜索无法直接穷举。", "第 01、02 章推导")

add("grouping", "01", "按通道、按 token 与按组量化", "1.5",
    "共享尺度的归约轴决定哪些元素互相影响。权重通常沿输入维分组，激活可按 token 逐行量化；缩小组能够隔离离群值，但增加元数据。",
    ["权重 [0.1,0.2,0.3,10]", "沿输入轴分成 g 个一组", "每组各自统计 scale", "同时比较 MSE 与有效位宽"],
    '''
    import numpy as np
    from experiments.quant_lab import group_qdq, mse
    w = np.array([[.1, .2, .3, 10.]])
    for bits in (4, 8):
        for group in (4, 2):
            restored = group_qdq(w, bits=bits, group_size=group)
            blocks = w.reshape(1, -1, group)
            step = np.max(abs(blocks), axis=-1, keepdims=True)/(2**(bits-1)-1)
            expected = (step*np.rint(blocks/step)).reshape(w.shape)
            assert np.allclose(restored, expected)
            print("bits, group, MSE, effective bits:",
                  bits, group, mse(w, restored), bits + 16/group)
    assert mse(w, group_qdq(w, group_size=2)) < mse(w, group_qdq(w, group_size=4))
    ''',
    ["group_qdq 把 [输出通道,输入通道] reshape 成 [输出通道,组,组内元素]，只在最后一轴求最大值。",
     "16/group 假设每组保存一个 FP16 scale；这里按四种配置同时报告误差和成本，便于修改参数。"],
    "示例只支持完整组，不做 padding；小组精度改善是本例观察，不能据此保证真实任务或速度单调改善。", "第 01 章分组与位宽公式")

add("minmax", "01", "MinMax / AbsMax 校准", "1.8",
    "MinMax 保留观察到的两端范围，AbsMax 将其变成对称范围。优点是不主动丢掉观测值；缺点是一个极端值即可放大所有元素共享的步长。",
    ["普通值加一个离群值 10", "统计 min/max 或 absmax", "范围决定全组步长", "小值可能舍入成零"],
    '''
    import numpy as np
    from experiments.quant_lab import symmetric_qdq, affine_qdq
    x = np.array([.1, .2, .3, 10.])
    symmetric, _, scale = symmetric_qdq(x)
    affine, _, _, zero = affine_qdq(x, bits=4)
    assert np.all(symmetric[:3] == 0)
    print("AbsMax scale:", scale.item(), "output:", symmetric)
    print("Affine output and zero point:", affine, zero)
    ''',
    ["对称量化的步长由 10 决定，前三个值全部小于半步长，信息被舍掉。",
     "仿射版本保留整数零点，不能把全正范围直接映射到不含零的浮点 offset 格式。"],
    "校准集之外仍可能出现更大值；保存全部校准范围不代表完全没有线上溢出。", "第 01 章范围校准")

add("percentile", "01", "Percentile 分位数裁剪", "1.8",
    "用绝对值分位数代替绝对最大值，让极少数尾部元素承担裁剪误差，换取主体更细的量化步长。分位数是超参数，必须结合输出或任务误差选择。",
    ["收集校准值的绝对值", "取 99% 分位数", "尾部裁剪到阈值", "统计主体与尾部的总误差"],
    '''
    import numpy as np
    from experiments.quant_lab import symmetric_qdq, mse
    rng = np.random.default_rng(7)
    x = rng.normal(0, .5, 10000); x[0] = 10.
    bound = np.quantile(abs(x), .99)
    clipped = symmetric_qdq(x, bound=bound)[0]
    full = symmetric_qdq(x)[0]
    assert 0 < bound < np.max(abs(x))
    print("bound, clipped fraction:", bound, np.mean(abs(x) > bound))
    print("MSE percentile, AbsMax:", mse(x, clipped), mse(x, full))
    ''',
    ["quantile 的输入是 abs(x)，所以得到的是对称阈值。尾部占比可能因并列值而不精确等于 1%。",
     "误差仍对原始 x 计算；如果先把 x 裁掉再计算 MSE，会漏掉尾部损失。"],
    "本例不把分位数法优于 AbsMax 写成断言，避免把构造数据的收益误当成定理。", "第 01 章范围校准")

add("mse_clipping", "01", "MSE 裁剪搜索", "1.8",
    "在候选裁剪范围上直接计算原数据与重建数据的平方误差。阈值太大导致舍入粗糙，太小则裁掉尾部；搜索在两种误差之间折中。",
    ["生成候选阈值 c", "分别裁剪、量化、反量化", "比较对原数据的 MSE", "保留候选中误差最小的 c"],
    '''
    import numpy as np
    from experiments.quant_lab import symmetric_qdq, mse
    rng = np.random.default_rng(8)
    x = rng.normal(0, .5, 4096); x[0] = 7.
    bounds = np.linspace(.3, np.max(abs(x)), 80)
    errors = [mse(x, symmetric_qdq(x, bound=c)[0]) for c in bounds]
    best = int(np.argmin(errors))
    assert errors[best] <= errors[-1]
    print("best bound, MSE, AbsMax MSE:", bounds[best], errors[best], errors[-1])
    ''',
    ["候选集合包含 absmax，所以最佳候选的 MSE 不会大于该集合中的 AbsMax。",
     "每次都相对于未裁剪的 x 算误差；此处优化的是权重 MSE，换成层输出误差会得到不同目标。"],
    "这是有限网格上的最优候选，不是连续阈值的全局解析最优。", "第 01 章裁剪误差积分")

add("kl_calibration", "01", "KL 直方图校准", "1.8",
    "比较校准直方图与量化后重建直方图的 KL 散度。必须定义尾部折叠、量化桶合并和空 bin 的处理，否则同名 KL 校准也可能产生不同阈值。",
    ["非负直方图 P", "将阈值外尾部折进末 bin", "合并成低比特桶，再展开为 Q", "选择 KL(P||Q) 小的阈值"],
    '''
    import numpy as np
    from experiments.technique_kernels import histogram_kl
    counts = np.array([100, 90, 75, 60, 45, 30, 20, 10, 5, 3, 2, 1])
    thresholds = range(5, len(counts)+1)
    divergences = [histogram_kl(counts, t, bins=4) for t in thresholds]
    assert min(divergences) >= -1e-12
    assert abs(histogram_kl(np.ones(8), 8, bins=4)) < 1e-12
    print("selected bin count:", list(thresholds)[np.argmin(divergences)])
    print("KL values:", divergences)
    ''',
    ["histogram_kl 先折叠尾部，然后对连续 bins 合并，每个桶只向原本非空的 bins 平均分配质量。",
     "p 和 q 都归一化后，才计算非空位置的 p*log(p/q)。均匀直方图可以无失真地合并、展开，作为独立控制。"],
    "这是明确约定的非负直方图教学算法，不能宣称与某版本 TensorRT 校准器逐位一致；KL 也不等于任务损失。", "第 01 章 KL 定义")

add("integer_matmul", "01", "整数矩阵乘与零点修正", "1.9",
    "在一个归约组内尺度固定时，可以先算整数点积，再补偿零点并乘尺度。尺度沿归约维变化时必须分组累加，不能把它们合并成一个全局系数。",
    ["X、W 编码为 q 和 z", "INT32 点积", "减去零点交叉项", "乘 sx*sw 恢复输出"],
    '''
    import numpy as np
    qx = np.array([2, 5, 7], dtype=np.int32)
    qw = np.array([1, 4, 6], dtype=np.int32)
    zx, zw, sx, sw = 3, 2, .1, .2
    corrected = qx @ qw - zw*qx.sum() - zx*qw.sum() + len(qx)*zx*zw
    integer_output = sx * sw * corrected
    float_output = (sx*(qx-zx)) @ (sw*(qw-zw))
    assert np.isclose(integer_output, float_output)
    print("corrected integer sum, restored output:", corrected, integer_output)
    ''',
    ["展开 (qx-zx)(qw-zw) 后出现两个求和修正项和一个常数项，代码与正文展开式逐项对应。",
     "最后用直接反量化后的浮点内积交叉核对，验证的是编码对应的输出，不是原始未量化输出。"],
    "NumPy 例子不测量 Tensor Core 性能；真实内核还需管理溢出、归约长度与累加类型。", "第 01 章整数点积推导")

add("stochastic_rounding", "01", "随机舍入", "1.10",
    "对相邻两格按距离分配概率，使无裁剪条件下的重建期望等于原值。它保留的是期望，不是单次精度，非线性网络也不会因此整体无偏。",
    ["x/s=k+r", "采样均匀随机数 u", "u<r 时选 k+1，否则选 k", "多次平均接近原始 x"],
    '''
    import numpy as np
    rng = np.random.default_rng(9)
    x, s = .26, .1
    base = np.floor(x/s)
    fraction = x/s-base
    samples = s*(base + (rng.random(100000) < fraction))
    se = samples.std(ddof=1)/np.sqrt(len(samples))
    assert abs(samples.mean()-x) < 6*se
    print("mean, target, standard error:", samples.mean(), x, se)
    ''',
    ["fraction=0.6，因此约 60% 的样本重建成 0.3，其余重建成 0.2。",
     "用均值的标准误判断蒙特卡洛偏差；不能要求有限次采样平均值精确等于 x。"],
    "截断到编码边界会破坏这里的无偏条件；随机数开销和训练收益需另外评估。", "第 01 章随机舍入期望推导")

add("gptq", "02", "OBQ / GPTQ：二阶误差补偿", "2.4",
    "量化一个权重后，把它造成的输出误差分配给尚未量化的自由权重。逆 Hessian 提供最优连续补偿方向；固定该变量后，还必须消去对应行列，保持下步自由变量问题一致。",
    ["H=2XᵀX，计算逆矩阵 M", "当前权重选择离散点 qi", "按 M 的首列补偿剩余权重", "消元后继续\n固定权重不再修改"],
    '''
    import numpy as np
    X = np.array([[1., .5], [0., np.sqrt(3)/2]])
    w = np.array([.6, .6]); current = w.copy()
    M = np.linalg.inv(2*X.T@X)
    for i in range(len(w)):
        qi = np.clip(np.rint(current[i]), 0, 1)
        current[i:] += (qi-current[i])*M[:, 0]/M[0, 0]
        M = M[1:, 1:] - np.outer(M[1:, 0], M[0, 1:])/M[0, 0]
    loss = lambda q: np.sum((X@(q-w))**2)
    assert np.allclose(current, [1, 0]) and np.isclose(loss(current), .28)
    print("compensated, loss, RTN loss:", current, loss(current), loss(np.rint(w)))
    ''',
    ["current[i:] 只修改尚未固定的后缀；补偿系数的符号使用 qi-current[i]，所以直接相加。",
     "Schur 消元得到下一次约束问题的逆 Hessian；不能反复使用最初的完整逆矩阵。",
     "原始 OBQ 还会按损失选择变量；GPTQ 共享列顺序并使用 Cholesky 与分块更新降低开销。"],
    "代码是固定顺序、单输出行的 OBQ/GPTQ 核心机制，没有实现完整 GPTQ 的块更新、分组尺度和权重打包。", "R01")

add("gptaq", "02", "GPTAQ / GPTQv2：非对称校准", "2.7",
    "学生层接收已经量化的上游输入 Xq，教师目标来自原输入 Xf，因此当前权重处已有非零重建残差。局部二次问题出现梯度项 g，需要同时修正已有偏差和当前舍入约束。",
    ["教师目标 yf=Xf*w", "学生残差 Xq*w-yf", "形成梯度 g 与 Hessian H", "带约束求解补偿 delta"],
    '''
    import numpy as np
    Xf = np.array([[1., .5], [0., 1.]])
    Xq = Xf + np.array([[.1, 0.], [0., -.1]])
    w = np.array([.6, .6]); e = np.array([1., 0.]); epsilon = .4
    H = 2*Xq.T@Xq
    g = 2*Xq.T@(Xq@w-Xf@w)
    M = np.linalg.inv(H)
    delta = -M@g + (epsilon + e@M@g)/(e@M@e)*(M@e)
    kkt = np.block([[H, e[:, None]], [e[None, :], np.zeros((1, 1))]])
    independent = np.linalg.solve(kkt, np.r_[-g, epsilon])[:2]
    assert np.allclose(delta, independent) and np.isclose(delta[0], epsilon)
    print("existing gradient, constrained correction:", g, delta)
    ''',
    ["g 来自真实的教师/学生输入差异，没有任意指定；H 则只由学生输入决定。",
     "第一项 -M@g 修正已有残差，第二项把选中权重锁到舍入误差 epsilon。独立 KKT 求解核对约束解。"],
    "这是一轮局部二次补偿，不是论文完整的可扩展矩阵更新流程；名称与版本对应关系见正文。", "R02")

add("damping", "02", "Damping：正则化二阶求解", "2.6",
    "相关输入使 Hessian 奇异或病态。在对角线上加入正数 λ 等价于惩罚过大的权重改动，能稳定求解，但会改变目标并削弱相关性信息。",
    ["完全相关的输入列", "H 存在零特征值", "加入 λI 使其正定", "求解受正则化的补偿方向"],
    '''
    import numpy as np
    X = np.array([[1., 1.], [2., 2.]])
    H = 2*X.T@X
    damping = .01*np.mean(np.diag(H))
    regularized = H + damping*np.eye(2)
    direction = np.linalg.solve(regularized, [1., 0.])
    assert np.linalg.matrix_rank(H) == 1
    assert np.linalg.eigvalsh(regularized).min() > 0
    assert np.allclose(regularized@direction, [1, 0])
    print("lambda, direction:", damping, direction)
    ''',
    ["原 H 的秩为 1，不能稳定地直接求逆；加入 λ 后所有特征值都严格为正。",
     "solve 计算所需的一列逆作用，不需要为这个例子显式形成全部逆矩阵。"],
    "全零 Hessian 的对角均值仍为零，需要另设 λ 下限；这里用非零奇异矩阵展示常用相对 damping。", "R01")

add("llm_int8", "03", "LLM.int8()：离群通道分流", "3.2",
    "将包含激活离群值的输入特征列拆到高精度支路，其余列进行向量级 INT8 量化。两条支路的矩阵乘相加，避免少量极端通道控制所有普通值的步长。",
    ["检测激活超过阈值的列", "普通列：INT8 QDQ 后矩阵乘", "离群列：高精度矩阵乘", "两支路求和得到输出"],
    '''
    import numpy as np
    from experiments.quant_lab import symmetric_qdq, mse
    X = np.array([[.1, 30., -.2], [.2, -25., .3]])
    W = np.array([[.4, -.2, .8], [.3, .1, -.5]])
    outlier = np.max(abs(X), axis=0) > 6
    normal = ~outlier
    x8 = symmetric_qdq(X[:, normal], bits=8, axis=1)[0]
    w8 = symmetric_qdq(W[:, normal], bits=8, axis=1)[0]
    output = x8@w8.T + X[:, outlier]@W[:, outlier].T
    exact_split = X[:, normal]@W[:, normal].T + X[:, outlier]@W[:, outlier].T
    assert np.allclose(exact_split, X@W.T)
    assert np.array_equal(outlier, [False, True, False])
    print("outlier columns, output MSE:", np.flatnonzero(outlier), mse(output, X@W.T))
    ''',
    ["axis=0 的检测在 token 维归约，得到需要分流的特征列；权重必须拆相同输入列。",
     "普通激活按 token、普通权重按输出行统计尺度，对应向量级量化。exact_split 检查分支拆分本身保持函数。"],
    "高精度支路在这里使用 float64 验证代数；真实实现常用 FP16，并需处理空支路及检测、索引开销。", "R03")

add("smoothquant", "03", "SmoothQuant：平衡激活与权重范围", "3.4",
    "选择通道尺度 s=a^α/b^(1-α)，把输入除以 s、权重乘以 s。提高 α 会均衡激活跨通道范围，而非保证各通道绝对幅度下降；目标是改善 W8A8 的联合误差。",
    ["统计激活 a 与权重 b", "计算 s=a^α/b^(1-α)", "等价变换 X/s 与 W*s", "两侧 W8A8，比较输出误差"],
    '''
    import numpy as np
    from experiments.quant_lab import symmetric_qdq, mse
    rng = np.random.default_rng(3)
    X = rng.normal(size=(64, 8)); X[:, 0] *= 20
    W = rng.normal(size=(4, 8))*.2
    a = np.maximum(np.max(abs(X), axis=0), 1e-8)
    b = np.maximum(np.max(abs(W), axis=0), 1e-8)
    for alpha in (0., .5, 1.):
        s = a**alpha / b**(1-alpha)
        assert np.allclose((X/s)@(W*s).T, X@W.T)
        output = symmetric_qdq(X/s, bits=8, axis=1)[0] @ symmetric_qdq(W*s, bits=8, axis=1)[0].T
        print("alpha, output MSE, activation max spread:", alpha, mse(X@W.T, output), np.max(a/s)/np.min(a/s))
    ''',
    ["s 的长度等于输入维 d，NumPy 广播同时作用于 X 的特征列和 W 的输入列。",
     "先验证浮点等价，再插入两侧量化；最后一项是通道最大幅度的比值，α=1 时应接近 1。"],
    "实际网络还需处理共享输入分支、归一化融合和校准范围漂移；示例并不寻找真实模型的最优 α。", "R04")

add("awq", "03", "AWQ：激活感知的权重缩放", "3.5",
    "用激活幅度识别重要输入方向，放大对应权重列再进行共享范围量化，以改善它们除回尺度后的有效分辨率。收益依赖分组共享 scale，过大缩放会牺牲其他通道。",
    ["统计平均绝对激活", "搜索通道尺度 a^α", "对 W*s 做分组量化", "以 X/s 的输出重建误差选优"],
    '''
    import numpy as np
    from experiments.quant_lab import group_qdq, mse
    rng = np.random.default_rng(3)
    X = rng.normal(size=(64, 8)); X[:, 0] *= 20
    W = rng.normal(size=(4, 8))*.2
    a = np.maximum(np.mean(abs(X), axis=0), 1e-8)
    alphas = np.linspace(0, 1, 21); errors = []
    for alpha in alphas:
        s = a**alpha; s /= np.sqrt(s.max()*s.min())
        reconstructed = (X/s)@group_qdq(W*s, group_size=4).T
        errors.append(mse(X@W.T, reconstructed))
    best = int(np.argmin(errors))
    assert errors[best] <= errors[0]
    print("selected alpha, best MSE, RTN MSE:", alphas[best], errors[best], errors[0])
    ''',
    ["α=0 时所有缩放为 1，因此候选集合含有原始 RTN 基线，搜索不会丢掉这个候选。",
     "激活始终保持高精度，只对 W*s 量化；这与 SmoothQuant 两侧都量化的目标不同。"],
    "未实现完整 AWQ 的裁剪搜索、Transformer 模块融合或设备内核；保存的是低比特主体，不是额外保留固定 1% FP16 权重。", "R05")

add("zeroquant", "03", "ZeroQuant：细粒度量化与逐层蒸馏", "3.7",
    "结合分组权重量化、per-token 激活量化、逐层蒸馏和融合内核。局部蒸馏只保存当前层的训练状态，以教师层输出指导低比特学生层适配。",
    ["教师层生成目标 XWᵀ", "激活逐 token、权重分组量化", "STE 更新当前层主权重", "导出后需融合量化与计算"],
    '''
    import numpy as np
    from experiments.quant_lab import group_qdq, symmetric_qdq, mse
    rng = np.random.default_rng(11)
    X = rng.normal(size=(32, 8)); W = rng.normal(size=(4, 8))
    target = X@W.T
    Xq = symmetric_qdq(X, bits=8, axis=1)[0]
    master = W.copy(); best = np.inf; initial = None
    for step in range(30):
        Wq = group_qdq(master, bits=4, group_size=4)
        error = Xq@Wq.T-target
        best = min(best, mse(target, Xq@Wq.T))
        if initial is None:
            initial = best
        gradient = 2*error.T@Xq/error.size  # STE for weight QDQ.
        if step == 0:
            # Differentiate the smooth reconstruction in Wq, not through round.
            direction = np.arange(W.size).reshape(W.shape)/W.size
            eps = 1e-5
            loss = lambda weights: mse(target, Xq@weights.T)
            numerical = (loss(Wq+eps*direction)-loss(Wq-eps*direction))/(2*eps)
            assert np.isclose(numerical, np.sum(gradient*direction), atol=1e-9)
            previous_master = master.copy()
        master -= .03*gradient
        if step == 0:
            assert loss(Wq+master-previous_master) < loss(Wq)
    assert np.linalg.norm(master-W) > 0
    assert best < initial  # Regression for this fixed example, not every QAT run.
    print("best local reconstruction MSE:", best)
    ''',
    ["target 固定为原始教师层输出；更新的仅是当前层 master，避免同时训练整模型。",
     "把 QDQ 的权重导数近似为 1 后，梯度等于普通线性层的梯度；有限差分核对的是 Wq 上的光滑重建目标，不是 round 的真实导数。",
     "首次参数更新必须降低平滑重建目标，以捕获方向写反；固定量化示例还检查最好损失的改善，但不要求每步下降。"],
    "用单层 NumPy 和 W4A8 展示局部蒸馏机制；不包含论文的完整配置、数据流程或 fused kernel，不能测出其系统加速。", "R06")

add("rptq", "03", "RPTQ：通道聚类重排", "3.8",
    "把范围相近的激活通道放在同一量化组，减少大、小范围通道共享步长造成的损失。置换只改变顺序，权重输入列必须同步置换以保持线性输出。",
    ["通道范围 1、100、2、80", "将相似范围放在同一组", "同时重排激活与权重列", "按新组独立量化"],
    '''
    import numpy as np
    from experiments.quant_lab import group_qdq, mse
    X = np.array([[.5, 100., 1., 80.], [-1., -50., -2., -40.]])
    W = np.array([[.2, .1, .3, -.1]])
    order = np.argsort(np.max(abs(X), axis=0))
    Xp, Wp = X[:, order], W[:, order]
    assert np.allclose(Xp@Wp.T, X@W.T)
    before = group_qdq(X, group_size=2)@W.T
    after = group_qdq(Xp, group_size=2)@Wp.T
    print("order, output MSE before/after:", order, mse(before, X@W.T), mse(after, X@W.T))
    ''',
    ["argsort 用范围排序演示重排动机；真实 RPTQ 使用通道范围特征聚类。",
     "这里把每行激活沿特征轴分成两组，仅示范不同通道被安排到同组后的量化效果。"],
    "实际方法还要把置换融合到归一化及权重布局；每层额外执行 gather 可能抵消收益。", "R07")

add("omni_lwc", "04", "OmniQuant LWC：可学习权重裁剪", "4.2",
    "用 sigmoid 参数化上下界的收缩比例，把量化范围变成待优化变量。主权重保持固定，以块输出重建误差选择合适的裁剪，而不只盯着权重误差。",
    ["固定主权重 W", "sigmoid(u/v) 收缩上下界", "计算仿射 scale 与 zero", "按输出重建目标优化范围"],
    '''
    import numpy as np
    from experiments.quant_lab import mse
    X = np.array([[1., .1, 2., 1.], [.5, -.2, 1., .3]])
    W = np.array([[-.3, -4., .2, 2.]])
    candidates = []
    for u in (-2., 0., 2.):
        for v in (-2., 0., 2.):
            lo = W.min()/(1+np.exp(-u))
            hi = W.max()/(1+np.exp(-v))
            scale = (hi-lo)/15; zero = np.clip(np.rint(-lo/scale), 0, 15)
            Wq = scale*(np.clip(np.rint(W/scale)+zero, 0, 15)-zero)
            candidates.append((mse(X@W.T, X@Wq.T), u, v))
    best = min(candidates)
    assert np.isfinite(best[0])
    assert best[1:] == (2., 2.)  # Independently recorded optimum for these inputs.
    assert best[0] < next(cost for cost, u, v in candidates if u == v == 0.)
    print("best output MSE, u, v:", best)
    ''',
    ["u/v 经 sigmoid 后在 (0,1)，保证负端点和正端点分别向零收缩。",
     "用小网格直接搜索两个变量，清楚展示 LWC 的可优化对象和目标；原论文通过 STE 梯度进行块级优化。",
     "本组输入的最优候选为 (2,2)，误差低于 (0,0)；断言核对选中的候选，避免选到最差项仍通过。"],
    "示例只处理同时含正负值的一组权重，使用枚举代替优化器；这不是完整 OmniQuant 校准实现。", "R08")

add("omni_let", "04", "OmniQuant LET：可学习平移与缩放", "4.3",
    "输入减中心再除通道尺度，权重对应乘尺度，并将 Wc 加到偏置。浮点输出严格不变，而量化前分布改变；校准时可联合优化这些平移和尺度。",
    ["选择通道中心 c 与尺度 s", "X′=(X-c)/s，W′=W*s", "偏置补偿 b′=b+Wc", "量化后按块重建目标优化"],
    '''
    import numpy as np
    from experiments.quant_lab import symmetric_qdq, mse
    X = np.array([[10., .2], [12., -.1], [9., .3]])
    W = np.array([[.4, .8], [-.2, .5]]); b = np.array([.1, -.3])
    c = X.mean(axis=0); s = np.array([2., .5])
    Xp, Wp, bp = (X-c)/s, W*s, b+W@c
    assert np.allclose(Xp@Wp.T+bp, X@W.T+b)
    approximate = symmetric_qdq(Xp, bits=4, axis=1)[0] @ symmetric_qdq(Wp, axis=1)[0].T + bp
    print("bias correction, quantized output MSE:", W@c, mse(approximate, X@W.T+b))
    ''',
    ["c 与 s 的长度为输入维，bp 长度为输出维；W@c 的方向不能写反。",
     "先用固定候选验证 LET 的代数基础，再插入量化观察误差。学习时把 c/s 交给块重建优化器即可。"],
    "这里只展示 LET 参数化及前向，不实现联合优化；无偏置模型、归一化与多分支融合须按架构处理。", "R08")

add("autoround", "04", "AutoRound / SignRound：优化舍入决策", "4.4",
    "允许舍入偏移改变某个权重向上或向下的选择，使联合输出更接近教师输出。符号梯度控制每次更新方向，部署时只导出选定的量化编码。",
    ["权重 0.49、0.46", "前向 round(w+v)", "用输出误差计算 STE 梯度", "符号更新 v 并保留最好编码"],
    '''
    import numpy as np
    w = np.array([.49, .46]); v = np.zeros(2)
    target = w.sum(); best_loss = np.inf; best_q = None
    for step in range(20):
        q = np.clip(np.rint(w+v), 0, 1)
        error = q.sum()-target
        if error**2 < best_loss:
            best_loss, best_q = error**2, q.copy()
        gradient = np.full(2, 2*error)  # Identity STE in the active range.
        v -= .02*np.sign(gradient)
    assert best_loss < (np.rint(w).sum()-target)**2
    print("best codes, output squared error:", best_q, best_loss)
    ''',
    ["两个权重的初始 RTN 都为零，但输出目标为 0.95。联合选择 [1,0] 可以大幅减小输出误差。",
     "梯度通过 round 的恒等替代导数传到 v；硬舍入可能来回跳，因此必须保留最优已见状态。"],
    "仅演示固定尺度下的舍入偏移与 SignSGD；完整 AutoRound 还涉及裁剪、块重建、版本相关的优化策略。", "R09")

add("hqq", "04", "HQQ：半二次分裂与浮点 offset", "4.5",
    "在无需激活的权重重建中，引入辅助残差，把稳健误差与二次问题拆开。p=1 时残差子问题有精确软阈值解，随后用二次项的驻点更新浮点 offset。",
    ["固定步长，重算整数编码", "残差 r=w-w_hat", "p=1：软阈值得到 e", "均值公式更新浮点 z"],
    '''
    import numpy as np
    w = np.array([-.3, -.1, .06, .5])
    s, z, beta = .1, 5., 10.
    for step in range(5):
        q = np.clip(np.rint(w/s+z), 0, 15)
        r = w-s*(q-z)
        e = np.sign(r)*np.maximum(abs(r)-1/beta, 0)
        z = np.mean(q-(w-e)/s)
        assert abs(np.sum(w-s*(q-z)-e)) < 1e-12
        beta *= 1.2
    print("floating offset, final reconstruction:", z, s*(q-z))
    ''',
    ["e 的软阈值是 p=1 子问题的解析解；更新 z 后二次残差之和为零，断言直接检查驻点条件。",
     "每轮重新从原 w 算编码，和作者主要 proximal 路径相同的更新结构；作者代码的 scale 常表示这里 1/s。"],
    "固定 s、p=1 的教学路径，没有作者实现的完整提前停止逻辑；p<1 的收缩通常是近似，z 也不是整数 zero point。", "R10")

add("quarot", "05", "QuaRot：合法位置的配对旋转", "5.4",
    "正交旋转保持长度和内积，却能改变坐标峰值。QuaRot 将残差流和内部激活在合法位置协同旋转，再量化；可融合的变换离线进入权重，其余保留在线 Hadamard。",
    ["融合 RMSNorm 通道增益", "残差流采用一致坐标系", "X→XR，W→WR", "合法位置量化并保留必要在线旋转"],
    '''
    import numpy as np
    from experiments.quant_lab import hadamard, symmetric_qdq, mse
    X = np.array([[12., .1, -.2, .3], [4., -.3, .1, -.2]])
    W = np.array([[.2, .5, -.1, .3]])
    R = hadamard(4)
    rms = lambda x: x/np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+1e-6)
    assert np.allclose((X@R)@(W@R).T, X@W.T)
    assert np.allclose(rms(X@R), rms(X)@R)
    approximate = symmetric_qdq(X@R, axis=1)[0] @ symmetric_qdq(W@R, axis=1)[0].T
    print("peaks before/after:", np.max(abs(X)), np.max(abs(X@R)))
    print("rotated W4A4 output MSE:", mse(X@W.T, approximate))
    ''',
    ["R=H/√d 保证 RᵀR=I；两个线性输入侧同时右乘 R，乘积保持不变。",
     "rms 不含逐通道增益，所以旋转可以穿过它；带增益的 RMSNorm 必须先处理参数融合。"],
    "只验证一个线性层与无增益 RMSNorm；整个 Transformer 还要处理残差、非线性、RoPE 和在线旋转位置。", "R11")

add("spinquant", "05", "SpinQuant：学习正交旋转", "5.5",
    "把旋转矩阵作为参数，在带量化的前向中优化损失，并用正交约束保持未量化模型的等价性。Cayley 更新使用反对称矩阵，使更新后的旋转仍然正交。",
    ["量化前向：Q(XR)Q(WR)ᵀ", "STE 得到旋转梯度 G", "A=GRᵀ-RGᵀ", "Cayley 更新，继续优化 R"],
    '''
    import numpy as np
    from experiments.quant_lab import symmetric_qdq
    rng = np.random.default_rng(12)
    X = rng.normal(size=(16, 4)); X[:, 0] *= 5
    W = rng.normal(size=(3, 4)); R = np.eye(4); losses = []
    for step in range(20):
        Xq = symmetric_qdq(X@R, axis=1)[0]
        Wq = symmetric_qdq(W@R, axis=1)[0]
        E = Xq@Wq.T-X@W.T
        losses.append(np.mean(E**2))
        G = 2*(X.T@(E@Wq) + W.T@(E.T@Xq))/E.size
        A = G@R.T-R@G.T
        R = np.linalg.solve(np.eye(4)+.005*A, (np.eye(4)-.005*A)@R)
        assert np.allclose(R.T@R, np.eye(4), atol=1e-10)
    print("initial/best local loss:", losses[0], min(losses))
    ''',
    ["梯度经过两个量化分支；本例对 round 使用恒等 STE，并将当前尺度视为常量。",
     "Aᵀ=-A，Cayley 的左右因子构成正交映射。每步断言验证几何约束，而不要求硬量化损失逐步下降。"],
    "用单层输出重建展示学习旋转；原始 SpinQuant 的模型级损失、旋转位置、动量和优化预算不在此例中。", "R12")

add("flatquant", "05", "FlatQuant：结构化可逆变换", "5.6",
    "一般可逆变换可同时调整方向和尺度，配对关系是 XT 与 WT⁻ᵀ。Kronecker 结构把一个大矩阵作用拆成两次小矩阵作用，降低在线变换成本。",
    ["选择小矩阵 A、B", "T=A⊗B", "输入 XT，权重 WT⁻ᵀ", "联合校准变换与量化范围"],
    '''
    import numpy as np
    from experiments.quant_lab import symmetric_qdq, mse
    rng = np.random.default_rng(13)
    X = rng.normal(size=(8, 4)); W = rng.normal(size=(3, 4))
    A = np.array([[1., .2], [0., .8]])
    B = np.array([[.8, 0.], [.1, 1.2]])
    T = np.kron(A, B)
    transformed = np.einsum('nij,ia,jb->nab', X.reshape(-1, 2, 2), A, B).reshape(X.shape)
    Wp = np.linalg.solve(T, W.T).T
    assert np.allclose(transformed, X@T)
    assert np.allclose(transformed@Wp.T, X@W.T)
    approximate = symmetric_qdq(transformed, axis=1)[0] @ symmetric_qdq(Wp, axis=1)[0].T
    print("condition number, QDQ MSE:", np.linalg.cond(T), mse(approximate, X@W.T))
    ''',
    ["einsum 对 reshape 后的两个特征轴分别作用 A 和 B，与显式 Kronecker 矩阵乘的结果一致。",
     "solve(T,W.T).T 实现 WT⁻ᵀ，不能直接写成 W@T；条件数展示一般可逆变换的数值风险。"],
    "给定一组变换验证结构和前向，没有复现 FlatQuant 对变换、裁剪及多模块位置的完整学习过程。", "R13")

add("quip", "05", "QuIP：非相干化与 LDLQ", "5.7",
    "随机预处理分散权重与 Hessian 的敏感方向，再用 LDL 分解形成误差反馈。采用 H=LDLᵀ 时，从后向前量化，已知后缀误差用于修正当前输入到量化器的值。",
    ["随机正交预处理 W 与 H", "分解 H=LDLᵀ", "从后向前反馈已知误差", "当前值选择最近格点"],
    '''
    import numpy as np
    from experiments.technique_kernels import ldl_feedback
    H = np.array([[2., 1.], [1., 2.]])
    w = np.array([.6, .6]); grid = np.array([0., 1.])
    quantized, L, D = ldl_feedback(w, H, grid)
    e = quantized-w
    assert np.allclose(L@np.diag(D)@L.T, H)
    assert np.isclose(e@H@e, np.sum(D*(L.T@e)**2))
    assert np.array_equal(quantized, [0., 1.])
    print("reverse-order LDLQ, quadratic loss:", quantized, e@H@e)
    ''',
    ["helper 从 Cholesky 因子提取单位下三角 L 和正对角 D，随后执行正文的反向反馈公式。",
     "结果与正向 GPTQ 的 [1,0] 不同，但本例对称，损失相同。遍历方向、分解形式和反馈方向必须配套。"],
    "只验证 LDLQ 子步骤；完整 QuIP 还包括非相干预处理、有限网格范围与理论假设，不能由本例宣称 2 bit 任务质量保证。", "R14")

add("quip_sharp", "06", "QuIP#：Hadamard 与 E8 格点子问题", "6.5",
    "QuIP# 将随机 Hadamard 非相干化、块级 LDLQ 和有限 E8P 向量码本结合。八维联合编码能利用空间几何；理解有限码本之前，可先学习 E8 格的最近点规则。",
    ["随机 Hadamard 分散坐标", "八维向量交给码本", "比较偶数和整数点与半整数点", "完整方法用有限 E8P 与块反馈"],
    '''
    import numpy as np
    from experiments.quant_lab import hadamard
    from experiments.technique_kernels import nearest_e8
    x = np.array([1.2, .1, -.3, .4, .8, -.1, .2, .5])
    signs = np.diag([1., -1., 1., 1., -1., 1., -1., 1.])
    R = signs@hadamard(8)
    rotated = x@R
    lattice_point = nearest_e8(rotated)
    restored = lattice_point@R.T
    fractions = np.mod(lattice_point, 1)
    assert np.allclose(fractions, 0) or np.allclose(fractions, .5)
    assert int(round(lattice_point.sum())) % 2 == 0
    assert np.isclose(np.sum((x-restored)**2), np.sum((rotated-lattice_point)**2))
    print("E8 point, reconstruction error:", lattice_point, np.linalg.norm(x-restored))
    ''',
    ["nearest_e8 分别找最近的偶数和整数向量，以及平移半格后的候选，再比较距离；奇偶不符时修正代价最小的坐标。",
     "反向乘 R.T 恢复原坐标。正交变换保持平方误差，因此旋转域和原域误差相同。"],
    "代码量化到无限 E8 格，不能把结果直接宣称为 16 bit/向量或完整 QuIP#；有限 E8P 截断、索引打包和 BlockLDLQ 需按 R15 实现。", "R15")

add("lloyd_max", "06", "Lloyd-Max / k-means：学习标量码本", "6.2",
    "固定码本时选最近中心；固定归属时用簇内加权均值更新中心。这两个子步骤交替降低相同的重建目标；敏感性权重越大，中心越向该元素靠近。",
    ["初始化代表值", "每个元素选择最近中心", "按重要性更新簇内均值", "重复到目标稳定"],
    '''
    import numpy as np
    from experiments.technique_kernels import lloyd, nearest
    values = np.array([0., 1., 3., 4.])
    importance = np.array([9., 1., 1., 1.])
    centers, losses = lloyd(values, 2, importance)
    assert np.all(np.diff(losses) <= 1e-10)
    assert np.allclose(centers, [.1, 3.5])
    print("centers, reconstructed values:", centers, nearest(values, centers))
    ''',
    ["第一个簇的最优中心为 (9*0+1*1)/10=0.1；普通均值 0.5 会忽略敏感性差异。",
     "固定归属的最优更新和最近中心重新分配都不会增加同一目标，但只能保证局部收敛。"],
    "这是有限样本的加权一维 k-means；连续分布的 Lloyd-Max 用区间条件均值，初始化也会影响局部解。", "第 06 章推导，R17")

add("squeezellm", "06", "SqueezeLLM：加权非均匀主体与稀疏例外", "6.3",
    "用任务敏感性加权学习非均匀码本，并把少数极端或敏感元素单独保留。主体与稀疏项必须在提取位置正确互补，避免高精度元素被重复计算。",
    ["估计逐元素敏感性", "提取极端值为稀疏项", "主体使用加权码本", "主体与稀疏输出合并"],
    '''
    import numpy as np
    from experiments.technique_kernels import lloyd, nearest
    W = np.array([[.1, -.2, .3, 8.], [-.3, .2, .4, -.1]])
    importance = np.array([[9., 1., 2., 1.], [1., 3., 1., 1.]])
    protected = abs(W) > 3
    centers, _ = lloyd(W[~protected], 4, importance[~protected])
    dense = nearest(W, centers); dense[protected] = 0
    sparse = np.where(protected, W, 0)
    reconstructed = dense+sparse
    assert np.array_equal(reconstructed[protected], W[protected])
    print("centers, protected count:", centers, protected.sum())
    print("weighted error:", np.sum(importance*(reconstructed-W)**2))
    ''',
    ["importance 在这里是给定的正权重；实际 SqueezeLLM 通过任务梯度统计近似 Fisher 对角敏感性。",
     "对保护位置强制 dense=0，然后 sparse 保存原值，保证合并后恰好恢复，避免码本中无零时的重复贡献。"],
    "示例省略 Fisher 采集、稀疏索引编码和专用查表内核；稀疏项在 NumPy 中仍以稠密数组表示。", "R17")

add("aqlm", "06", "AQLM：多个向量码本相加", "6.6",
    "每组权重由多个码本向量相加表示，用多个小索引代替一个巨大码本。索引选择受到输入相关性的影响，因此常以层输出重建误差搜索，而不只用向量距离。",
    ["权重分成短向量", "每个码本选一个索引", "将代表向量相加", "按输出误差搜索并学习码本"],
    '''
    import itertools
    import numpy as np
    C1 = np.array([[1., 0.], [0., 1.]])
    C2 = np.array([[.2, .2], [-.2, -.2]])
    w = np.array([1.15, .18]); X = np.array([[1., .5], [.2, 1.]])
    candidates = []
    for i, j in itertools.product(range(2), repeat=2):
        reconstructed = C1[i]+C2[j]
        candidates.append((np.sum((X@(reconstructed-w))**2), i, j))
    loss, i, j = min(candidates)
    assert (i, j) == (0, 0)
    print("indices, reconstruction, output error:", (i, j), C1[i]+C2[j], loss)
    print("index bits per weight:", 2*np.log2(len(C1))/len(w))
    ''',
    ["两个索引决定一个和向量，原型总数是码本大小之和，可能的组合数却是乘积。",
     "目标函数用 X 放大量化误差，保留了校准输入相关性。真实规模用 beam search 等方法代替穷举。"],
    "此例固定码本展示离散搜索；完整 AQLM 还会学习码本、行尺度并进行块级联合优化。", "R16")

add("spqr", "06", "SpQR：敏感离群值与二级元数据量化", "6.7",
    "用二阶受约束误差识别难以量化的权重，单独保留高精度残差。小组 scale/zero point 也需要再量化，否则元数据可能吞掉低比特收益。",
    ["分组量化形成候选主体", "按误差² / 逆H对角评分", "高分元素保存稀疏残差", "组尺度等元数据再量化"],
    '''
    import numpy as np
    from experiments.quant_lab import symmetric_qdq
    W = np.array([[.1, .3, .6, 3.], [-.2, .7, 1.1, 2.]])
    blocks = W.reshape(2, 2, 2)
    _, codes, scales = symmetric_qdq(blocks, axis=-1)
    compressed_scales = symmetric_qdq(scales, bits=8)[0]
    dense = (codes*compressed_scales).reshape(W.shape)
    M = np.linalg.inv(np.array([[2., .2, 0, 0], [.2, 1., 0, 0], [0, 0, 3., .1], [0, 0, .1, 1.]]))
    scores = (W-dense)**2 / np.diag(M)[None, :]
    protected = np.zeros(W.size, dtype=bool); protected[np.argmax(scores)] = True
    protected = protected.reshape(W.shape)
    sparse = np.where(protected, W-dense, 0)
    assert np.allclose((dense+sparse)[protected], W[protected])
    print("protected location, scale quantization error:", np.argwhere(protected), np.max(abs(scales-compressed_scales)))
    ''',
    ["先二次量化尺度，再形成真实的 dense，稀疏项保存 W-dense，避免忽略元数据误差。",
     "评分分母用逆 Hessian 对角，和普通权重绝对值排序不同；这里选一个最高分元素解释机制。"],
    "没有实现完整顺序补偿及 SpQR 的具体元数据布局；二级尺度使用教学 INT8，不能作为官方 codec 导出。", "R18")

add("owq", "06", "OWQ：高精度 weak columns", "6.8",
    "以输入列为单位保护对输出特别敏感的权重。规则的列结构更容易拆成两条矩阵乘路径，但一整列保留高精度也可能比逐元素保护花更多空间。",
    ["统计输入通道二阶矩", "结合该列量化误差打分", "高分列保留高精度", "低比特与高精度分支相加"],
    '''
    import numpy as np
    from experiments.quant_lab import symmetric_qdq, mse
    X = np.array([[1., 30., .2], [.5, -20., .3]])
    W = np.array([[.1, .21, 1.], [.3, -.26, .8]])
    Wq = symmetric_qdq(W, bits=3, axis=1)[0]
    scores = np.sum(X**2, axis=0)*np.sum((W-Wq)**2, axis=0)
    keep = np.zeros(3, dtype=bool); keep[np.argmax(scores)] = True
    output = X[:, ~keep]@Wq[:, ~keep].T + X[:, keep]@W[:, keep].T
    combined = Wq.copy(); combined[:, keep] = W[:, keep]
    assert np.allclose(output, X@combined.T)
    print("protected column, output MSE:", np.flatnonzero(keep), mse(output, X@W.T))
    ''',
    ["输入二阶矩大且量化残差大的列更值得保护；scores 是正文的对角敏感性代理。",
     "通过两支路输出和合并权重输出一致性，核对输入列拆分方向。"],
    "此评分是教学对角近似，不替代 OWQ 原实现的完整排序、预算分配与量化补偿。", "R19")

add("mixed_precision", "06", "层级与组级混合精度：在预算内分配 bit", "6.10",
    "在总存储预算下，为不同层或同一层的不同权重组选择位宽。比较候选格式的失真和完整存储成本，把更多预算用于敏感位置；最终还要验证完整层输出及内核支持。",
    ["分配单位：层或组", "测量候选位宽的误差", "统计权重和元数据成本", "预算内选择\n验证完整输出"],
    '''
    import itertools
    import numpy as np
    parameters = np.array([8, 16]); sensitivity = np.array([20., 1.])
    budget = 96
    feasible = []
    for bits in itertools.product([2, 4, 8], repeat=2):
        b = np.array(bits)
        cost = int(parameters@b)
        if cost <= budget:
            error = np.sum(sensitivity*2.**(-2*b))
            feasible.append((error, bits, cost))
    error, bits, cost = min(feasible)
    assert cost <= budget and bits == (8, 2)
    print("selected bits, payload budget, proxy error:", bits, cost, error)

    # One layer, two groups: measure candidate errors, including FP16 scales.
    from experiments.quant_lab import symmetric_qdq
    groups = np.tile([[.13, .21, .77, 1.], [.01, -.02, .03, .04]], (1, 8))
    group_size = groups.shape[1]
    X = np.diag(np.repeat([4., 1.], group_size))
    choices = (2, 4, 8); decoded = {}; distortion = {}
    for g, group in enumerate(groups):
        columns = slice(g*group_size, (g+1)*group_size)
        for b in choices:
            _, codes, scale = symmetric_qdq(group, bits=b)
            decoded[g, b] = codes*scale.astype(np.float16).astype(np.float64)
            distortion[g, b] = np.mean((X[:, columns]@(decoded[g, b]-group))**2)
    metadata_bits = 16+8  # FP16 scale and one byte specifying the format.
    group_budget = 368
    candidates = []
    for allocation in itertools.product(choices, repeat=len(groups)):
        stored_bits = sum(group_size*b+metadata_bits for b in allocation)
        if stored_bits <= group_budget:
            loss = sum(distortion[g, b] for g, b in enumerate(allocation))
            candidates.append((loss, allocation, stored_bits))
    loss, allocation, stored_bits = min(candidates)
    restored = np.concatenate([decoded[g, b] for g, b in enumerate(allocation)])
    full_layer_mse = np.mean((X@(restored-groups.ravel()))**2)
    uniform4_mse = sum(distortion[g, 4] for g in range(len(groups)))
    assert allocation == (8, 2) and stored_bits == group_budget
    assert np.isclose(loss, full_layer_mse) and loss < uniform4_mse
    print("group bits, stored bits, bytes:", allocation, stored_bits, stored_bits/8)
    print("full-layer MSE, uniform-4 MSE:", full_layer_mse, uniform4_mse)
    ''',
    ["第一段是按层分配的解析代理例子：第二层参数量更大，但敏感系数低，所以在 96 bit 的纯权重预算内选择 8/2 bit。",
     "第二段的 groups 来自同一层，shape 为 [2,32]；每组分别试 2/4/8 bit，并用校准输入计算该组造成的输出 MSE。",
     "组级成本计入 FP16 scale 和一字节格式标签，且用转成 FP16 后的实际尺度解码，避免只算元数据成本却忽略尺度舍入。",
     "穷举在 368 bit 预算内选出 8/2 bit；第一组获得更高精度，第二组节省预算。不同组的尺度独立、位宽也不同。",
     "X 的对角结构保证跨组误差不耦合，因此可以将分组失真之和与完整层 MSE 独立核对；一般校准输入下未必相等。"],
    "层级例子使用高分辨率误差代理且只计 payload；组级例子显式计入所声明的元数据，但未物理打包或实现混合位宽 kernel。真实部署还要计入实际对齐和调度成本，并核对后端支持。", "第 06 章混合精度推导")

add("qat_ste", "07", "QAT / STE：高精度累积、低精度前向", "7.2",
    "前向使用量化权重，反向用人为指定的替代梯度更新高精度主权重。主权重能够累积小于量化步长的更新，避免每一步都把小更新舍掉。",
    ["高精度 master 权重", "前向舍入为低比特值", "STE 传递区间内梯度", "更新 master，下一步重新量化"],
    '''
    import numpy as np
    target = np.array([.26, -.37]); master = np.zeros(2)
    s = .1; losses = []
    for step in range(30):
        q = np.clip(np.rint(master/s), -7, 7)
        reconstructed = s*q
        losses.append(np.mean((reconstructed-target)**2))
        active = abs(master/s) < 7
        gradient = (reconstructed-target)*active  # Mean-square gradient for N=2.
        master -= .2*gradient
    assert min(losses) < losses[0]
    assert np.any(abs(master/s-np.rint(master/s)) > 1e-6)
    print("master, forward weights, best loss:", master, reconstructed, min(losses))
    ''',
    ["优化器只更新 master；q 是当前步的离散前向状态，下次重新计算。",
     "active 实现裁剪区间内的 STE。主权重不是网格点，这说明它确实保留了尚不足以跨格的小更新。"],
    "仅训练一个权重向量的平方误差；真实 QAT 还需管理激活、任务损失、梯度和优化器状态，训练内存不会变成纯 4 bit。", "第 07 章 STE 推导")

add("lsq", "07", "LSQ：学习量化步长", "7.3",
    "量化步长既影响外部的乘法尺度，也影响内部归一化值 w/s。用 STE 对两条路径求导，未裁剪区间得到 round(w/s)-w/s，而非真实阶梯函数的导数。",
    ["归一化 v=w/s", "量化输出 s*round(v)", "两条尺度路径合并求导", "按共享元素数缩放梯度"],
    '''
    import numpy as np
    w = np.array([.26, -.91, .8]); s = .1
    v = w/s; q = np.clip(np.rint(v), -7, 7)
    derivative = np.where(v < -7, -7., np.where(v > 7, 7., q-v))
    loss_gradient = np.array([1., .2, -.1])
    scaled_gradient = (loss_gradient@derivative)/np.sqrt(w.size*7)
    w0, s0 = .26, .1; q0, v0 = 3., 2.6
    surrogate = lambda t: t*(q0+w0/t-v0)
    eps = 1e-6
    numerical = (surrogate(s0+eps)-surrogate(s0-eps))/(2*eps)
    assert np.isclose(numerical, .4)
    print("STE derivatives, scaled loss gradient:", derivative, scaled_gradient)
    ''',
    ["区间外输出为 s*qmin 或 s*qmax，所以导数为边界编码；区间内才使用 q-v。",
     "有限差分作用于明确构造的 surrogate，不能对真实 round 函数做差分后宣称验证 STE。"],
    "演示 LSQ 的步长替代梯度，未构建训练器；初始化和梯度缩放应按权重/激活的实际设置选择。", "R21")

add("pact", "07", "PACT：学习非负激活裁剪阈值", "7.4",
    "将激活上界 α 作为可训练参数。增大它会保留更多尾部，但也增大量化步长；反向梯度把裁剪和分辨率的折中传回 α。",
    ["激活截到 [0,α]", "步长 s=α/(2^b-1)", "量化并计算输出损失", "更新正值阈值 α"],
    '''
    import numpy as np
    x = np.array([-.2, .2, .8, 2.4]); target = np.maximum(x, 0)
    alpha = 1.; levels = 15
    for step in range(20):
        s = alpha/levels
        clipped = np.clip(x, 0, alpha)
        q = np.rint(clipped/s); output = s*q
        derivative = np.where(x > alpha, 1., np.where(x < 0, 0., q/levels-x/alpha))
        gradient = np.mean(2*(output-target)*derivative)
        alpha = max(alpha-.1*gradient, 1e-6)
    assert alpha > 1. and output[0] == 0
    print("learned alpha, quantized nonnegative output:", alpha, output)
    ''',
    ["x>α 时输出饱和在 α，导数为 1；负输入经过非负裁剪，导数为零。",
     "区间内的 q/levels-x/α 来自步长与归一化两条路径的 STE；这是完整量化前向的一种明确梯度约定。"],
    "本例使用完整步长 STE 展示可学习裁剪，原 PACT 的具体梯度和正则化配置需查 R20；有正负值的 LLM 激活不能直接套单边裁剪。", "R20")

add("llm_qat", "07", "LLM-QAT：教师分布蒸馏", "7.5",
    "使用原模型生成的文本和软概率目标训练量化学生，不要求拿到原始预训练集。蒸馏保留整份词表分布的信息，比只匹配教师 argmax 更丰富。",
    ["教师生成训练文本", "保存教师软概率分布", "学生执行量化前向", "以 KL 损失和 STE 更新学生"],
    '''
    import numpy as np
    from experiments.quant_lab import softmax, symmetric_qdq
    rng = np.random.default_rng(14)
    X = rng.normal(size=(12, 4)); teacher = rng.normal(size=(5, 4))
    target = softmax(X@teacher.T)
    master = teacher + .1*rng.normal(size=teacher.shape)
    losses = []
    for step in range(30):
        Wq = symmetric_qdq(master, bits=4, axis=1)[0]
        probabilities = softmax(X@Wq.T)
        losses.append(np.mean(np.sum(target*np.log(target/probabilities), axis=1)))
        gradient = (probabilities-target).T@X/len(X)
        if step == 0:
            direction = np.arange(master.size).reshape(master.shape)/master.size
            loss = lambda weights: np.mean(np.sum(target*np.log(target/softmax(X@weights.T)), axis=1))
            eps = 1e-5
            numerical = (loss(Wq+eps*direction)-loss(Wq-eps*direction))/(2*eps)
            assert np.isclose(numerical, np.sum(gradient*direction), atol=1e-9)
        master -= .1*gradient
    assert min(losses) >= -1e-12
    assert np.allclose(target.sum(axis=1), 1)
    assert min(losses) < losses[0]  # Expected only for this fixed teaching fixture.
    print("initial/best teacher-student KL:", losses[0], min(losses))
    ''',
    ["target 固定为教师概率；KL 对学生 logits 的梯度是 probabilities-target，因此可以手写小线性分类器的蒸馏更新。",
     "学生前向插入 QDQ，反向通过恒等 STE 更新 master，教师参数始终不变。",
     "有限差分核对 Wq 上的平滑 KL 梯度；固定数据还检查最好 KL 的改善，删除更新或反向更新都会失败。"],
    "用随机特征代替文本和 Transformer，只演示蒸馏更新；没有实现文本生成以及权重、激活、KV 的完整联合 QAT。", "R22")

add("efficientqat", "07", "EfficientQAT：Block-AP 与 E2E-QP", "7.6",
    "先逐块更新较多参数以适应量化，再固定量化权重编码，在端到端模型中主要训练步长。第二阶段用较少可训练参数协调层间累计误差。",
    ["Block-AP：主权重与尺度适配", "冻结得到的离散权重编码", "E2E-QP：调整量化步长", "少量参数修正全局输出偏差"],
    '''
    import numpy as np
    X = np.array([[1., .2], [.3, 1.], [1., -.5]])
    target = X@np.array([.23, -.66]); master = np.array([.23, -.66]); s = .2
    for step in range(20):  # Toy Block-AP: update both master weights and scale.
        q = np.clip(np.rint(master/s), -7, 7)
        gradient = 2*X.T@(X@(s*q)-target)/len(X)
        ds = gradient@(q-master/s)
        master -= .03*gradient; s = max(s-.01*ds, 1e-4)
    frozen_codes = np.clip(np.rint(master/s), -7, 7).copy()
    before = np.mean((X@(s*frozen_codes)-target)**2)
    for step in range(40):  # Toy E2E-QP: only s is updated.
        direction = X@frozen_codes
        ds = 2*direction@(s*direction-target)/len(X)
        s = max(s-.01*ds, 1e-4)
    after = np.mean((X@(s*frozen_codes)-target)**2)
    assert after <= before + 1e-12
    print("fixed codes, final scale, before/after MSE:", frozen_codes, s, before, after)
    ''',
    ["第一阶段更新 master 和 s；第二阶段把 q 固定，梯度只经过乘法 s*q。",
     "主体编码固定不代表反量化权重数值完全不变：尺度变化仍会改变 s*q，这正是第二阶段的自由度。"],
    "单层玩具例子展示两阶段的可训练对象，无法体现真实逐块训练节省内存或跨 Transformer 层的端到端协调。", "R23")

add("lora", "07", "LoRA：低秩增量", "7.7",
    "用 BA 表示可训练权重增量，秩 r 远小于输入和输出维。前向可把低秩分支拆成两次小矩阵乘，只训练 A/B，从而减少训练参数和优化器状态。",
    ["输入 X 分成两条支路", "基础路径：XWᵀ", "低秩路径：(XAᵀ)Bᵀ", "两路相加，仅更新 A/B"],
    '''
    import numpy as np
    rng = np.random.default_rng(15)
    X = rng.normal(size=(5, 8)); W = rng.normal(size=(6, 8))
    A = rng.normal(size=(2, 8)); B = rng.normal(size=(6, 2))
    adapter_scale = 4/2
    separate = X@W.T + adapter_scale*(X@A.T)@B.T
    merged = X@(W+adapter_scale*B@A).T
    assert np.allclose(separate, merged)
    assert A.size+B.size < W.size
    print("base and trainable parameters:", W.size, A.size+B.size)
    ''',
    ["X@A.T 把 8 维映射到秩 2，再乘 B.T 输出 6 维；无需在每步显式形成完整 BA。",
     "adapter_scale 展示常见 α/r 缩放；初始化和合并时必须一并计算这个系数。"],
    "代码验证线性合并关系，不包含 dropout 或多模块训练；把增量合入量化基础权重后通常还需重新量化。", "第 07 章 LoRA 推导，R24")

add("qlora", "07", "QLoRA：冻结 NF4 基础模型，训练 LoRA", "7.8",
    "基础权重压缩保存、计算时反量化，训练只更新高精度低秩适配器。梯度仍可通过基础计算路径传递到适配器，但基础权重和编码保持冻结。",
    ["输入 X；基础 NF4 编码冻结", "基础路径：反量化 W 后相乘", "适配路径：(XAᵀ)Bᵀ", "输出求和，仅更新 A/B"],
    '''
    import numpy as np
    from experiments.quant_lab import NF4, codebook_qdq
    rng = np.random.default_rng(16)
    X = rng.normal(size=(16, 8)); W = rng.normal(size=(4, 8))
    base = codebook_qdq(W, NF4, group_size=8); snapshot = base.copy()
    A = rng.normal(size=(2, 8))*.1; B = np.zeros((4, 2))
    target = X@W.T; losses = []
    for step in range(60):
        low = X@A.T
        error = X@base.T+low@B.T-target
        losses.append(np.mean(error**2))
        grad_B = 2*error.T@low/error.size
        grad_A = 2*(error@B).T@X/error.size
        A -= .2*grad_A; B -= .2*grad_B
    assert np.array_equal(base, snapshot) and np.linalg.norm(B) > 0
    assert min(losses) < losses[0]
    print("frozen-base adapter loss before/best:", losses[0], min(losses))
    ''',
    ["B 从零开始，因此初始 adapter 不改变基础模型输出；第一步先更新 B，之后梯度也进入 A。",
     "两个梯度在参数更新前同时计算，避免把新 B 与旧 A 混在同一次梯度中。基础权重快照验证冻结状态。"],
    "为便于 NumPy 检验，base 是常驻 float64 重建数组，不能据此测量 QLoRA 显存；实际实现保留打包基础权重并按算子反量化。", "R24")

add("nf4", "07", "NF4：非均匀 16 点码本", "7.9",
    "NF4 在零附近放置更密的代表值，每组权重用 absmax 归一化后存最近代表值的索引。16 个编码包括零，但并非等间隔，也不具有 FP4 的指数位语义。",
    ["按组计算 absmax", "权重归一化到 [-1,1]", "查找最近 NF4 代表值", "存索引，解码时乘回 absmax"],
    '''
    import numpy as np
    from experiments.quant_lab import NF4
    w = np.array([[-1., -.3, 0., .1, .7, 2.]])
    scale = np.max(abs(w), axis=1, keepdims=True)
    indices = np.argmin(abs(w[..., None]/scale[..., None]-NF4), axis=-1)
    restored = scale*NF4[indices]
    assert np.array_equal(indices[:, 2], [7]) and restored[0, 2] == 0
    assert indices.min() >= 0 and indices.max() < 16
    print("indices, decoded weights:", indices, restored)
    ''',
    ["多增加一维后同时比较每个归一化权重与 16 个代表值的距离，argmin 返回可保存的索引。",
     "索引 7 对应精确零；必须查 NF4 表才能解释编码，不能直接把索引当作有符号整数权重。"],
    "此例不做物理 nibble 打包。NF4 与 15 级对称 INT4、16 级等间隔及仿射格式的完整对照见基础实验 08。", "R24")

add("double_quant", "07", "Double Quantization：压缩量化尺度", "7.10",
    "低比特权重每组都要保存尺度，组越小元数据比例越高。再将第一层尺度用 8 bit 编码，并为一批尺度保存第二层尺度，就能摊薄元数据成本。",
    ["权重分组得到一级尺度", "尺度减去公共中心", "一级尺度用 8 bit 编码", "二级尺度与中心恢复一级尺度"],
    '''
    import numpy as np
    rng = np.random.default_rng(17)
    weights = rng.normal(size=(256, 64))
    scales = np.max(abs(weights), axis=1)
    offset = scales.mean(); residual = scales-offset
    second_scale = np.max(abs(residual))/127
    codes = np.clip(np.rint(residual/second_scale), -127, 127).astype(np.int8)
    restored_scales = offset+second_scale*codes
    assert np.max(abs(restored_scales-scales)) <= second_scale/2+1e-12
    original_bpw = 32/64
    nested_bpw = 8/64+32/(64*256)
    print("scale overhead before/after, extra center bits:", original_bpw, nested_bpw, 32/weights.size)
    ''',
    ["8/64 是每个权重摊到的一级尺度编码，32/(64*256) 是二级 FP32 scale 的摊销。",
     "offset 也需要保存，代码另行报告它的开销。重建尺度误差还会传播到整组权重。"],
    "这里用均匀 INT8 示范嵌套量化，不等同于 bitsandbytes 的具体 8 bit 动态码本和序列化格式。", "R24")

add("loftq", "07", "LoftQ：量化与低秩补偿交替初始化", "7.11",
    "低秩适配器不必从任意残差开始。LoftQ 交替量化 W-BA，并对剩余误差做截断 SVD，使量化主体和低秩增量在微调前就相互配合。",
    ["先量化 W-BA", "计算剩余误差 W-Q", "截断 SVD 得到低秩 BA", "交替若干次，再开始任务微调"],
    '''
    import numpy as np
    from experiments.quant_lab import group_qdq
    rng = np.random.default_rng(18)
    W = rng.normal(size=(6, 8)); correction = np.zeros_like(W); rank = 2
    for step in range(4):
        Q = group_qdq(W-correction, group_size=4)
        E = W-Q
        U, singular, Vt = np.linalg.svd(E, full_matrices=False)
        B = U[:, :rank]*np.sqrt(singular[:rank])
        A = np.sqrt(singular[:rank])[:, None]*Vt[:rank]
        correction = B@A
        assert np.isclose(np.sum((E-correction)**2), np.sum(singular[rank:]**2))
        print("iteration, residual SSE:", step, np.sum((W-Q-correction)**2))
    ''',
    ["固定 Q 时，截断 SVD 给出 Frobenius 范数下最佳 rank-2 修正；断言使用舍弃奇异值能量独立核对。",
     "下一轮量化 W-correction，让低比特主体与低秩分支重新分工。"],
    "当前 quantizer 的尺度重新估计是启发式，不能保证交替过程每轮单调下降；实际 LoRA 的 α/r 缩放也需纳入初始化。", "R25")

add("bitnet", "07", "BitNet b1.58：三值权重", "7.12",
    "使用平均绝对值尺度将前向权重映射到 {-1,0,1}，并配合原生训练适应这种表示。三值点积可拆成正项相加与负项相减，但激活与整体尺度仍需要处理。",
    ["高精度潜在权重", "除以 absmean 并舍入", "三值点积：正项和减负项和", "乘尺度恢复，训练中继续适配"],
    '''
    import numpy as np
    w = np.array([-.7, -.1, .2, .8]); x = np.array([1., 2., 3., 4.])
    beta = np.mean(abs(w))
    q = np.clip(np.rint(w/max(beta, 1e-8)), -1, 1).astype(int)
    additions = x[q == 1].sum()-x[q == -1].sum()
    output = beta*additions
    assert set(q).issubset({-1, 0, 1})
    assert np.isclose(output, (beta*q)@x)
    print("ternary weights, output, ideal bits:", q, output, np.log2(3))
    ''',
    ["q 只控制是否加、减或跳过一个输入；最后乘 beta 才恢复权重的整体幅度。",
     "log2(3) 是三值块编码的理想平均位数，不代表 NumPy int 数组已经只占 1.58 bit/元素。"],
    "未实现完整 BitLinear 的激活量化、归一化或原生训练；直接把现成 BF16 模型三值化通常不能复现论文能力。", "R26")

add("paged_optimizer", "07", "Paged Optimizer：按需迁移优化器状态", "7.10",
    "分页优化器将暂时不用的优化器状态放到主存，在需要时迁入设备，以缓和显存峰值。它改变状态放在哪里，不会改变 NF4 编码，也不意味着迁移没有带宽成本。",
    ["优化器状态分成页面", "当前更新需要哪些页面", "缺页迁入，空间不足则迁出", "更新状态并统计迁移成本"],
    '''
    from collections import OrderedDict
    import numpy as np
    host = {j: np.zeros(4) for j in range(3)}
    resident = OrderedDict(); reference = {j: np.zeros(4) for j in range(3)}
    loads = 0
    for page in [0, 1, 0, 2, 1]:
        if page not in resident:
            if len(resident) == 2:
                old, state = resident.popitem(last=False); host[old] = state.copy()
            resident[page] = host[page].copy(); loads += 1
        resident.move_to_end(page)
        resident[page] = .9*resident[page]+.1*np.ones(4)
        reference[page] = .9*reference[page]+.1*np.ones(4)
        assert len(resident) <= 2
    for page in host:
        assert np.allclose(resident.get(page, host[page]), reference[page])
    print("resident pages, total loads:", len(resident), loads)
    ''',
    ["OrderedDict 在这里模拟两个页面的驻留预算，淘汰页面时必须把更新后的状态写回 host。",
     "与不分页的 reference 比较，验证状态迁移没有丢失动量；loads 展示容量下降带来的数据搬运。"],
    "这是 LRU 状态迁移的教学模拟；实际 QLoRA paged optimizer 使用统一内存等机制，不等同于本例调度器，也没有在这里测量 GPU 内存。", "R24")

add("kivi", "08", "KIVI：不同量化轴与流式残留区", "8.3",
    "Key 沿时间轴对每个通道分组，Value 沿特征轴对每个 token 分组。Key 缓冲积满后批量量化，Value 维持最近窗口并逐行淘汰；两者的残留长度不必始终相同。",
    ["新 token 进入\nK/V 高精度残留区", "K 满 R 后沿时间分组压缩", "V 超 R 时压缩最旧一行", "压缩前缀与残留\n共同参与 Attention"],
    '''
    import numpy as np
    from experiments.technique_kernels import TinyKIVI
    cache = TinyKIVI(dim=4, group=2, residual=4, bits=2)
    keys, values = [], []
    assert cache.reconstruct()[0].shape == (0, 4)
    for t in range(9):
        k = t*np.array([1., 8., .1, .5])+np.arange(4)*.03
        v = np.sin(np.arange(4)+t)
        keys.append(k); values.append(v); cache.append(k, v)
        K, V = cache.reconstruct()
        assert K.shape == V.shape == (t+1, 4)
        if cache.k_recent:
            assert np.array_equal(K[-1], k)
        assert np.array_equal(V[-1], v)
    assert (len(cache.k_recent), len(cache.v_recent)) == (1, 4)
    assert np.array_equal(V[-4:], np.array(values)[-4:])
    print("K/V high-precision tail lengths:", len(cache.k_recent), len(cache.v_recent))
    ''',
    ["TinyKIVI 的 encode(axis=0) 给 Key 的每个固定通道沿 token 分组；Value reshape 后用 axis=1 沿特征分组。",
     "9 个 token、残留阈值 4 时，Key 已批量压缩两次，只剩 1 行；Value 仍保留最近 4 行。检查尾部原值和总顺序。"],
    "单头 CPU 模型，2 bit 编码存入 uint8 容器，未物理打包；不实现融合 Attention、分页或实际压缩率测量。", "R27")

add("kvquant", "08", "KVQuant：pre-RoPE Key 与非均匀稀疏表示", "8.4",
    "RoPE 前的 Key 通道结构较稳定，适合按通道校准非均匀码本。动态离群值另存稀疏残差；读取时恢复 Key 后必须按原位置正确施加 RoPE。",
    ["校准 pre-RoPE 通道码本", "新 Key 选择码本索引", "离群位置额外保存残差", "解码后按位置执行 RoPE"],
    '''
    import numpy as np
    from experiments.technique_kernels import lloyd, nearest, rope_rows
    rng = np.random.default_rng(20)
    calibration = rng.normal(size=(128, 4))*[.1, 8., .2, 2.]
    K = rng.normal(size=(6, 4))*[.1, 8., .2, 2.]; K[2, 1] = 40.
    dense = np.empty_like(K)
    for j in range(4):
        centers, _ = lloyd(calibration[:, j], 4)
        dense[:, j] = nearest(K[:, j], centers)
    protected = abs(K) > np.max(abs(calibration), axis=0)
    residual = np.where(protected, K-dense, 0)
    restored = dense+residual
    positions = np.arange(len(K))
    rotated = rope_rows(restored, positions)
    reference = rope_rows(K, positions)
    assert restored[2, 1] == K[2, 1]
    assert np.isclose(np.sum((rotated-reference)**2), np.sum((restored-K)**2))
    print("sparse exceptions, post-RoPE SSE:", protected.sum(), np.sum((rotated-reference)**2))
    ''',
    ["码本在校准集上学习，在线 Key 只做索引选择和稀疏残差生成；两个阶段的数据分开。",
     "RoPE 是逐位置正交变换，因此解码误差的二范数保持不变，但量化轴的分布统计会改变。"],
    "为缩短例子，码本未使用真实 Fisher 敏感性加权；展示的是 pre-RoPE、码本和稀疏恢复机制，未实现 KVQuant 的全部 Value 策略与 kernel。", "R28")

add("turboquant", "08", "TurboQuant：球面分布的预计算量化", "8.5",
    "对单位向量做 Haar 随机旋转后，单坐标有已知球面边缘分布，可按维度提前设计 Lloyd-Max 码本。存储范数和坐标索引，解码时逆旋转恢复原坐标。",
    ["按维度预计算球面码本", "向量拆成范数与方向", "随机正交旋转后逐坐标编码", "查表、逆旋转并乘回范数"],
    '''
    import numpy as np
    from experiments.technique_kernels import lloyd, nearest
    rng = np.random.default_rng(21); d = 8
    grid = np.linspace(-.999, .999, 2048)
    density = (1-grid**2)**((d-3)/2)
    centers, _ = lloyd(grid, 8, importance=density)
    orthogonal, triangular = np.linalg.qr(rng.normal(size=(d, d)))
    R = orthogonal@np.diag(np.sign(np.diag(triangular)))
    x = rng.normal(size=d); norm = np.linalg.norm(x)
    unit = x/norm; rotated = R@unit
    decoded = norm*(R.T@nearest(rotated, centers))
    assert np.allclose(R.T@R, np.eye(d))
    assert np.isclose(np.linalg.norm(rotated), 1)
    print("precomputed centers, reconstruction MSE:", centers, np.mean((x-decoded)**2))
    ''',
    ["density 是球面坐标密度的未归一化权重；加权均值不受统一归一化常数影响，因此可直接学习离散近似码本。",
     "QR 后修正 R 因子的对角符号得到高斯矩阵诱导的 Haar 正交矩阵；码本只依赖维度和位宽，不拟合模型权重。"],
    "用离散积分近似标量码本，只演示 MSE 路径；零向量需直接编码为零，快速结构化旋转也不自动满足相同 Haar 假设。", "R29")

add("qjl", "08", "QJL：一位残差内积修正", "8.6",
    "先进行 MSE 量化，再用独立高斯投影的符号编码估计剩余内积。乘以残差范数和正确归一化常数后，对随机映射取期望可恢复真实残差内积。",
    ["计算残差 r=x-x_hat", "保存 sign(G*r) 与范数", "查询计算 G*q", "符号内积经归一化后补回"],
    '''
    import numpy as np
    rng = np.random.default_rng(22)
    residual = rng.normal(size=8); query = rng.normal(size=8)
    G = rng.normal(size=(100000, 8))
    signs = np.sign(G@residual)
    samples = np.linalg.norm(residual)*np.sqrt(np.pi/2)*signs*(G@query)
    target = residual@query
    se = samples.std(ddof=1)/np.sqrt(len(samples))
    assert abs(samples.mean()-target) < 6*se
    print("true dot, estimated mean, standard error:", target, samples.mean(), se)
    ''',
    ["每个高斯投影产生一个可保存的符号位；查询投影保持实值，避免两侧同时符号化改变估计器。",
     "100000 个样本用于检查概率恒等式，不是每个实际 KV 向量都要保存这么多 bit；实际投影数决定方差与存储开销。"],
    "G 必须满足独立性与分布条件；残差为零时应直接返回零。内积无偏不推出 softmax、Attention 或生成文本无偏。", "R29")

add("sageattention", "08", "SageAttention：Key 平滑与 INT8 QK", "8.8",
    "所有 Key 减去同一向量，只会给每行 logits 减去同一常数，精确 softmax 概率不变。平滑后再用 INT8 处理 QK，可减轻 Key 离群值对共享范围的影响。",
    ["K 减去共同中心 μK", "每行 logits 仅差一个常数", "Q/K 量化后做点积", "softmax 后与 V 相乘"],
    '''
    import numpy as np
    from experiments.quant_lab import symmetric_qdq, softmax, mse
    rng = np.random.default_rng(23)
    Q = rng.normal(size=(3, 8)); K = rng.normal(size=(12, 8))+5
    V = rng.normal(size=(12, 4)); centered = K-K.mean(axis=0)
    reference = softmax(Q@K.T/np.sqrt(8))@V
    assert np.allclose(softmax(Q@centered.T/np.sqrt(8))@V, reference)
    Q8 = symmetric_qdq(Q, bits=8, axis=1)[0]
    K8 = symmetric_qdq(centered, bits=8, axis=1)[0]
    output = softmax(Q8@K8.T/np.sqrt(8))@V
    print("smoothed INT8 attention output MSE:", mse(output, reference))
    ''',
    ["先用未量化点积验证平移恒等式，确保观察到的后续误差来自量化。",
     "这里保留 V 和 PV 的高精度，以单独观察 QK INT8 与 Key 平滑；真实版本的具体类型见原论文。"],
    "按行 QDQ 的 CPU 教学实现，没有复现分块在线 softmax、融合内核或论文的速度数字。", "R30")

add("sageattention2", "08", "SageAttention2：Query 修正、INT4 与 FP8", "8.9",
    "Query 减中心后产生随 Key 位置变化的修正项，不能像 Key 平移常数一样丢掉。SageAttention2 结合更细粒度 INT4 QK、FP8 PV 和累加精度控制。",
    ["同时平滑 Q 与 K", "INT4 点积加回 Query 修正项", "在线 softmax 的 P 与 V 用 FP8", "高精度缓冲合并累加"],
    '''
    import numpy as np
    from experiments.quant_lab import symmetric_qdq, softmax, mse
    from experiments.technique_kernels import float_qdq
    rng = np.random.default_rng(24)
    Q = rng.normal(size=(3, 8))+2; K = rng.normal(size=(16, 8))+3
    V = rng.normal(size=(16, 4)); Kc = K-K.mean(axis=0)
    mean_Q = Q.mean(axis=0, keepdims=True); Qc = Q-mean_Q
    correction = mean_Q@Kc.T
    assert np.allclose(Qc@Kc.T+correction, Q@Kc.T)
    assert np.ptp(correction) > 0
    Q4 = symmetric_qdq(Qc, bits=4, axis=1)[0]
    K4 = symmetric_qdq(Kc, bits=4, axis=1)[0]
    P = softmax((Q4@K4.T+correction)/np.sqrt(8))
    output = float_qdq(P, axis=1)@float_qdq(V, axis=0)
    reference = softmax(Q@K.T/np.sqrt(8))@V
    print("INT4 QK plus FP8 PV output MSE:", mse(output, reference))
    ''',
    ["correction 在不同 Key 位置不同，np.ptp 检查它不是可被 softmax 消除的常数。",
     "先恢复 Query 修正，再执行 softmax；P/V 通过有限 E4M3 码本做数值模拟，累加在 NumPy 高精度中完成。"],
    "未复现 per-thread 数据布局，代码用按行 QDQ 与显式 P 展示数据流；原生 FP8 部分累加和 FP32 buffer 的误差需在 GPU 上验证。", "R31")

add("sageattention3", "08", "SageAttention3：FP4 Attention 与 P 两级缩放", "8.10",
    "在支持的硬件上用微缩放 FP4 执行 QK 和 PV。对概率块先做行级缩放，再用 E4M3 微块尺度量化 FP4，使小概率对应的尺度能够使用更合适的数值范围。",
    ["Q/K 微块 FP4 点积", "P 的行级尺度\nα=max(P)/(448×6)", "P/α：每 16 个值\n做一次 FP4 微缩放", "恢复两级尺度\n再计算 PV"],
    '''
    import numpy as np
    from experiments.quant_lab import softmax, mse
    from experiments.technique_kernels import micro_fp4
    rng = np.random.default_rng(25)
    Q = rng.normal(size=(3, 32)); K = rng.normal(size=(32, 32))
    V = rng.normal(size=(32, 32))
    Q4, _ = micro_fp4(Q); K4, _ = micro_fp4(K)
    P = softmax(Q4@K4.T/np.sqrt(32))
    alpha = np.max(P, axis=1, keepdims=True)/(448*6)
    P4 = np.empty_like(P)
    for row in range(len(P)):
        normalized, _ = micro_fp4(P[row:row+1]/alpha[row], tensor_scale=1.)
        P4[row] = alpha[row]*normalized[0]
    V4, _ = micro_fp4(V.T); V4 = V4.T  # Group along the PV reduction axis.
    output = P4@V4
    assert np.all(np.isfinite(output)) and np.all(P4 >= 0)
    print("two-level FP4 attention MSE:", mse(output, softmax(Q@K.T/np.sqrt(32))@V))
    ''',
    ["α 的行级缩放和 micro_fp4 内的 16 元素尺度是两层不同尺度；恢复时必须一起乘回。",
     "V 先转置再量化，使微块沿 PV 的归约轴组织，不能仅因为 shape 可乘就随意选择分组轴。"],
    "展示完整概率矩阵的数值路径；真实内核使用未最终归一化的分块概率并维护在线 softmax 统计，不会显式存整个 P。有限 LUT 舍入也不是硬件逐位仿真。", "R32")

add("fp8", "09", "FP8：E4M3 与 E5M2", "9.1",
    "指数位控制动态范围，尾数位控制同一数量级的分辨率。E4M3FN 和 E5M2 的有限值集合不同，量化时通常还要用张量或局部尺度把数据映射进有限范围。",
    ["明确指数、尾数与特殊编码", "选择 E4M3FN 或 E5M2", "用外部 scale 归一化", "映射到有限值并恢复尺度"],
    '''
    import numpy as np
    from experiments.technique_kernels import fp_levels, float_qdq
    e4 = fp_levels('e4m3'); e5 = fp_levels('e5m2')
    assert e4.max() == 448 and e5.max() == 57344
    assert e4[e4 > 0].min() == 2.**-9
    assert e5[e5 > 0].min() == 2.**-16
    x = np.array([0., .1, 1.3, 20., 1000.])
    print("E4M3 scaled reconstruction:", float_qdq(x, 'e4m3'))
    print("E5M2 scaled reconstruction:", float_qdq(x, 'e5m2'))
    ''',
    ["fp_levels 按指数/尾数枚举有限值并剔除保留编码；边界断言核对具体格式而不是泛称 FP8。",
     "float_qdq 先除以 amax/Fmax，再映射到格式代表值，最后乘回尺度。浮点有指数仍然可能需要外部 scale。"],
    "不含 NaN、无穷传播和有符号零；LUT 在恰好半格处取排序中的先值，不是硬件 ties-to-even 的逐位替代。", "R33")

add("dynamic_scaling", "09", "静态、当前动态与延迟缩放", "9.3",
    "静态尺度来自离线校准，当前动态尺度依赖当前输入的最大值，延迟尺度使用历史统计以减轻当前归约依赖。突发范围变化可能使后两种方案产生不同的裁剪行为。",
    ["静态：固定校准 amax", "当前动态：先归约本步 amax", "延迟：读取历史 amax", "记录本步统计供未来使用"],
    '''
    import numpy as np
    from experiments.technique_kernels import fp_levels, nearest
    levels = fp_levels('e4m3')
    def quantize(row, bound):
        scale = max(bound, 1e-12)/448
        return scale*nearest(row/scale, levels)
    fixed_amax = history_amax = 1.
    for row in (np.array([.2, 1.]), np.array([1., 16.])):
        current_amax = np.max(abs(row))
        static = quantize(row, fixed_amax)
        current = quantize(row, current_amax)
        delayed = quantize(row, history_amax)
        print("static/current/delayed:", static, current, delayed)
        history_amax = current_amax
    assert current[-1] == 16 and delayed[-1] == static[-1] == 1
    ''',
    ["第二步突然出现 16，当前动态方案及时扩大范围；静态与只看上一步的延迟方案仍按最大值 1 编码。",
     "更新 history_amax 必须在使用历史尺度之后，否则就偷偷变成当前动态缩放。块动态缩放见后面的 MXFP4/NVFP4 示例。"],
    "生产延迟缩放常用多步历史最大值和 margin，本例只用一帧历史展示时序；饱和、溢出策略也需按后端确定。", "R33，第 09 章缩放分类")

add("fp4", "09", "FP4 E2M1：指数型非均匀代表值", "9.4",
    "E2M1 的非负有限值为 0、0.5、1、1.5、2、3、4、6，越大数值之间的绝对间隔越大。它与等间隔 INT4 和查表 NF4 的编码语义不同。",
    ["1 个符号、2 个指数、1 个尾数位", "枚举 E2M1 有限值", "选最近代表值", "共享尺度扩展可用范围"],
    '''
    import numpy as np
    from experiments.technique_kernels import fp_levels, nearest
    levels = fp_levels('e2m1')
    assert np.array_equal(levels[levels >= 0], [0, .5, 1, 1.5, 2, 3, 4, 6])
    x = np.array([.1, .6, 1.2, 2.7, 5.8])
    q = nearest(x, levels)
    print("inputs, E2M1 values:", x, q)
    print("unique real values:", len(levels))
    assert len(levels) == 15
    ''',
    ["16 个物理编码包含正零和负零，因此只有 15 个不同实数值；LUT 把两个零合并了。",
     "2.7 可以映射到 3，而 5.8 映射到 6，间距随数量级变化。scale 如何选择由上层微缩放格式决定。"],
    "代码不返回 E2M1 的原始 4 bit 位模式，也不模拟有符号零；只检查有限值几何。", "R34、R35")

add("mxfp4", "09", "MXFP4：32 元素共享二次幂尺度", "9.5",
    "每 32 个 E2M1 元素共享一个 E8M0 二次幂尺度。小块缩放适应局部范围，而二次幂尺度便于硬件实现；尺度过粗也会损失有效分辨率。",
    ["沿归约轴每 32 个值分块", "选择 E8M0 二次幂尺度", "块内映射到 E2M1", "存储 4 bit 值与 8 bit 块尺度"],
    '''
    import numpy as np
    from experiments.technique_kernels import micro_fp4
    x = np.r_[np.linspace(-.8, .8, 32), np.linspace(-8, 8, 32)]
    restored, metadata = micro_fp4(x, kind='mx')
    scales = metadata['block_scale'].ravel()
    assert np.allclose(np.log2(scales), np.rint(np.log2(scales)))
    assert np.array_equal(micro_fp4(np.zeros(32), 'mx')[0], np.zeros(32))
    print("block scales, MSE, bits/value:", scales, np.mean((x-restored)**2), 4+8/32)
    ''',
    ["helper 将末轴分成 32 个一块，计算理想 amax/6，再使用 ceil(log2) 选择不小于它的二次幂尺度。",
     "4+8/32=4.25 是主体和尺度的平均位宽，不能只按每个元素 4 bit 计算。"],
    "这里采用保守 ceil 选尺度策略，不承诺等于任一设备的默认 recipe；E8M0 极端范围和特殊编码未逐位实现。", "R34")

add("nvfp4", "09", "NVFP4：16 元素、E4M3 尺度与全局尺度", "9.6",
    "每 16 个 E2M1 元素共享一个 E4M3 尺度，再由张量级高精度尺度覆盖总体范围。相较二次幂尺度，局部尺度更细，但需要更多元数据。",
    ["张量 amax 确定全局尺度", "每 16 个值计算局部范围", "局部尺度用 E4M3 表示", "全局尺度×局部尺度×FP4 值"],
    '''
    import numpy as np
    from experiments.technique_kernels import micro_fp4, fp_levels
    x = np.r_[np.linspace(-.8, .8, 16), np.linspace(-8, 8, 16)]
    restored, meta = micro_fp4(x, kind='nv')
    assert np.all(np.isin(meta['block_scale'], fp_levels('e4m3')))
    recomputed = meta['tensor_scale']*meta['block_scale']*meta['payload']
    assert np.array_equal(restored, recomputed.reshape(x.shape))
    print("tensor scale, block scales:", meta['tensor_scale'], meta['block_scale'].ravel())
    print("MSE, block-only bits/value:", np.mean((x-restored)**2), 4+8/16)
    ''',
    ["张量尺度把局部尺度的可用范围映射到 E4M3，解码时必须与局部尺度一起乘回。",
     "4.5 bit/value 尚未计张量尺度、padding 或未量化张量；block scale 是 FP8 不代表数据本身也是 FP8。"],
    "NumPy 数值分解不执行原生 NVFP4 指令；真实速度、数据布局和舍入规则依赖硬件及后端。", "R35")

add("gguf", "09", "GGUF：容器与量化类型分离", "9.7",
    "GGUF 负责组织文件头、元数据、张量描述和数据区，具体量化类型由各张量的类型决定。改变扩展名或容器头不会自动将 FP16 权重量化成 INT4。",
    ["GGUF 文件头与版本", "键值元数据与张量描述", "张量类型及数据偏移", "对应解码器读取真实数据"],
    '''
    import struct
    # GGUF v3 header only: magic, version, tensor_count, metadata_kv_count.
    header = struct.pack('<4sIQQ', b'GGUF', 3, 0, 0)
    magic, version, tensors, metadata = struct.unpack('<4sIQQ', header)
    assert magic == b'GGUF' and version == 3
    assert len(header) == 24 and tensors == metadata == 0
    print("header fields:", magic, version, tensors, metadata)
    ''',
    ["4s/I/Q/Q 对应 4 字节魔数、32 bit 版本以及两个 64 bit 计数；小端格式由 '<' 指定。",
     "这个头本身没有任何数值量化信息；真正的张量描述和数据块才决定如何解码参数。"],
    "只构造空容器头来解释格式层次，不生成可加载的 LLM，也不覆盖完整 GGUF 元数据或对齐规范。", "R36，GGUF 官方规范")

add("q4_0", "09", "Q4_0：真实 nibble 打包与块解码", "9.8",
    "Q4_0 每块 32 个权重，用一个 FP16 尺度和 16 字节 nibble 数据保存。编码通常减 8 后得到 [-8,7]，尺度可为负；它与教程基础 [-7,7] 对称网格不同。",
    ["32 个权重确定有符号尺度 d", "映射到 0..15 编码", "前16项放低 nibble、后16项放高位", "读取 FP16 d 后按 d*(q-8) 解码"],
    '''
    import numpy as np
    w = np.linspace(-1, 1, 32, dtype=np.float32)
    extreme = w[np.argmax(abs(w))]
    d = float(extreme/-8) if extreme != 0 else 0.
    codes = np.clip(np.floor(w/d+8.5), 0, 15).astype(np.uint8) if d else np.full(32, 8, dtype=np.uint8)
    packed = codes[:16] | (codes[16:] << 4)
    block = np.array([d], dtype='<f2').tobytes()+packed.tobytes()
    decoded_scale = float(np.frombuffer(block, dtype='<f2', count=1)[0])
    data = np.frombuffer(block, dtype=np.uint8, offset=2)
    unpacked = np.r_[data & 15, data >> 4].astype(int)
    restored = decoded_scale*(unpacked-8)
    assert np.array_equal(unpacked, codes) and len(block) == 18
    print("bytes, effective bits, reconstruction MSE:", len(block), len(block)*8/32, np.mean((w-restored)**2))
    ''',
    ["Q4_0 的同一字节分别放第 j 和第 j+16 个编码，并非简单把相邻两个权重塞在一起。",
     "解码必须使用已存成 FP16 的尺度；用量化前 float32 尺度验证会漏掉尺度自身的舍入误差。"],
    "展示基础参考量化与块布局，不承诺与含 importance 优化的所有 llama.cpp 版本编码一致；本例不拼装完整 GGUF。", "R36")

add("q4_k", "09", "Q4_K：超块与子块参数量化", "9.8",
    "256 个权重组成超块，内部使用 8 个 32 元素子块，各有尺度和最小值参数。子块参数进一步量化为 6 bit，并由两个 FP16 超块参数恢复。",
    ["256 个权重分成 8 个子块", "每块计算 scale 与 min", "scale/min 再用 6 bit 编码", "4 bit 数据加层级元数据恢复权重"],
    '''
    import numpy as np
    rng = np.random.default_rng(26)
    blocks = rng.normal(size=(8, 32))
    minimum = -blocks.min(axis=1, keepdims=True)
    scales = (blocks.max(axis=1, keepdims=True)+minimum)/15
    d, dmin = float(np.float16(scales.max()/63)), float(np.float16(minimum.max()/63))
    scale6 = np.clip(np.rint(scales/d), 1, 63)
    min6 = np.clip(np.rint(minimum/dmin), 0, 63)
    effective_s, effective_m = d*scale6, dmin*min6
    q = np.clip(np.rint((blocks+effective_m)/effective_s), 0, 15)
    reconstructed = q*effective_s-effective_m
    total_bytes = 128+12+4
    assert total_bytes*8/256 == 4.5
    print("Q4_K byte accounting, toy MSE:", total_bytes, np.mean((blocks-reconstructed)**2))
    ''',
    ["scale6/min6 一共 16 个 6 bit 参数，占 12 字节；两个 FP16 参数另占 4 字节，主体 4 bit 编码占 128 字节。",
     "先量化子块参数，再根据实际有效尺度选择 q，避免用未存储的理想参数反量化。"],
    "用简单 min/max 展示分层数值关系，不生成 Q4_K 官方位布局；真实编码器的搜索、负最小值约定和边界处理更加完整。", "R36")

add("q4_k_m", "09", "Q4_K_M：混合量化预设", "9.8",
    "Q4_K_M 是为不同张量选择不同存储类型的预设，不是一个统一 block 的名称。部分张量使用更高精度，因此模型平均位宽需按实际张量分配加权计算。",
    ["识别模型中的各类张量", "按预设选择 Q4_K 或较高精度", "计算每个类型的块成本", "加总字节得到真实平均位宽"],
    '''
    tensors = [
        ('toy_mlp', 4096*4096, 4.5),
        ('toy_attention', 4096*4096, 6.5625),  # Q6_K block: 210 bytes / 256 values.
        ('toy_norm', 4096, 32.),
    ]
    total_parameters = sum(n for _, n, _ in tensors)
    total_bytes = sum(n*bpw/8 for _, n, bpw in tensors)
    average = total_bytes*8/total_parameters
    assert average > 4.5
    print("weighted effective bits, bytes:", average, total_bytes)
    ''',
    ["平均位宽必须用参数量加权，而不是把类型位宽直接取算术平均；少量 FP32 张量也要计入。",
     "这里的名字带 toy，表示人为指定的组合，用来演示预设与 block 类型的区别。"],
    "上述张量分配不是 llama.cpp 的固定 Q4_K_M 默认规则；真实映射随架构与版本变化，应读取实际 GGUF 张量类型。", "R36")

add("iq", "09", "IQ 系列与 importance matrix", "9.8",
    "IQ 包含多种非均匀和向量编码格式，重要性统计帮助编码器把误差放在较不敏感的方向。输入平方统计可构成对角权重，但不等于完整任务 Hessian。",
    ["校准输入的平方统计", "形成通道重要性 h", "比较有限编码候选的加权误差", "选择格式支持的代表组合"],
    '''
    import numpy as np
    X = np.array([[1., 10.], [2., -10.]])
    importance = np.sum(X**2, axis=0)
    w = np.array([.2, .2])
    toy_codebook = np.array([[.2, 0.], [0., .25]])
    plain = np.sum((toy_codebook-w)**2, axis=1)
    weighted = np.sum(importance*(toy_codebook-w)**2, axis=1)
    assert plain.argmin() == 0 and weighted.argmin() == 1
    print("importance, unweighted/weighted choices:", importance, plain.argmin(), weighted.argmin())
    ''',
    ["第二个输入方向幅度更大，在那里产生相同误差会更昂贵，因此加权后改选更照顾第二维的候选。",
     "有限向量码本的候选选择存在耦合，重要性可以改变最优代表；不是简单给重要层多几个 bit。"],
    "toy_codebook 不是任何具体 IQ 格式的官方码本；IQ2、IQ3、IQ4 等各有规则，部署须选择对应编码器和解码内核。", "R36")

add("roofline", "09", "Roofline 与 Amdahl：估计加速边界", "9.10",
    "权重压缩主要减少数据读取，原生低精度运算还可能提高计算吞吐。总时间受计算、带宽和额外处理共同影响；局部算子加速不能直接等同于端到端加速。",
    ["估计计算量与搬运字节", "取计算/带宽瓶颈较大者", "加入解包与转换成本", "按 Amdahl 合并未加速部分"],
    '''
    flops = 2*7e9
    compute = 100e12; bandwidth = 1e12
    fp16_time = max(flops/compute, 14e9/bandwidth)
    int4_time = max(flops/compute, 3.5e9/bandwidth)+.001
    local_speedup = fp16_time/int4_time
    total_speedup = 1/(.4+.6/4)
    assert local_speedup < 4
    assert abs(total_speedup-1.818181818) < 1e-8
    print("estimated FP16/INT4 milliseconds:", fp16_time*1000, int4_time*1000)
    print("local estimate and Amdahl example:", local_speedup, total_speedup)
    ''',
    ["max 反映计算与数据供给中的主要瓶颈，额外 1 ms 用来说明解包等开销会压低理想压缩收益。",
     "Amdahl 例子假设只有 60% 的原总时间能加速 4 倍，剩余 40% 保持不变。"],
    "全部数值都是预算假设，没有进行 CPU/GPU 性能测量；通信、缓存和依赖也未包含在这个简单模型中。", "第 09 章性能模型")


from experiments.latest_techniques import LATEST_CARDS

CARDS.extend(LATEST_CARDS)

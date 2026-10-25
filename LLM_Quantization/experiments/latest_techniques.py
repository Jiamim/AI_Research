"""Numerical lessons accompanying the 2026-10-11 literature update.

Each lesson demonstrates a named, bounded mechanism from its cited paper.
These are synthetic CPU examples, not paper benchmark reproductions.
"""

from textwrap import dedent

LATEST_CARDS = []


def add(key, chapter, title, section, paper_id, principle, steps, code, explanation, boundary):
    LATEST_CARDS.append(dict(id=key, chapter=chapter, title=title, section=section,
                             paper_id=paper_id, principle=principle, steps=steps,
                             code=dedent(code).strip(), explanation=explanation,
                             boundary=boundary, refs=f'arXiv:{paper_id}'))


add('duquant', '12', 'DuQuant：块内旋转与跨块重排', '12.1', '2406.01721',
    '先在局部块内分散离群值，再通过 zigzag 重排平衡不同块，随后再次旋转。整个变换及其逆变换必须配对，才能保持原线性函数。',
    ['块内分散离群值', '按峰值进行 zigzag 重排', '再次进行块内旋转', '权重配对变换后量化'],
    '''
    import numpy as np
    from experiments.quant_lab import hadamard
    X = np.array([[10., 8., .2, .1, .1, .3, .2, .1], [4., 2., .3, .1, .2, .4, .1, .2]])
    W = np.arange(24, dtype=float).reshape(3, 8)/20
    R = np.kron(np.eye(2), hadamard(4))
    order = np.argsort(-np.max(abs(X@R), axis=0))
    buckets = [[], []]
    for rank, channel in enumerate(order):
        bucket = rank % 2 if (rank//2) % 2 == 0 else 1-rank % 2
        buckets[bucket].append(channel)
    permutation = np.eye(8)[:, buckets[0]+buckets[1]]
    T = R@permutation@R
    assert np.allclose(T.T@T, np.eye(8))
    assert np.allclose((X@T)@(W@T).T, X@W.T)
    print('zigzag order:', buckets[0]+buckets[1])
    print('peak before/after:', np.max(abs(X)), np.max(abs(X@T)))
    ''',
    ['重排按块号往返分配大、小峰值，避免最大的通道全部进入同一个块。',
     'R、置换和第二次 R 都正交，因此它们的乘积仍正交；权重右乘同一 T 保持输出。'],
    '用固定 Hadamard 代替论文的离群值引导贪心旋转，省略初始平滑和完整校准；本例验证组合变换，不保证每组数据的峰值都下降。')

add('ostquant', '12', 'OSTQuant：正交与尺度联合变换', '12.2', '2501.13987',
    '把正交矩阵和对角尺度共同作为等价变换参数，改善量化空间利用率；在全局校准中使用教师 top-k 预测信息，缓解小校准集上的目标偏差。',
    ['设置正交矩阵和通道尺度', '以 T 和逆转置配对', '前向加入量化误差', '以教师 top-k 信息校准'],
    '''
    import numpy as np
    from experiments.quant_lab import hadamard, symmetric_qdq, softmax
    rng = np.random.default_rng(1202)
    X = rng.normal(size=(6, 4)); W = rng.normal(size=(3, 4))
    T = hadamard(4)@np.diag([.5, 1., 2., 1.])
    Wp = np.linalg.solve(T, W.T).T
    assert np.allclose((X@T)@Wp.T, X@W.T)
    teacher = softmax(X@W.T)
    student = softmax(symmetric_qdq(X@T, axis=1)[0] @ symmetric_qdq(Wp, axis=1)[0].T)
    indices = np.argsort(teacher, axis=1)[:, -2:]
    p = np.take_along_axis(teacher, indices, axis=1)
    q = np.take_along_axis(student, indices, axis=1)
    kl_top = np.mean(np.sum(p*np.log(p/q), axis=1))
    assert np.isfinite(kl_top)
    print('teacher top-2 indices and partial KL:', indices, kl_top)
    ''',
    ['T=R D 的逆转置一般不等于 T；solve 明确保持矩阵方向，避免把一般缩放当成正交变换。',
     '示例按原文 top-k 求和式保留原概率，不重新归一化；截断部分和不能直接套用完整 KL 非负性的结论。'],
    '仅验证变换和 KL-Top 的计算，没有实现 QSUR 评估、WOMI 初始化或跨 Transformer 块的优化与融合。')

add('paroquant', '12', 'ParoQuant：可并行的成对旋转', '12.3', '2511.10645',
    '使用独立二维 Givens 旋转和通道缩放来改善 weight-only 量化。成对结构减少变换开销，但不可吸收的在线旋转仍需要相应内核。',
    ['通道两两配对', '构造二维 Givens 旋转', '联合通道尺度选择变换', '匹配量化布局与执行内核'],
    '''
    import numpy as np
    from experiments.quant_lab import group_qdq, mse
    rng = np.random.default_rng(1203)
    X = rng.normal(size=(16, 4)); X[:, 0] *= 8
    W = rng.normal(size=(3, 4)); scores = []
    D = np.diag([.5, 2., 1., 1.])
    for theta in [0., np.pi/8, np.pi/4]:
        c, s = np.cos(theta), np.sin(theta)
        R = np.kron(np.eye(2), [[c, -s], [s, c]])
        T = D@R; Wp = np.linalg.solve(T, W.T).T
        assert np.allclose((X@T)@Wp.T, X@W.T)
        scores.append(mse(X@W.T, (X@T)@group_qdq(Wp, group_size=2).T))
    print('candidate-angle output MSE:', scores)
    ''',
    ['二维旋转块之间互不依赖；缩放和旋转的先后顺序由 T 明确表示。',
     '只量化权重，保持激活浮点；用少量角度枚举展示局部校准目标。'],
    '枚举代替多轮成对变换学习；省略非对称块重建、第二阶段权重与量化参数微调，以及融合 kernel 和真实推理测时。')

add('vptq', '12', 'VPTQ：二阶感知向量与残差码本', '12.4', '2409.17066',
    '同一输入列沿输出维做欧氏向量编码，二阶统计用于共享码本初始化与跨列补偿。',
    ['二阶统计指导共享码本初始化', '列内向量按欧氏距离选码字', '可选残差码字后固定当前列', '按输入 Hessian 补偿剩余列'],
    '''
    import numpy as np
    W = np.array([[.85, .4], [-.32, .6]])
    H = np.array([[2., 1.], [1., 2.]])  # Input-column Hessian, not an output-vector metric.
    M = np.linalg.inv(H)
    vector = W[:, 0]
    main = np.array([[0., 0.], [1., 0.], [.65, -.32], [1., -1.]])
    residual = np.array([[0., 0.], [.2, .05], [-.15, -.25]])
    cost = lambda candidates: np.sum((candidates-vector)**2, axis=-1)
    first = int(np.argmin(cost(main)))
    second = int(np.argmin(cost(main[first]+residual)))
    assert (first, second) == (2, 1)  # Euclidean search within one input column.
    restored = main[first]+residual[second]
    e = restored-vector
    delta = np.outer(e, M[0, :]/M[0, 0])
    compensated_loss = .5*np.trace(delta@H@delta.T)
    no_compensation = np.column_stack([e, np.zeros_like(e)])
    assert np.allclose((W+delta)[:, 0], restored)
    assert np.isclose(compensated_loss, (e@e)/(2*M[0, 0]))
    assert compensated_loss < .5*np.trace(no_compensation@H@no_compensation.T)
    assert cost(restored) <= cost(main[first])
    print('main/residual indices:', first, second)
    print('column error and compensated loss:', e, compensated_loss)
    ''',
    ['向量来自同一输入列的多个输出行，码字搜索用欧氏距离；H 描述两个输入列的相关性。',
     'delta 的第一列精确满足码字约束，第二列连续补偿；独立比较二次型与受约束最小损失公式。'],
    '固定小码本，仅演示一列与下一列的连续补偿；未实现共享码本训练、完整顺序量化、离群码本或模型微调。')

add('qtip', '12', 'QTIP：有状态 trellis 量化', '12.5', '2406.11235',
    '通过状态转移编码长向量，避免把码本大小与量化维度指数绑定。bitshift trellis 的后续码字受当前状态约束，编码必须搜索整条路径。',
    ['每个状态对应一个重建值', '输入 bit 决定下一状态', '动态规划保留最低代价路径', '按状态序列解码长向量'],
    '''
    import itertools
    import numpy as np
    values = np.array([-.8, -.1, .9, .2, -.4])
    codebook = np.array([-1., -.25, .25, 1.])
    paths = {0: (0., [])}
    for value in values:
        next_paths = {}
        for state, (cost, bits) in paths.items():
            for bit in (0, 1):
                nxt = ((state << 1) | bit) & 3
                candidate = (cost+(codebook[nxt]-value)**2, bits+[bit])
                if nxt not in next_paths or candidate[0] < next_paths[nxt][0]:
                    next_paths[nxt] = candidate
        paths = next_paths
    best_cost, best_bits = min(paths.values(), key=lambda item: item[0])
    def decode(bits):
        state = 0; output = []
        for bit in bits:
            state = ((state << 1) | bit) & 3; output.append(codebook[state])
        return np.array(output)
    brute = min(np.sum((decode(b)-values)**2) for b in itertools.product((0, 1), repeat=len(values)))
    assert np.isclose(best_cost, brute)
    print('bit path, decoded values, SSE:', best_bits, decode(best_bits), best_cost)
    ''',
    ['一个新 bit 同时受旧状态约束，不能对每个元素独立选最近码字；动态规划利用相同尾状态合并路径。',
     '对五步路径穷举提供独立最优性检查，展示 trellis 如何在小状态空间里处理更长序列。'],
    '四状态、固定起点的教学 trellis；未实现 QTIP 的非相干预处理、尾咬合约束、Gaussian 码构造及 GPU 解码。')

add('higgs', '12', 'HIGGS：Hadamard 与高斯 MSE 码本', '12.6', '2411.17525',
    '将权重旋转并归一化，使码本设计可近似转为标准高斯的 MSE 优化问题。理论连接依赖小扰动等条件，不能把低权重误差等同于任意任务质量保证。',
    ['预计算高斯 MSE 码本', '按组做 RMS 归一化', '随机 Hadamard 后查表', '乘回尺度并恢复坐标'],
    '''
    import numpy as np
    from experiments.quant_lab import hadamard
    from experiments.technique_kernels import lloyd, nearest
    grid = np.linspace(-5, 5, 2001)
    centers, _ = lloyd(grid, 16, importance=np.exp(-grid**2/2))
    w = np.array([4., -.1, .2, .3, -.2, .1, .4, -.3])
    R = np.diag([1, -1, 1, -1, 1, 1, -1, 1])@hadamard(8)
    scale = np.sqrt(np.mean(w*w))
    normalized = (w@R)/scale
    quantized = nearest(normalized, centers)
    restored = scale*(quantized@R.T)
    assert np.isclose(np.mean(normalized**2), 1.)
    assert np.allclose(scale*(normalized@R.T), w)
    print('Gaussian grid and weight MSE:', centers, np.mean((restored-w)**2))
    ''',
    ['使用 RMS 而非 L2 范数直接归一化，使旋转后每维的平均平方幅度为 1；两种写法相差 sqrt(group_size)。',
     '码本在一个截断高斯积分网格上预计算，没有读取模型校准激活。'],
    '使用一维离散 Lloyd 近似展示流程；完整 HIGGS 使用多维 MSE 网格、特定内核及可选非均匀位宽分配。')

add('slim_llm', '12', 'SliM-LLM：显著性驱动的组级位宽', '12.7', '2405.14917',
    '根据权重显著性的组间差异分配位宽，并用显著性加权校准保护组内重要元素。等大小组中，高低位宽成对调整可保持平均 payload 位宽。',
    ['以逆 Hessian 的 Cholesky 因子评分', '高显著组提高一位', '低显著组降低一位', '校准量化器并核查总开销'],
    '''
    import numpy as np
    W = np.array([[1., 1., .4, .4, .1, .1, .05, .05]])
    H = np.diag([1., 1., 4., 4., 1., 1., 1., 1.])
    H_reg = H + .01*np.eye(8)  # Absolute damping in the pinned get_salience method.
    M = np.linalg.inv(H_reg)
    U = np.linalg.cholesky(M).T
    assert np.allclose(U.T@U, M)
    scores = W**2 / np.diag(U)[None, :]**2
    # Independent scalar check, valid here because H_reg is diagonal.
    assert np.allclose(scores, W**2*np.diag(H_reg)[None, :])
    salience = scores.reshape(4, 2).sum(axis=1)
    paper_salience = (W**2/np.diag(M)[None, :]**2).reshape(4, 2).sum(axis=1)
    assert salience.argmax() == 0 and paper_salience.argmax() == 1
    order = np.argsort(salience)
    bits = np.full(4, 2)
    bits[order[0]] = 1; bits[order[-1]] = 3
    assert np.array_equal(bits, [3, 2, 2, 1]) and bits.mean() == 2
    print('literal paper / implementation salience:', paper_salience, salience)
    print('group salience, assigned bits:', salience, bits)
    print('payload bits:', np.sum(bits*2))
    ''',
    ['M 是阻尼后 Hessian 的逆，U 才是作者代码变量 Hinv 实际保存的上三角 Cholesky 因子；本例按 w²/Ujj² 评分并按组求和。',
     '对角 H 的独立标量计算核对实现分数；若将 U 的对角误换成 M 的对角，最高显著组从第 1 组变为第 2 组。',
     '一组升到 3 bit、一组降到 1 bit，其余保持 2 bit；此等数量约束只在组大小一致时保持相同 payload。'],
    '按固定作者版本演示 SBA 评分与预算约束，并对照论文正文的字面公式；未实现候选数量搜索、SQC、1-bit 编码和实际打包。')

add('dash_q', '13', 'DASH-Q：稳定对角曲率与加权回归', '13.1', '2604.13806',
    '用输入平方统计构成对角敏感性，减少有限校准数据下交叉曲率噪声。关键不是只删除 Hessian 非对角项，还要迭代优化共享尺度和浮点 offset。',
    ['统计输入对角二阶矩', '固定尺度选择整数编码', '加权回归更新尺度与偏移', '迭代并验证重建目标'],
    '''
    import numpy as np
    w = np.array([-.9, -.2, .12, .43, .8]); h = np.array([1., 8., 8., 3., 1.])
    ridge = .01; s = np.ptp(w)/3; z = -w.min()
    for step in range(8):
        q = np.clip(np.rint((w+z)/s), 0, 3)
        mq, mw = np.average(q, weights=h), np.average(w, weights=h)
        s = np.sum(h*(q-mq)*(w-mw))/(np.sum(h*(q-mq)**2)+ridge)
        z = s*mq-mw
    error = s*q-z-w
    assert abs(h@error) < 1e-10
    assert abs((h*error)@q+ridge*s) < 1e-10
    print('scale, floating offset, weighted SSE:', s, z, np.sum(h*error**2))
    ''',
    ['本例重建为 s*q-z，z 是实值 offset，不是前文整数 zero point；两者不能混用。',
     '固定 q 的两个一阶驻点条件独立核对回归解。只固定独立网格再用对角权重，并不会改变逐元素 RTN。'],
    '一组 2-bit 权重的回归子问题；未实现整模型校准、分组导出、停止准则和论文部署后端。')

add('arcquant', '13', 'ARCQuant：扩展激活残差通道', '13.2', '2601.07475',
    '在线量化难以表示的激活残差，并复制相应权重列，把补偿并入同一个增强 GEMM。增加的通道仍采用 NVFP4，而非额外高精度矩阵乘。',
    ['主激活与权重量化', '选中通道计算激活残差', '拼接残差与复制的权重列', '一次增强矩阵乘合并输出'],
    '''
    import numpy as np
    from experiments.technique_kernels import micro_fp4
    rng = np.random.default_rng(1302)
    X = rng.normal(size=(4, 32)); X[:, :16] *= 5
    W = rng.normal(size=(3, 32))
    Xq, _ = micro_fp4(X); Wq, _ = micro_fp4(W)
    selected = np.arange(16)
    residual = X[:, selected]-Xq[:, selected]
    Rq, _ = micro_fp4(residual)
    augmented_X = np.concatenate([Xq, Rq], axis=1)
    augmented_W = np.concatenate([Wq, Wq[:, selected]], axis=1)
    merged = augmented_X@augmented_W.T
    separate = Xq@Wq.T+Rq@Wq[:, selected].T
    assert np.allclose(merged, separate)
    print('reduction dimension before/after:', X.shape[1], augmented_X.shape[1])
    print('corrected output MSE:', np.mean((merged-X@W.T)**2))
    ''',
    ['残差来自激活 X-Xq；权重分支复制已量化的 Wq 对应列，不是再计算一份权重残差。',
     '选择 16 个通道让新增尾块满足微块尺寸；等价性断言核对拼接矩阵乘与两项输出之和。'],
    '预先指定补偿通道，省略论文阈值选择与重排；micro_fp4 是 CPU 数值模拟，无法测出统一 GEMM 的实际加速。')

add('torq', '13', 'TORQ：块间均衡与块内码本对齐', '13.3', '2605.19561',
    '宏观上在相同块内位置之间做跨块旋转以均衡方差，微观上搜索块内旋转改善 FP4 码字占用。两层目标不同，不能用一次任意 Hadamard 代替。',
    ['估计跨块位置协方差', '旋转均衡块间二阶矩', '块内搜索码字对齐', '与权重配对并量化'],
    '''
    import numpy as np
    from experiments.quant_lab import hadamard
    rng = np.random.default_rng(1303)
    X = rng.normal(size=(64, 8)); X[:, :4] *= 5
    W = rng.normal(size=(3, 8)); T = np.eye(8)
    for position in range(4):
        columns = [position, position+4]
        covariance = X[:, columns].T@X[:, columns]/len(X)
        _, U = np.linalg.eigh(covariance)
        R = U@hadamard(2)
        T[np.ix_(columns, columns)] = R
        assert np.allclose(np.diag(R.T@covariance@R), np.trace(covariance)/2)
    assert np.allclose(T.T@T, np.eye(8))
    assert np.allclose((X@T)@(W@T).T, X@W.T)
    print('inter-block covariance diagonals equalized at all four positions')
    ''',
    ['每次只旋转两个块中同一位置的通道；不是将整个块任意混成一个量化组。',
     '谱分解加 Hadamard 是一个清楚的等方差构造，用于核对宏观目标；各位置互不重叠，因此可组成一个正交 T。'],
    '演示宏观均衡子问题，未实现论文的 Givens 构造或微观占用率搜索；FP4 下方差均衡是代理目标，不能声称严格最优失真。')

add('soar', '13', 'SOAR：闭式尺度优化与解耦搜索', '13.4', '2605.12245',
    '固定 FP4 码字时，全局尺度和局部尺度都有最小二乘更新；随后投影局部尺度到 E4M3，再结合编码侧与解码侧的尺度搜索。',
    ['固定当前 FP4 编码', '闭式更新全局尺度', '局部尺度投影到 E4M3', '重算码字并联合搜索'],
    '''
    import numpy as np
    from experiments.technique_kernels import nearest, fp_levels
    W = np.linspace(-1, 1, 32).reshape(2, 16)
    block_scale = np.array([[.25], [.5]])
    q = nearest(W/block_scale, fp_levels('e2m1'))
    U = block_scale*q
    alpha = np.sum(W*U)/np.sum(U*U)
    before = np.sum((W-U)**2)
    after = np.sum((W-alpha*U)**2)
    assert abs(np.sum((alpha*U-W)*U)) < 1e-12
    assert after <= before+1e-12
    assert np.all(np.isin(block_scale, fp_levels('e4m3')))
    print('closed-form global scale, SSE before/after:', alpha, before, after)
    ''',
    ['对固定局部尺度和码字，alpha 是一个实数最小二乘系数；驻点条件和误差比较共同验证更新。',
     '局部尺度已经是 E4M3 数值，alpha 则是独立的高精度尺度，二者的可用值集合不同。'],
    '只实现 CJSO 的全局尺度子步；完整 SOAR 还重算码字、投影局部尺度，并执行 DSS 联合候选搜索。')

add('focus', '13', 'FOCUS：编码与解码尺度解耦', '13.5', '2608.01847',
    '编码时决定 FP4 索引的尺度可以是高精度且更细粒度的，而实际保存的解码尺度仍服从硬件格式。训练完成后丢弃编码侧系数，保留标准 FP4 表示。',
    ['固定硬件解码尺度', '子块使用高精度编码系数', '按输出误差选择 FP4 码字', '丢弃系数，保存标准格式'],
    '''
    import itertools
    import numpy as np
    from experiments.technique_kernels import nearest, fp_levels
    rng = np.random.default_rng(1305)
    W = rng.normal(size=16)*.3
    X = rng.normal(size=(8, 16)); X[:, 1] = X[:, 0]
    decode_scale = .125; candidates = []
    for c0, c1 in itertools.product([.8, 1., 1.2], repeat=2):
        encoding_scale = decode_scale*np.repeat([c0, c1], 8)
        q = nearest(W/encoding_scale, fp_levels('e2m1'))
        restored = decode_scale*q
        candidates.append((np.mean((X@(restored-W))**2), (c0, c1)))
    best = min(candidates)
    baseline = next(loss for loss, pair in candidates if pair == (1., 1.))
    assert best[0] <= baseline and decode_scale in fp_levels('e4m3')
    print('chosen subblock coefficients and output MSE:', best[1], best[0])
    ''',
    ['同一个 16 元素块共享解码尺度，但两个 8 元素子块有独立编码系数，展示 DGS。',
     '目标是校准输出误差；给定固定重建格点时，独立 RTN 已最小化纯权重 MSE，不能用更差的舍入宣称普遍降低权重误差。'],
    '有限候选搜索代替端到端尺度学习，只展示 CRS/DGS 自由度；不实现整模型训练和导出。')

add('prquant', '13', 'PRQuant：离线权重残差与连续尾块', '13.6', '2609.22106',
    '把量化残差较大的权重输入列重排到连续尾部，并离线准备尾部残差。推理时复用连续激活切片，避免分散 gather，再通过增强 GEMM 合并补偿。',
    ['衡量缩放后权重列误差', '敏感列重排到连续尾部', '离线保存尾部权重残差', '重复尾部激活并合并 GEMM'],
    '''
    import numpy as np
    from experiments.quant_lab import group_qdq
    rng = np.random.default_rng(1306)
    X = rng.normal(size=(5, 8)); W = rng.normal(size=(3, 8))
    Q = group_qdq(W, group_size=4); E = W-Q
    ranked = np.argsort(np.sum(E**2, axis=0))
    Xp, Qp, Ep = X[:, ranked], Q[:, ranked], E[:, ranked]
    tail = 2
    Xa = np.concatenate([Xp, Xp[:, -tail:]], axis=1)
    Wa = np.concatenate([Qp, Ep[:, -tail:]], axis=1)
    output = Xa@Wa.T
    expected = X@Q.T+Xp[:, -tail:]@Ep[:, -tail:].T
    assert np.allclose(output, expected)
    print('contiguous tail columns:', ranked[-tail:])
    print('augmented reduction dimension:', Xa.shape[1])
    ''',
    ['与 ARCQuant 相反，本例补的是权重残差，并复制激活尾部；残差不需要每个 token 在线重新计算。',
     '排列本身保持线性运算，主要价值还包括使尾部访问和执行布局规整；不能把它单独当成误差下降的来源。'],
    '使用 INT4 数值代理验证增强乘法，残差保留浮点；真实 PRQuant 还使用缩放、MXFP4 残差编码和融合内核。')

add('osfp4', '13', 'OSFP4：平滑与块尺度联合优化', '13.7', '2610.08231',
    '针对 NVFP4 的矩阵乘误差联合选择对角平滑与块尺度，并区别 RTN 和 successive interference cancellation。随机乘性 dither 用于分析，部署仍落到确定性硬件码字。',
    ['采集矩阵乘二阶统计', '联合平滑与尺度目标', '搜索合法 E4M3 块尺度', '按 RTN 或 SIC 导出'],
    '''
    import numpy as np
    from experiments.technique_kernels import micro_fp4
    rng = np.random.default_rng(1307)
    X = rng.normal(size=(8, 32)); X[:, 0] *= 8
    W = rng.normal(size=(4, 32))
    target = X@W.T; losses = []
    for exponent in [0., .5, 1.]:
        s = np.maximum(np.max(abs(X), axis=0), 1e-6)**exponent
        assert np.allclose((X/s)@(W*s).T, target)
        Xq, _ = micro_fp4(X/s)
        for multiplier in [.8, 1., 1.2]:
            alpha = multiplier*np.max(abs(W*s))/(448*6)
            Wq, _ = micro_fp4(W*s, tensor_scale=alpha)
            losses.append((np.mean((Xq@Wq.T-target)**2), exponent, multiplier))
    print('best limited joint candidate:', min(losses))
    assert np.isfinite(min(losses)[0])
    ''',
    ['同时改变 s 和量化尺度，而不是先固定一个平滑方案再仅调整名称为 NVFP4。',
     '量化前先核对等价变换；量化后的目标用完整矩阵乘输出比较。'],
    '少量平滑与全局尺度候选仅解释联合目标；未实现论文的乘性 dither 分析、分块优化、精细 E4M3 搜索或 SIC。')

add('drq', '13', 'DRQ：固定量化网格上的分布鲁棒修正', '13.8', '2610.11226',
    '在校准损失外考虑输入分布变化造成的最坏方向误差，再离线修改既有整数码。位宽、分组、尺度和推理算子保持原表示，因此重点是改进离线目标。',
    ['构造校准二阶矩 H', '限定分布变化的预算', '寻找最坏方向 u', '在固定格点上修正整数码'],
    '''
    import numpy as np
    H = np.array([[1., .8], [.8, 1.]])
    C = .8*H+.2*np.eye(2); alpha = .3
    E = np.array([[.4, -.6], [.1, .2]])
    values, U = np.linalg.eigh(C)
    inverse_root = (U*(1/np.sqrt(values)))@U.T
    _, singular, Vt = np.linalg.svd(E@inverse_root)
    u = inverse_root@Vt[0]
    Delta = alpha*np.outer(u, u)
    baseline = np.trace(E@H@E.T)
    robust = baseline+alpha*singular[0]**2
    assert np.isclose(np.trace(C@Delta), alpha)
    assert np.isclose(np.trace(E@(H+Delta)@E.T), robust)
    print('calibration loss and attained worst-case loss:', baseline, robust)
    ''',
    ['约束 tr(C Delta)≤alpha 下，最坏增量由一个秩一协方差方向达到；这里验证它与谱范数闭式目标相等。',
     'C=(1-beta)H+beta I 保证正定，避免未被校准激活覆盖的方向完全不受约束。'],
    '验证 DRQ 的最坏分布子问题；未执行码值更新排序、检查集校准损失增长限制或整模型离线细化。')

add('differentiable_bits', '13', 'DBW：可微位宽与零位剪枝', '13.9', '2610.06026',
    '把 SVD 分量的位宽作为可学习预算分配变量，允许 0 bit 表示删除分量。归一化维持总预算，离散化的余量通过 Hamilton 最大余数规则补齐。',
    ['SVD 分量获得非负预算权重', '归一化并分配连续位宽', '向下取整后按余数补齐', '0 bit 分量被剪除'],
    '''
    import numpy as np
    beta = np.array([2.5, 1., .1, 0.]); budget = 8
    positive = np.maximum(beta, 0)
    continuous = budget*positive/positive.sum()
    bits = np.floor(continuous).astype(int)
    leftover = budget-int(bits.sum())
    order = np.argsort(-(continuous-bits), kind='stable')
    bits[order[:leftover]] += 1
    W = np.diag([4., 1., .2, .01])
    U, singular, Vt = np.linalg.svd(W)
    kept = bits > 0
    pruned = (U[:, kept]*singular[kept])@Vt[kept]
    assert bits.sum() == budget and np.any(bits == 0)
    assert np.linalg.matrix_rank(pruned) == kept.sum()
    print('continuous/integer component bits:', continuous, bits)
    ''',
    ['最大余数法避免所有分量各自向下取整后浪费预算；分量尺寸相同时位宽总和可代表统一成本单位。',
     '0 bit 决定该分量不再参与重建；位宽分配和秩截断在同一个决策空间里。'],
    '仅实现预算离散化和零位掩码，保留分量未做实际量化；省略 STE 位宽训练、奇异值熵损失、双层预算和 mixed-precision GEMM。')

add('align_correct', '13', 'Align, Then Correct：对齐后重新截断补偿', '13.10', '2610.08164',
    '先用非对称、Fisher 加权目标建立低秩补偿，再在已补偿模型上重新测量梯度和曲率。第二阶段重新截断整个更新后的适配器，维持最终秩预算。',
    ['教师与量化输入非对称对齐', '加权 SVD 得到低秩适配器', '重新采集梯度与曲率', '自然梯度后整体截断到原秩'],
    '''
    import numpy as np
    rng = np.random.default_rng(1310)
    Xq, _ = np.linalg.qr(rng.normal(size=(8, 4)))
    Xf = Xq+.05*rng.normal(size=Xq.shape)
    W = rng.normal(size=(3, 4)); Wq = np.round(W*2)/2
    target = Xf@W.T-Xq@Wq.T
    def rank_one(matrix):
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
        return s[0]*np.outer(U[:, 0], Vt[0])
    C1 = rank_one(Xq.T@target)  # Identity input metric and output Fisher here.
    gradient = .02*rng.normal(size=C1.shape)  # A given re-measured gradient.
    C2 = rank_one(C1-gradient)
    naive = C1+rank_one(-gradient)
    assert np.linalg.matrix_rank(C2, tol=1e-10) == 1
    assert np.linalg.matrix_rank(naive, tol=1e-10) == 2
    assert np.sum((Xq@C1-target)**2) <= np.sum(target**2)+1e-12
    print('final rank versus separate-truncation rank:', np.linalg.matrix_rank(C2), np.linalg.matrix_rank(naive))
    ''',
    ['先构造来自 Xf 与 Xq 的真实非对称残差；Xq 正交列使输入度量为单位阵，简化了加权回归。',
     '关键是 rank_r(C1-step)，不是 C1+rank_r(-step)；后者可能使最终适配器秩翻倍。'],
    '省略非单位 Fisher 的白化、信赖比例缩放和真实反向统计采集；gradient 是给定示例，不能据此宣称训练免费或实际任务恢复。')

add('quest', '14', 'QuEST：高斯拟合与信任梯度', '14.1', '2502.05003',
    '前向在 Hadamard 域进行 RMS 归一化与高斯 MSE 拟合，反向仅信任量化误差处于阈值内的分量，再把梯度变回原坐标。',
    ['Hadamard 与 RMS 归一化', '高斯 MSE 决定量化范围', '量化误差形成 trust mask', '掩码梯度逆变换回主权重'],
    '''
    import numpy as np
    from experiments.quant_lab import hadamard
    from experiments.technique_kernels import nearest
    grid = np.linspace(-5, 5, 1001); density = np.exp(-grid**2/2)
    alpha = min(np.linspace(.5, 3., 80), key=lambda a: np.sum(density*(nearest(grid, np.linspace(-a, a, 4))-grid)**2))
    R = hadamard(8)
    x = np.array([3., -.7, .2, .1, -.2, .4, -.6, .8])@R.T
    xh = x@R; rms = np.sqrt(np.mean(xh*xh)); normalized = xh/rms
    q = nearest(normalized, np.linspace(-alpha, alpha, 4))
    mask = abs(q-normalized) <= alpha/3+1e-12
    grad_h = np.arange(1., 9.); grad_x = (mask*grad_h)@R.T
    direction = np.arange(8.)/8
    assert np.any(mask) and np.any(~mask)
    assert np.isclose(grad_x@direction, (mask*grad_h)@(direction@R))
    print('Gaussian-fit range and trusted coordinates:', alpha, mask)
    ''',
    ['四级均匀网格的半步长为 alpha/3；阈值在归一化域比较，不能混用原值与归一化值的单位。',
     '掩码作用在量化发生的 Hadamard 域，随后通过逆变换传回梯度；不是永久冻结同名原始通道。'],
    '只展示一次前向拟合和信任梯度传播；没有训练完整 LLM、优化稀疏扩展或测试 GPU kernel。')

add('paretoq', '14', 'ParetoQ：位宽、网格与训练预算一起比较', '14.2', '2502.02631',
    '公平比较极低位宽时，必须共同控制量化器、模型规模和训练预算。不同位宽适合的网格不同，尤其要区分对称四级网格与包含零的整数网格。',
    ['固定比较的模型与训练预算', '按位宽选择量化网格', '学习尺度并进行 QAT', '比较质量、存储和实际速度'],
    '''
    import numpy as np
    from experiments.technique_kernels import nearest
    w = np.array([-.9, -.4, -.1, .05, .3, .8])
    symmetric_four = np.linspace(-1., 1., 4)
    zero_integer_four = np.array([-2., -1., 0., 1.])
    assert len(symmetric_four) == len(zero_integer_four) == 4
    assert 0. not in symmetric_four and 0. in zero_integer_four
    scale = 1.
    for _ in range(8):
        q = nearest(w/scale, symmetric_four)
        scale = (w@q)/(q@q)
    assert abs((scale*q-w)@q) < 1e-12
    print('two four-level geometries:', symmetric_four, zero_integer_four)
    print('learned representative scale:', scale)
    ''',
    ['偶数个等间距、严格对称的代表值无法同时包含零；这会影响极低位宽量化器的设计。',
     '固定码字后的尺度回归说明 scale 也是自由度，而位宽本身不能决定结果优劣。'],
    '仅示范网格族和单组尺度优化，不是 ParetoQ 的完整 SEQ/LSQ 梯度或训练调度；论文的最优位宽结论依赖其预算与模型设置。')

add('onlineqat', '14', 'OnlineQAT：在学生生成的前缀上蒸馏', '14.3', '2610.09346',
    '先获得可用的低比特初始化，再由量化学生生成回答；教师在学生实际访问的前缀上给出概率，用采样 reverse-KL 信号修复离线数据未覆盖的状态。',
    ['块级 QAT 得到初始学生', '学生采样回答与前缀', '教师计算同一前缀概率', '采样 reverse-KL 更新学生'],
    '''
    import numpy as np
    from experiments.quant_lab import softmax, symmetric_qdq
    rng = np.random.default_rng(1403)
    logits = np.array([1.1, -.3, .2])
    student = softmax(symmetric_qdq(logits, bits=3)[0])
    teacher = softmax(np.array([.8, .2, -.1]))
    actions = rng.choice(3, size=100000, p=student)
    log_ratio = np.log(student/teacher)
    advantage = log_ratio[actions]  # Treated as stop-gradient in the estimator.
    score = np.eye(3)[actions]-student
    estimated = np.mean(advantage[:, None]*score, axis=0)
    exact = student*(log_ratio-student@log_ratio)
    assert np.allclose(estimated, exact, atol=.01)
    print('sampled/exact reverse-KL logit gradients:', estimated, exact)
    ''',
    ['动作按学生分布采样，log_ratio 作为停止梯度的权重；若从教师采样，优化的估计器就不同。',
     '用单前缀上的解析 reverse-KL 梯度核对蒙特卡洛估计，不把采样损失值当作真实 KL。'],
    '只验证固定前缀的梯度估计；未实现长序列 rollout、块级初始化、策略更新或训练稳定性。')

add('eaquant', '14', 'EAQuant：专家共享平滑与路由校准', '14.4', '2506.13329',
    'MoE 的多个专家共享上游归一化，不能分别把任意尺度融合到同一输入。EAQuant 聚合专家和路由器的平滑需求，并校准路由一致性与专家数据覆盖。',
    ['统计专家与路由器尺度需求', '聚合为共享输入尺度', '联合校准路由与专家', '补充低激活专家的校准样本'],
    '''
    import numpy as np
    rng = np.random.default_rng(1404)
    X = rng.normal(size=(12, 4)); X[:8, 0] = 2.; X[8:, 0] = -1.
    gate = np.array([[1., 0, 0, 0], [-1., 0, 0, 0]])
    experts = rng.normal(size=(2, 3, 4))
    route = np.argmax(X@gate.T, axis=1); requirements = []
    for e in range(2):
        a = np.max(abs(X[route == e]), axis=0)
        b = np.maximum(np.max(abs(experts[e]), axis=0), 1e-6)
        requirements.append(np.sqrt(a/b))
    requirements.append(np.sqrt(np.max(abs(X), axis=0)/np.maximum(np.max(abs(gate), axis=0), 1e-6)))
    shared = np.max(requirements, axis=0)
    assert np.allclose((X/shared)@(gate*shared).T, X@gate.T)
    for W in experts:
        assert np.allclose((X/shared)@(W*shared).T, X@W.T)
    print('routed sample counts:', np.bincount(route, minlength=2))
    print('shared smoothing scale:', shared)
    ''',
    ['聚合必须包含路由器，否则共享输入变换可能改变路由 logits；等价性在量化前分别核对。',
     '每个专家收到的样本数不同，覆盖不足会影响统计；论文通过继续采样补齐稀少专家，而不是随意改变推理路由。'],
    '只演示共享平滑和覆盖检测，未实现路由 KL-Top/MSE 优化、数据补采集及完整 MoE 激活量化。')

add('milo', '14', 'MiLo：MoE 的低秩补偿组合', '14.5', '2504.02658',
    '为低比特 MoE 权重增加不同秩的补偿器，并交替优化量化表示与低秩残差。秩也消耗内存，应按矩阵结构和收益分配，不能全部使用相同高秩。',
    ['分别量化各专家权重', '计算专家残差的奇异值谱', '按预算选择不同补偿秩', '合并专家低比特与低秩输出'],
    '''
    import numpy as np
    from experiments.quant_lab import symmetric_qdq
    rng = np.random.default_rng(1405)
    experts = rng.normal(size=(2, 6, 8)); ranks = [1, 3]
    extra_parameters = 0
    for W, rank in zip(experts, ranks):
        Wq = symmetric_qdq(W, bits=3, axis=1)[0]
        E = W-Wq
        U, singular, Vt = np.linalg.svd(E, full_matrices=False)
        correction = (U[:, :rank]*singular[:rank])@Vt[:rank]
        after = np.sum((E-correction)**2)
        assert np.isclose(after, np.sum(singular[rank:]**2))
        extra_parameters += rank*sum(W.shape)
        print('rank and residual SSE:', rank, after)
    print('additional low-rank parameters:', extra_parameters)
    ''',
    ['不同专家可以分配不同的低秩容量；截断 SVD 的剩余能量给出此残差子问题的独立检查。',
     'rank*(输出维+输入维) 是两因子额外参数量，必须与量化主体一同计入存储。'],
    '使用 RTN 代替 HQQ 并预设秩，补偿因子保持浮点；省略因子的 INT3 量化、秩策略和专用推理内核。')

add('bitsmoe', '14', 'BitsMoE：共享谱基与成本感知位宽', '14.6', '2606.00079',
    '把专家权重拆成共享输入谱基与专家特有分量，将谱强度、路由加权激活重要性和位宽失真结合，再在统一内存预算下进行离散分配。',
    ['跨专家 SVD 提取共享基', '计算谱强度和路由重要性', '测量各位宽分量失真', '预算内分配谱分量位宽'],
    '''
    import itertools
    import numpy as np
    from experiments.quant_lab import symmetric_qdq
    rng = np.random.default_rng(1406)
    W = rng.normal(size=(2, 2, 3)); X = rng.normal(size=(12, 3))
    U, singular, Vt = np.linalg.svd(W.reshape(4, 3), full_matrices=False)
    P = (U*singular).reshape(2, 2, 3); Phi = Vt.T
    assert np.allclose(P@Phi.T, W)
    gates = np.tile([.8, .2], (len(X), 1)); costs = {}
    for e in range(2):
        for k in range(3):
            magnitude = np.linalg.norm(P[e, :, k]); direction = P[e, :, k]/magnitude
            importance = np.mean(gates[:, e]*(X@Phi[:, k])**2)
            for bits in (2, 4):
                error = np.sum((direction-symmetric_qdq(direction, bits=bits)[0])**2)
                costs[e, k, bits] = magnitude**2*importance*error
    candidates = []
    for bits in itertools.product((2, 4), repeat=6):
        if sum(bits) <= 18:
            candidates.append((sum(costs[i//3, i%3, b] for i, b in enumerate(bits)), bits))
    best = min(candidates)
    assert sum(best[1]) <= 18
    print('component bits and factorized cost:', best[1], best[0])
    ''',
    ['共享 Phi 只保存一份；P 的列范数、激活沿 Phi 的投影和实际量化误差组成三因子成本。',
     '这里所有专家方向长度相同，位宽总和可表示可变成本；实际优化还需加入共享基、尺度及元数据的固定成本。'],
    '用六分量穷举代替 ILP，共享基保持浮点；省略共享基 GPTQ、低位宽经验系数、误差交叉项和 GPU 执行。')

add('polarquant_recursive', '15', 'PolarQuant：递归极坐标编码', '15.1', '2502.02617',
    '随机预条件后递归合并二维向量的半径，把一个方向表示成一棵角度树。不同层角度具有不同分布，量化角度后仍需保留或恢复整体范数。',
    ['随机预条件后两两配对', '保存角度，递归合并半径', '按层量化角度', '从根范数逐层恢复坐标'],
    '''
    import numpy as np
    x = np.array([1., -.3, .2, .7, -.5, .4, .8, -.1])
    radii = x.copy(); exact_angles = []; quantized_angles = []
    while len(radii) > 1:
        pairs = radii.reshape(-1, 2)
        angles = np.arctan2(pairs[:, 1], pairs[:, 0])
        lower, upper = (-np.pi, np.pi) if not exact_angles else (0., np.pi/2)
        step = (upper-lower)/16
        codes = np.clip(np.floor((angles-lower)/step), 0, 15)
        exact_angles.append(angles); quantized_angles.append(lower+(codes+.5)*step)
        radii = np.linalg.norm(pairs, axis=1)
    def decode(levels):
        values = radii.copy()
        for angles in reversed(levels):
            values = np.column_stack([values*np.cos(angles), values*np.sin(angles)]).ravel()
        return values
    exact, approximate = decode(exact_angles), decode(quantized_angles)
    assert np.allclose(exact, x)
    assert np.isclose(np.linalg.norm(approximate), np.linalg.norm(x))
    assert sum(len(a) for a in exact_angles) == len(x)-1
    print('angle count, root radius, MSE:', len(x)-1, radii[0], np.mean((approximate-x)**2))
    ''',
    ['叶层角度允许正负坐标，范围是完整圆；更高层合并非负半径，角度范围变为第一象限。',
     '8 维向量只需 7 个树角度和一个根范数。量化角度保持重建范数，不表示方向无误差。'],
    '省略随机预条件和按理论密度设计的分层码本，均匀角度仅用于解释递归结构；不是另一篇 RoPE 配对的同名 PolarQuant。')

add('polarquant_rope', '15', 'PolarQuant：RoPE 配对与 QK 查表', '15.2', '2502.00527',
    '将 post-RoPE Key 的配对维度编码成半径和角度，并针对当前 Query 构建 QK 查找表。有限极坐标组合可以复用，而不必对每个历史 Key 重复做完整反量化乘法。',
    ['post-RoPE 维度两两配对', '分别编码半径与角度', '当前 Query 建立二维 LUT', '历史 Key 索引直接查 QK'],
    '''
    import numpy as np
    K = np.array([[1., .2], [-.4, .8], [.1, -.3]])
    query = np.array([.7, -.2])
    radius_grid = np.array([0., .5, 1., 2.])
    angle_grid = np.arange(8)*2*np.pi/8
    radii = np.linalg.norm(K, axis=1); angles = np.arctan2(K[:, 1], K[:, 0]) % (2*np.pi)
    ri = np.argmin(abs(radii[:, None]-radius_grid), axis=1)
    ai = np.rint(angles/(2*np.pi)*8).astype(int) % 8
    lut = radius_grid[:, None]*(query[0]*np.cos(angle_grid)+query[1]*np.sin(angle_grid))
    decoded = radius_grid[ri, None]*np.column_stack([np.cos(angle_grid[ai]), np.sin(angle_grid[ai])])
    assert np.allclose(lut[ri, ai], decoded@query)
    print('lookup QK, full-precision QK:', lut[ri, ai], K@query)
    ''',
    ['角度索引用模运算处理圆周边界；半径与角度索引一起决定查表位置。',
     '断言比较查表与解码后内积，证明查表变换本身等价；它们均不同于原始未量化 Key 的内积。'],
    '单个二维配对，码本为手工候选；实际模型必须遵守各自 RoPE 配对布局，量化器校准与 Triton 内核未复现。')

add('oscar_spectral', '15', 'OSCAR：Attention 感知的谱旋转', '15.3', '2605.17757',
    '用 Query 二阶矩指导 Key 旋转，用注意力加权 Value 统计指导 Value 旋转，再结合 Hadamard、bit-reversal、裁剪与低比特缓存布局。',
    ['计算 Query 与加权 Value 统计', '谱基加 Hadamard 和位反转', '固定旋转后做 INT2 量化', '匹配 Query 与输出坐标'],
    '''
    import numpy as np
    from experiments.quant_lab import hadamard, softmax, affine_qdq
    rng = np.random.default_rng(1503)
    Q = rng.normal(size=(3, 4)); K = rng.normal(size=(8, 4)); V = rng.normal(size=(8, 4))
    P = softmax(Q@K.T/2)
    CQ = Q.T@Q/len(Q); CV = V.T@P.T@P@V/len(Q)
    permutation = np.eye(4)[:, [0, 2, 1, 3]]
    def rotation(C):
        _, U = np.linalg.eigh(C)
        return U[:, ::-1]@hadamard(4)@permutation
    RK, RV = rotation(CQ), rotation(CV)
    assert np.allclose((Q@RK)@(K@RK).T, Q@K.T)
    assert np.allclose((P@(V@RV))@RV.T, P@V)
    Kq = affine_qdq(K@RK, bits=2, axis=1)[0]
    Vq = affine_qdq(V@RV, bits=2, axis=1)[0]
    output = (softmax((Q@RK)@Kq.T/2)@Vq)@RV.T
    print('spectral INT2 attention MSE:', np.mean((output-P@V)**2))
    ''',
    ['Key 的敏感性统计来自 Query，而不是只对 K 本身做 PCA；Value 目标还包含注意力矩阵。',
     'Key 旋转要同时作用于 Query，Value 的输出要逆旋转；两条浮点恒等式先分别验证。'],
    '只展示谱目标和数值数据流，省略百分位裁剪、sink/recent 高精度区及分页缓存内核；理论最优性依赖冻结误差等假设。')

add('oscar_token', '15', 'OScaR：旋转后的 token 范数平衡', '15.4', '2605.19660',
    '针对 token 范数差异污染 per-channel 量化的问题，先用 Hadamard 分散通道离群值，再进行 token 范数归一化。它与协方差谱旋转的 OSCAR 是不同论文。',
    ['Hadamard 分散通道能量', '按 token 保存 L2 范数', '归一化后沿通道量化', '乘回范数并恢复坐标'],
    '''
    import numpy as np
    from experiments.quant_lab import hadamard, affine_qdq
    K = np.array([[12., .1, -.2, .3], [1., .2, .1, -.1], [.01, .02, -.01, .01], [0., 0., 0., 0.]])
    R = hadamard(4); rotated = K@R
    norms = np.linalg.norm(rotated, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.)
    normalized = rotated/safe
    assert np.allclose((normalized*safe)@R.T, K)
    coded = affine_qdq(normalized, bits=2, axis=0)[0]
    restored = (coded*norms)@R.T
    assert np.array_equal(restored[-1], np.zeros(4))
    print('token norms:', norms.ravel())
    print('restored cache MSE:', np.mean((restored-K)**2))
    ''',
    ['先旋转再归一化，避免简单放大小范数 token 后产生新的通道伪离群值。',
     '范数属于额外元数据，解码时必须乘回；零 token 单独处理，不能除以零。'],
    '只处理一个 Key 块，省略多模态实现、完整 Value 路径、流式残留区和 CUDA 性能测量。')

add('kvarn', '15', 'KVarN：token 与通道双向方差归一化', '15.5', '2606.03458',
    '同时控制 token 和通道的尺度差异，减少长推理中的量级误差积累。迭代归一化必须保存两侧尺度，且不能把方差统计替换成最大绝对值。',
    ['先做通道 Hadamard', '交替归一化列与行标准差', '量化平衡后的矩阵', '恢复两侧尺度和原坐标'],
    '''
    import numpy as np
    from experiments.quant_lab import hadamard, symmetric_qdq
    rng = np.random.default_rng(1505)
    K = (rng.normal(size=(8, 8))+1)*np.arange(1., 9.)[:, None]
    R = hadamard(8); Z = K@R
    row_scale = np.ones((8, 1)); col_scale = np.ones((1, 8))
    for _ in range(5):
        cols = np.maximum(Z.std(axis=0, keepdims=True), 1e-8)
        Z /= cols; col_scale *= cols
        rows = np.maximum(Z.std(axis=1, keepdims=True), 1e-8)
        Z /= rows; row_scale *= rows
    assert np.allclose((row_scale*Z*col_scale)@R.T, K)
    Q = symmetric_qdq(Z, bits=3)[0]
    reconstructed = (row_scale*Q*col_scale)@R.T
    print('dual-scale cache MSE:', np.mean((reconstructed-K)**2))
    ''',
    ['左右尺度分别在 token 轴与通道轴广播；所有迭代的尺度乘积都要累积。',
     '先验证不量化时可恢复原矩阵，再加入量化，因此误差不会混入漏乘某轮尺度的实现问题。'],
    '直接标准差迭代解释可逆缩放，省略论文对数域稳定更新、实际 KV 分块布局和长推理评估。')

add('attn_qat', '15', 'Attn-QAT：前后向量化状态一致', '15.6', '2603.00040',
    'Attention 的 QAT 不能只在前向加入 FP4。反向重算 P 时要匹配前向量化，同时 softmax 梯度的缩减项需要高精度 P 形成的辅助输出。',
    ['Q/K/P/V 前向伪量化', '反向重算相同量化 P', '另存高精度 P 形成的输出', '按正确 softmax 恒等式求梯度'],
    '''
    import numpy as np
    from experiments.quant_lab import softmax
    from experiments.technique_kernels import micro_fp4
    rng = np.random.default_rng(1506)
    P = softmax(rng.normal(size=(3, 16))); V = rng.normal(size=(16, 8))
    PF, _ = micro_fp4(P); VF, _ = micro_fp4(V.T); VF = VF.T
    output = PF@VF; auxiliary = P@VF
    dO = rng.normal(size=output.shape); dP = dO@VF.T
    direct = P*(dP-np.sum(P*dP, axis=1, keepdims=True))
    via_auxiliary = P*(dP-np.sum(dO*auxiliary, axis=1, keepdims=True))
    wrong = P*(dP-np.sum(dO*output, axis=1, keepdims=True))
    assert np.allclose(direct, via_auxiliary)
    assert not np.allclose(direct, wrong, atol=1e-10)
    assert np.array_equal(PF, micro_fp4(P)[0])
    print('gradient mismatch from using low-precision output:', np.linalg.norm(direct-wrong))
    ''',
    ['O=PF*VF 是量化前向输出，但 softmax 梯度需要 Oprime=P*VF；两者不能混为同一张量。',
     '辅助输出让每行的缩减项由 dO 与 Oprime 内积获得，不必保存完整 P；这里用显式 P 作为参照。'],
    '验证一个关键反向恒等式，未实现 FlashAttention 分块重算、完整反向 kernel 或模型级 QAT。')

add('qserve', '15', 'QServe / QoQ：W4A8KV4 系统协同', '15.7', '2405.04532',
    '先把权重量化为受保护范围的 INT8，再把中间整数按组压缩到 INT4，使解码直接返回整数并参与 INT8 GEMM；量化流程必须与执行内核共同设计。',
    ['权重先量化到保护 INT8 范围', '中间整数再分组量化为 INT4', '寄存器内解回 INT8', '结合 KV4 与融合推理系统'],
    '''
    import numpy as np
    def group_encode(values):
        step = max(1, int(np.rint(np.ptp(values)/15)))
        zero = int(np.clip(np.rint(-values.min()/step), 0, 15))
        codes = np.clip(np.rint(values/step)+zero, 0, 15).astype(int)
        return (codes-zero)*step, codes, step, zero
    unsafe = np.array([-113, 120])
    decoded_bad, _, _, _ = group_encode(unsafe)
    protected = np.clip(unsafe, -119, 119)
    decoded_safe, codes, step, zero = group_encode(protected)
    assert decoded_bad.max() == 128
    assert decoded_safe.min() >= -128 and decoded_safe.max() <= 127
    print('unprotected/protected intermediate INT8:', decoded_bad, decoded_safe)
    print('INT4 codes, integer step, zero point:', codes, step, zero)
    ''',
    ['第二次量化会再次舍入，中间重建值可能从 120 变成 128；不能仅凭最初张量属于 INT8 就假设不会溢出。',
     '保护范围为后续重建留出余量；整数尺度让权重恢复到 INT8，区别于只压缩浮点 scale 的双重量化。'],
    '仅验证渐进式量化和保护范围反例；省略外层 FP16 scale、SmoothAttention、KV4 管理和融合 CUDA 内核。')

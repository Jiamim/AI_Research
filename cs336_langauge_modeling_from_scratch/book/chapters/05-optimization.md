# 05｜优化器、训练稳定性与实验纪律



## 5.1 AdamW 的每一项分别做什么

对于梯度 $g_t$，Adam 维护一阶、二阶指数滑动平均：

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.$$

由于初值为零，早期均值有向零偏差，用 $\hat m_t=m_t/(1-\beta_1^t)$、$\hat v_t=v_t/(1-\beta_2^t)$ 校正。AdamW 更新为

$$\theta_t=(1-\eta_t\lambda)\theta_{t-1}-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$

weight decay 在自适应缩放之外直接衰减参数，与把 $\lambda\theta$ 加进梯度再送入 Adam 不同。后者的正则化项也会进入动量与二阶状态，其作用取决于坐标的梯度历史。

这里的 $\eta_t$ 是学习率本身，衰减强度因此是 $\eta_t\lambda$：PyTorch 与 A1 讲义都这样写。AdamW 原论文（Algorithm 2）的衰减只乘日程倍数 $\eta_t$（一个 0–1 之间的系数），不乘基础学习率 $\alpha$。两种写法的差别在扫学习率时显现：固定 $\lambda$ 改变 $\alpha$，按 PyTorch 写法有效衰减也跟着变，按原论文写法则不变。读到"独立 weight decay"或换用不同框架的超参数时，要先确认是哪一种。

$\epsilon$ 不只是防止除零；在梯度很小时，它还改变有效步长。比较优化器实现必须对齐 bias correction、epsilon 放置位置、weight decay 顺序和步数起点。源码见 [optim.py](../../examples/optim.py)，测试会用同一梯度序列与 PyTorch AdamW 对照。

还要区别梯度为零与没有梯度。这个 AdamW 实现遇到 `grad=None` 会跳过该参数，包括状态步数与衰减；显式零梯度仍会更新动量、推进步数并执行 weight decay。已有动量也可能继续推动参数。MoE 未激活专家、条件分支或全零优势 batch 中，是否分配零梯度张量会因此改变训练轨迹；“这一项没有任务梯度”不自动等于“参数没变”。

2026 年 A1 讲义的 Algorithm 1 写法与上式略有不同：先做 $\theta\leftarrow\theta-\alpha\lambda\theta$，再更新 $m,v$，最后 $\theta\leftarrow\theta-\alpha_t\,m/(\sqrt v+\epsilon)$，其中 $\alpha_t=\alpha\sqrt{1-\beta_2^t}/(1-\beta_1^t)$。它等价于把上式中的 $\epsilon$ 换成 $\epsilon/\sqrt{1-\beta_2^t}$，两者在任意有限步的有效 $\epsilon$ 上都可能不同，早期差异更大，随步数增加趋于相同。`optim.py` 用的是 PyTorch 的写法。另一个容易忽略的约定是哪些参数做 weight decay：讲义对所有参数都做，本书的 `train.py` 跳过一维参数（RMSNorm 增益），这是很多开源训练代码的惯例。

**常用默认值。** 公开大模型多用 $\beta_1=0.9$、$\beta_2=0.95$（比 Adam 原论文的 0.999 小，二阶矩对近期梯度反应更快，有助于避免损失尖峰），weight decay 0.1，梯度裁剪 1.0，预训练基本不用 dropout。2026 年第三讲的观察是：在讲义所讨论的大数据预训练设置中，weight decay 与余弦学习率日程的相互作用显著影响优化轨迹；这不排除它在数据有限或重复较多时的正则化作用，也不能仅凭 token 数大于参数量推出不会过拟合。

## 5.2 学习率如何与 token 预算对应

warmup 让动量、激活尺度与早期更新逐渐进入工作区间。cosine decay 在预先固定终点的情况下把学习率平滑降至最小值。作业采用的形式是：$t<T_w$ 时 $\alpha_t=\frac{t}{T_w}\alpha_{max}$；$T_w\le t\le T_c$ 时

$$\alpha_t=\alpha_{min}+\tfrac12\Big(1+\cos\big(\pi\tfrac{t-T_w}{T_c-T_w}\big)\Big)(\alpha_{max}-\alpha_{min});$$

$t>T_c$ 后保持 $\alpha_{min}$。用 $N$ 步训练时，应让 $T_c=N$，使学习率在训练结束时降到最低；若步数 $t$ 从 0 计到 $N-1$，最后一次更新的学习率会略高于 $\alpha_{min}$，要么令 $T_c=N-1$，要么接受这点差别。WSD 则分成 warmup、stable、decay，便于在稳定阶段分叉多个退火分支，比较不同训练预算。`optim.py` 提供 `cosine_lr` 与 `wsd_lr` 两种日程。

不要将长 cosine 训练中途的 checkpoint 当成相同预算下完成退火的模型。两者已消耗 token 相同，学习率历史和终点状态不同，会影响缩放定律拟合。课程在 MiniCPM、DeepSeek 等案例中强调了这一点。[2026 第十一讲](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_11.pdf)

全局 batch token 数为 $P\times B_{micro}\times T\times A$，其中 $A$ 为梯度累积次数。增大并行数或累积次数后若仍保持相同步数，总训练 token 已改变。日志必须同时记录 step、tokens seen、学习率和时间。

## 5.3 梯度累积怎样才等价

设本次更新的第 $i$ 个 microbatch 有 $M_i$ 个有效 token，平均损失为 $\ell_i$，正确总损失为

$$\ell=\sum_i\frac{M_i}{\sum_jM_j}\ell_i.$$

只有各 $M_i$ 相等，才等于各 microbatch loss 除以累积次数。可变长度 SFT/RL 中，简单 `/accum_steps` 经常不符合想优化的目标。应先决定是按 token、按序列，还是按 prompt 平均，再实现对应分母。

全局 norm clipping 应在所有 microbatch 梯度累积完成后执行一次：

$$g\leftarrow g\min\left(1,\frac{c}{\|g\|_2+\epsilon}\right).$$

若逐 microbatch 裁剪，得到的是不同优化算法。FP16 loss scaling 时先反缩放再裁剪。分布式参数分片下，还必须计算所有分片的总平方范数，不能独立裁剪每张卡。

裁剪不是非有限值的修复：Inf 乘零仍可变成 NaN，负阈值也没有合法范数含义。教学实现保留 FP64 累积精度，用缩放后平方和避免有限大梯度的平方溢出，并在修改任何梯度前拒绝 NaN/Inf。FP16 `GradScaler` 可按其协议跳过溢出步；那属于明确的失败处理，需单独计数。

## 5.4 BF16、FP16 与 FP32 的角色

FP16 有较多尾数精度但指数范围较窄，BF16 有与 FP32 相同的指数位数但较少尾数位。BF16 更不容易因为数值幅度大而溢出，却仍会损失细小增量。低精度矩阵乘通常配合较高精度累加，具体行为由硬件和算子决定。

混合精度并非把整个训练状态统一 `.half()`。参数、主权重、梯度、Adam 状态、归约、激活可使用不同 dtype。纯 BF16 更新时，当更新远小于参数的可表示间距，更新可能直接消失；FP32 主权重或补偿求和等策略处理的是这个问题。

本书 CPU 模型默认 FP32，便于数值验证。不要把它的内存占用当成 BF16 大模型训练的显存账。

`examples/optim.py` 的 Adam 状态与参数使用同一 dtype，因此该教学优化器默认服务于 FP32/FP64 参数，没有实现 FP32 master weights。将整个模型 `.bfloat16()` 后直接复用它，并不等于实现了正文介绍的生产混合精度方案。autocast 也不负责自动把所有 optimizer state 提升为 FP32；应实际检查参数、梯度和状态的 dtype。

## 5.5 初始化与尺度迁移

线性层 $h'=Wh$ 若输入各维方差近似为一，$\operatorname{Var}(W_{ij})\sim1/d$ 可使输出方差保持常数阶。但保持激活尺度并不自动保持更新尺度。学习率、参数化方式和宽度共同决定一次梯度更新能改变多少函数输出。

μP 的目标是让不同宽度模型拥有可比较的特征更新尺度，从而使小模型调出的超参数更容易迁移。它涉及输入层、隐藏矩阵、读出层各自不同的缩放规则，并区分优化器；不能简单概括为“学习率除以宽度”。验证方法是跨宽度比较激活统计和最优学习率是否趋于稳定，再评估迁移误差。[Tensor Programs V](https://arxiv.org/abs/2203.03466)

Muon 类方法把矩阵梯度的动量近似正交化，若 $M=U\Sigma V^\top$，理想化方向为 $UV^\top$；实际使用近似迭代避免完整 SVD。它改变矩阵更新的几何结构，常需为 embedding、标量参数等配套其他优化器。其优劣必须在匹配调参预算、训练计算量和规模的条件下验证，不能从一个很小模型的速度比赛直接外推。[Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)

## 5.6 一个失败定位顺序

| 现象 | 首先检查 | 有辨识力的实验 |
|---|---|---|
| loss 突然 NaN | 全 mask 行、低精度归约、爆炸 logits | 捕获第一个非有限张量 |
| 小 batch 无法过拟合 | 标签移位、梯度是否存在、参数是否更新 | 单层/短序列退化实验 |
| 训练很好、生成胡乱 | 因果 mask、cache、模板、解码 | 完整前向与逐步前向对照 |
| 多卡结果变差 | 梯度平均、样本重复、token 分母 | 单步全局 batch 等价对照 |
| 恢复训练曲线跳变 | optimizer、scheduler、RNG、data cursor | 连续跑与中断恢复比较 |

checkpoint 应包含模型、优化器、学习率状态、随机数状态、数据游标/采样器状态、tokenizer 版本与配置。只保存模型权重可以继续训练，但不是严格续跑。

## 5.7 为什么“激活不爆炸”还不够：一个更新尺度推导

考虑单样本线性层 $h=Wx$，$x\in\mathbb R^n$，每个坐标为常数阶，因此 $\|x\|^2=\Theta(n)$。SGD 的该层更新为 $\Delta W=-\eta\delta x^\top$，其中 $\delta=\partial\ell/\partial h$。固定同一输入后，这次权重更新导致

$$\Delta h=\Delta Wx=-\eta\delta\|x\|^2=\Theta(\eta n\delta).$$

即使初始化 $W_{ij}\sim\mathcal N(0,1/n)$ 使每个 $h_i$ 方差保持常数阶，更新仍可能随宽度增长。若在这个局部算例中固定 $\delta_i=\Theta(1)$，就需 $\eta=\Theta(1/n)$ 才能控制 $\Delta h_i$；真实网络中的 $\delta$ 又由读出层、损失归一化和其他层共同决定，因此这不是完整 μP 配方。

μP 要同时控制初始化坐标与学习后的坐标变化，并使特征更新不在极限中消失。实施时需区分哪些维度随规模增长、哪些保持固定；input embedding、隐藏矩阵、readout 与优化器不能统一套同一比例。官方 `mup` 实现用 base shapes 标记这些维度，并配套读出层、优化器与 attention 缩放。注意它的某些 Transformer 配方会采用与标准 $1/\sqrt{d_h}$ 不同的缩放，这依赖其宽度极限设计；本书 TinyLM 仍是普通参数化模型。[μP 官方实现](https://github.com/microsoft/mup)

验证分两步：先跨宽度运行少量更新，检查每层激活 `abs().mean()` 和更新幅度是否稳定；再检查最优学习率区间能否迁移。坐标检查通过只验证尺度约定，并不保证最优超参数在深度、数据、batch 或优化器改变后仍然相同。

## 5.8 临界 batch 为什么与梯度噪声有关

假设独立样本梯度均值为 $g$、协方差为 $\Sigma$，$B$ 个样本均值梯度的噪声协方差为 $\Sigma/B$。在简化的各向同性局部模型里，信噪比达到同量级时可定义

$$B_{noise}\sim\frac{\operatorname{tr}\Sigma}{\|g\|^2}.$$

小于该尺度时，增大 batch 能明显降低噪声；远大于该尺度后，再增加样本对每次更新的方向帮助有限，但仍增加计算量。真实“临界 batch”还受曲率、学习率、样本相关性和目标损失影响，不能直接当成硬件最佳 batch。[An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)

对语言模型，同文档相邻 token 并非独立，batch token 数和独立梯度样本数不是一回事。报告大 batch 收益时，同时比较时间到目标损失与计算量到目标损失，才能区分系统利用率改善与统计效率下降。

## 5.9 中断恢复不能悄悄改变学习率日程

计划训练 120 步，在第 40 步中断，与“计划训练 40 步并完成 cosine decay”是两种不同轨迹。恢复时不仅要载入权重，还要恢复原计划终点、已完成步数、动量和采样状态。

```bash
python -m examples.train --steps 120 --stop-after 40 --out outputs/resume
python -m examples.train --steps 120 --resume outputs/resume/checkpoint.pt --out outputs/resume
```

`train.py` 的恢复接口固定完整日程，测试比较连续运行与中断恢复后的模型、Adam 状态、随机数状态和日志，要求逐值相同。这只适用于本例确定性 CPU 路径；多进程 data loader、不同 GPU 数及非确定性算子还需要单独处理。activation checkpointing 也要保留随机算子的状态，否则反向重算的 dropout 可能不同于原前向。[PyTorch checkpoint 源码说明](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/utils/checkpoint.py)

## 5.10 从 token 文件喂满训练设备

大语料通常提前编码成整数文件，训练时随机选取 $T+1$ 个连续 ID，前 T 个作为输入、后 T 个作为目标。词表若超过 65,536，`uint16` 无法表示全部 ID；可以用 uint32/int32 保存，在送入 embedding 前转成框架要求的整型。文件头、字节序、tokenizer 指纹、文档边界和 token 总数需要一并记录。

内存映射让操作系统按访问页加载数据，不等于整个文件已经在物理内存中。`.npy` 文件含头部，应该用 `np.load(path, mmap_mode='r')` 或 `open_memmap` 读取；直接 `np.memmap` 原始二进制文件时必须约定 dtype、shape 和 offset。只读映射与 PyTorch 共享内存时不能无意原地写入，可复制选中的小 batch 后再转换。

预取、多个 worker、页锁定内存和异步 host→device 复制可以减少设备等待，但 worker 过多也会造成随机 IO、重复数据和 CPU 竞争。`non_blocking=True` 并不保证所有传输都与计算重叠，还受页锁定、流与同步依赖限制。先在 profiler 中区分取数、复制和计算阶段，再调整缓存、worker 和 batch。

`data.py` 中的 `mmap_next_token_batch` 用只读 `.npy` 演示目标移位和边界检查；`train.py` 的 `--train/--valid` 选项用同样方式直接从 token 文件训练，第 20.4 节的 TinyStories 实验就是这样运行的。它只处理一个已编码文件，不替代支持文档边界、跨卡分片、shuffle buffer、数据游标恢复的生产 loader。[2024 第二讲数据加载部分](https://github.com/stanford-cs336/spring2024-lectures/blob/2eab561d4fc5f0bd77ab68afc85dd563a0b107ba/lecture_02.py)

## 5.11 学习率扫描：真实曲线长什么样

第 20.4 节在 TinyStories 上用同一个 8.3M 参数模型、同样 1,500 步扫描了六个学习率。最终验证损失（nats/token）：

| 峰值学习率 | 3e-4 | 1e-3 | **3e-3** | 1e-2 | 3e-2 | 1e-1 |
|---|---:|---:|---:|---:|---:|---:|
| 1,500 步验证损失 | 2.938 | 2.493 | **2.325** | 2.758 | 3.148 | 4.542 |

损失对学习率呈明显的 U 形，最优点附近相差 3 倍，损失就差 0.17–0.43。学习率过大时（0.03、0.1）训练没有出现 NaN，但损失长时间停在高位：0.1 时验证损失从第 100 步的 4.69 回升到第 500 步的 5.21，在 5.0–5.2 之间徘徊到第 1000 步，之后才持续下降（第 1500 步为 4.54）。这次扫描没有做去掉梯度裁剪的对照，不能把"没有发散"归因于裁剪；AdamW 的自适应比例有助于控制尺度，但 $|\hat m|/\sqrt{\hat v}$ 并没有一般性的上界 1，不能由此保证逐坐标更新不超过学习率。扫描到最优值的 33 倍仍未发散，所以这个实验还没有找到发散边界。作业要求学习率扫描里至少包含一个发散的运行，就要继续增大学习率，或者去掉裁剪做对照。

## 5.12 社区进展：Muon 的稳定性与低精度优化器状态

Kimi K2 用 Muon 预训练了 1T 参数的 MoE 模型。报告指出，扩大规模后 Muon 比 AdamW 更容易出现注意力 logit 爆炸；对完整 key 做 QK-Norm 会破坏 MLA 的简单投影吸收路径，因此不能无代价照搬（第 7.10 节）。它的解法 QK-Clip 是：每步更新后检查每个头在当前 batch 中的最大 logit $S^h_{max}$，超过阈值 $\tau$ 时把该头的 $W_q$、$W_k$ 各乘以 $\sqrt{\tau/S^h_{max}}$。这一步不改变当前步的前向和反向计算，只约束权重的增长。对 MLA，只缩放各头独有的部分：$q^C$、$k^C$ 各乘 $\sqrt{\gamma_h}$，$q^R$ 乘 $\gamma_h$，所有头共享的 $k^R$ 不动，其中 $\gamma_h=\min(1,\tau/S^h_{max})$。Muon 与 QK-Clip 合称 MuonClip，K2 用它训练 15.5T token，全程没有一次 loss spike。

精度方面，DeepSeek-V3 把 AdamW 的一阶、二阶矩改存为 BF16，主权重和梯度仍保留 FP32（第 34.3 节）。这与第 5.4 节"更新远小于参数的可表示间距时会直接消失"的分析一致：精度要留在累加更新的地方。

## 5.13 Muon 规模化：矩阵形状、更新尺度与状态切分

对 $W\in\mathbb R^{m\times n}$，满秩时理想化正交化动量 $O=UV^\top$ 有 $\|O\|_F^2=\min(m,n)$，所以逐元素 RMS 为 $1/\sqrt{\max(m,n)}$。直接使用 O 会使宽矩阵和窄矩阵的每参数更新量不同。Moonlight 的规模化研究采用

$$\Delta W=-\eta\left(0.2\sqrt{\max(m,n)}\,O+\lambda W\right),$$

使理想化动量方向的 RMS 约为 0.2，并配合 weight decay。系数 0.2 是论文匹配 AdamW 更新尺度的经验选择，不是正交化的数学常数。实际 Newton–Schulz 用有限迭代，输出也不必是精确极分解；测试 SVD 的尺度关系只能核验该直觉，不代表复现了训练效率。[Muon is Scalable for LLM Training，§2.2](https://arxiv.org/html/2502.16982v1)

Muon 常用于二维隐藏权重，embedding、读出层和一维参数可采用其他优化器。必须明确矩阵的重排、QKV 是否合并，以及专家矩阵的边界，不能把所有参数先摊平成一个矩阵后正交化。长方形矩阵常先选择较小侧做迭代以节约算力，这改变实现成本而不改变目标矩阵的形状约定。

分布式训练还要说明正交化作用于完整梯度还是局部分片：一般有 $\operatorname{polar}([G_1,G_2])\ne[\operatorname{polar}(G_1),\operatorname{polar}(G_2)]$。对分片分别执行正交化，是另一种更新规则，不能自动声称与单机等价。状态所有权、必要通信与全矩阵/分片更新应有小矩阵对照，之后才比较吞吐与收敛。MuonClip 的注意力稳定约束解决另一类问题，应与更新 RMS 匹配分开消融。

## 5.14 Muon 的几何：谱范数约束下走多远

设当前梯度/动量矩阵为 $G=U\Sigma V^\top$。线性化损失后，在谱范数步长约束下求最陡下降方向：

$$\min_{\|\Delta W\|_2\le\eta}\langle G,\Delta W\rangle_F=-\eta\|G\|_*,\qquad \Delta W=-\eta UV^\top.$$

$\|G\|_*$ 是奇异值之和。该结论来自谱范数与核范数的对偶性：每个奇异方向可使用至多 $\eta$ 的步长，所以让它们与 G 反向即可达到下界。SGD 保留奇异值大小，Muon 的理想化方向则把非零奇异方向的增益拉到相近量级；这解释了它改变矩阵更新几何的含义，并不证明非线性训练长期最优。[Muon 原论文的正交化和形状尺度](https://arxiv.org/html/2502.16982v1)

有一个边界：作为优化器的方向选择，零梯度应保持零任务更新（weight decay 另计）。实际 Newton–Schulz 多项式保留零奇异方向；对应的部分极分解写作 $U\operatorname{diag}(\mathbf1[\sigma_i>0])V^\top$。满秩时其平方 Frobenius 范数为 $\min(m,n)$，秩 r 时为 r。第 5.13 节的 RMS 公式适用于满秩；秩亏时尺度为 $\sqrt{r/(mn)}$。直接对零矩阵取 SVD 后无条件用 `u @ vh`，会把任意零空间基当成更新，本书的数值工具显式排除这些方向。

## 5.15 weight decay、梯度累积与严格恢复的条件

暂时忽略任务梯度，AdamW 的衰减满足

$$\theta_K=\theta_0\prod_{t=1}^K(1-\lambda\eta_t)\approx\theta_0\exp\!\left(-\lambda\sum_t\eta_t\right).$$

近似要求每个 $\lambda\eta_t$ 足够小。相同 $\lambda$ 配上不同步数或日程面积，累计收缩不同；batch 翻倍、token 预算不变时，更新次数往往减半。因此优化器对照不能只抄同一个 weight decay 数字。[Decoupled Weight Decay](https://arxiv.org/abs/1711.05101)

梯度累积的等价也有前提：参数在所有 microbatch 间固定、随机行为按协议一致、没有跨 microbatch 统计依赖。MoE 容量路由、CLIP 的 batch 负例、BatchNorm 或按组 reward 归一化不能任意切断；否则即使 loss 分母正确，前向目标也已改变。

严格恢复需要恢复“实验”，不只是张量。`train.py` 将学习率/日程、batch/context、优化器超参数、评估协议、线程、PyTorch 与训练代码指纹、训练/验证 token 内容哈希写入 checkpoint；同一路径的文件被替换也会被检测。旧 checkpoint 缺少这些信息时不冒充精确续跑，可另行加载权重作为新的训练阶段。内容哈希需一次顺序读文件。本例 `tokens_per_second` 只计训练循环及其中的验证，哈希和初始化属于启动开销，报告端到端时间时需另计。

恢复文件本身也需要失败语义：先把完整 checkpoint 写到同目录临时文件，写入成功后再替换正式文件，避免磁盘写满或序列化异常破坏上一个可恢复状态。优化器参数组则应在任何参数更新之前统一检查；否则第二组的非法学习率报错时，第一组可能已经更新。本书分别用模拟写入失败和两参数组反例检查这两条不变量。

训练 loss 有限也不代表本次运行可以参与模型选择。验证 loss 若出现 NaN/Inf，应终止运行并标记为失败，保存已实际完成的更新次数；不能只把 JSON 数值改为 null 后仍返回成功。初始验证失败意味着完成 0 次更新，最后一次更新后的验证失败则保留那个步数，但该运行仍不能作为最优 checkpoint。

## 练习

**1．** 把 batch 翻倍后验证损失更低，是否说明"大 batch 更好"？

**解答。** 尚不能。若步数不变，你也训练了两倍 token；若 token 不变，优化器更新次数减半。应分别匹配训练 token、计算预算与学习率调参，再判断关键因素。

**2．** $\beta_2=0.999$ 与 $\beta_2=0.95$ 的二阶矩分别大约记住最近多少步的梯度？突然出现一个大梯度时，哪一个的更新幅度更容易失控？

**解答。** 指数滑动平均的有效窗口约 $1/(1-\beta_2)$：分别约 1000 步和 20 步。固定一阶矩与历史二阶矩时，较小 $\beta_2$ 会更快把新尖峰计入分母，往往减小这次自适应比例。例如长期 $m=v=1$ 后出现 $g=100$，取 $\beta_1=0.9$、忽略 bias correction 与 epsilon，$m=10.9$；$\beta_2=0.999$ 得 $v=10.999$、比例约 3.29，0.95 得 $v=500.95$、比例约 0.487。实际历史、bias correction、裁剪与参数组不同，不能仅凭 $\beta_2$ 判断哪次更新必然失控。

**3．** 余弦日程设 $\alpha_{max}=3\times10^{-3}$、$\alpha_{min}=3\times10^{-4}$、$T_w=75$、$T_c=1500$。第 787 步的学习率是多少？

**解答。** $787.5$ 是 $T_w$ 与 $T_c$ 的中点，第 787 步几乎就在中点，余弦项约为 0，$\alpha\approx\alpha_{min}+\frac12(\alpha_{max}-\alpha_{min})=1.65\times10^{-3}$。

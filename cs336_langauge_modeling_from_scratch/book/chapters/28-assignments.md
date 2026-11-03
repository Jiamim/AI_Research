# 28｜五组作业实战导读（以 2026 年为准）

前面各章按知识组织，本章按作业组织：每组作业要交付什么、哪些地方最容易错、对应本书哪一章和哪个 demo、没有大算力时怎样降级完成。内容依据 2026 年作业讲义（[A1](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf)、[A2](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf)、[A3](https://github.com/stanford-cs336/assignment3-scaling/blob/03e9372992e913061b9e78b5cfcb62ad8a87de35/cs336_assignment3_scaling.pdf)、[A4](https://github.com/stanford-cs336/assignment4-data/blob/0555bea66369872d912652debf10b115ca0688c8/cs336_assignment4_data.pdf)、[A5](https://github.com/stanford-cs336/assignment5-alignment/blob/c2734a26308710949fe13226960a1e8cece94b7e/cs336_spring2026_assignment5_alignment.pdf)），2024/2025 版本的差异见第 35 章。

先说清楚两件事。第一，截至 2026-10-31，官网允许基础编程与高层概念提问，禁止用 LLM 直接解决作业题，并强烈建议关闭 AI 自动补全（详细政策以官网链接文档为准）；本书的 demo 是用来理解原理、核对思路的参考，不是作业答案，作业代码请自己写。第二，课程的 GPU 预算以 B200 小时计；下面给出的低资源路径是讲义本身的建议或本书在 CPU 上实测可行的做法。

## 28.1 A1 基础：tokenizer、Transformer、训练循环

**要交付什么。** 从零实现 byte-level BPE 训练与编码（不能用现成 tokenizer 库）、`Linear`/`Embedding`/RMSNorm/SwiGLU/RoPE/softmax/注意力/Transformer LM、交叉熵、AdamW、余弦学习率日程、梯度裁剪、数据加载、checkpoint、解码；然后在 TinyStories 上训练到每 token 验证损失不高于 1.45，在 OpenWebText 上训练，做消融，并完成排行榜题（计 6 分）：只能用讲义提供的 OpenWebText 训练数据，在一张 B200 上最多运行 45 分钟，交最终验证损失和以墙钟时间为横轴的学习曲线。官方测试通过 `adapters.py` 接入你的实现，用 `uv run pytest` 运行。注意 A1 禁止使用 `torch.nn`、`torch.nn.functional`、`torch.optim` 中的定义，`nn.Parameter`、容器类和 `Optimizer` 基类除外；本书的 `examples/` 为简洁用了 `nn.Linear`、`F.cross_entropy` 等，不能照搬。

**最容易错的地方。**

| 位置 | 坑 | 概念讲解 |
|---|---|---|
| BPE 并列 | 应选字典序**更大**的 pair；写成 `min` 会让整张 merge 表错位 | 2.2 节（讲义示例的 merge 顺序） |
| 特殊 token | 先按 `<|endoftext|>` 切段，标记本身不计数；编码时重叠的特殊 token 取最长 | 2.2、2.5 节 |
| 训练速度 | 先聚合 pre-token，再增量更新 pair 计数；预切分按特殊 token 对齐并行 | 2.3 节（10K 词表 6 秒） |
| 流式编码 | 块末未定型的 pre-token 和特殊 token 前缀要留到下一块 | 2.5 节 |
| 解码 | 非法 UTF-8 用 `errors="replace"` 输出 U+FFFD | 2.1 节 |
| 初始化 | 线性层 $\mathcal N(0,\frac{2}{d_{in}+d_{out}})$ 截断于 $\pm3\sigma$，embedding $\mathcal N(0,1)$ 截断于 $\pm3$ | 5.5 节 |
| RoPE | 相邻两维成对旋转；位置从 cache 长度开始；只作用于 Q/K | 4.2、4.5 节 |
| 学习率日程 | 作业的余弦日程以 $(t,\alpha_{max},\alpha_{min},T_w,T_c)$ 定义，$t>T_c$ 后保持 $\alpha_{min}$ | 5.2 节 |
| 梯度裁剪 | 全局范数，$\epsilon=10^{-6}$ 加在分母 | 5.3 节 |

**资源核算题。** `transformer_accounting` 与 `adamw_accounting` 请自己算。第 8.8 节用作业没有考的 Llama 2 7B 演示同样的方法（参数、各部分 FLOPs、显存与训练时间），`python -m examples.accounting` 可复算。

**训练与消融。** 作业给出的 TinyStories 基础配置：10K 词表、上下文 256、$d=512$、$d_{ff}=1344$、4 层 16 头、RoPE $\Theta=10000$，总共处理约 3.3 亿 token，在一张 B200 上约 20–30 分钟。学习率要扫描，并找到至少一个发散的学习率，讨论"最优学习率在稳定性边缘"的说法；batch 从 1 扫到显存上限。消融包括：去掉所有 RMSNorm、改成 post-norm、去掉位置编码（NoPE）、SwiGLU 换成 $d_{ff}=4d$ 的不门控 SiLU。本书在 CPU 上做了一个缩小版的完整流程（第 20.4 节），每种消融的结果都有真实曲线。

**低资源路径。** 讲义建议 CPU 或 Apple MPS 上把总 token 降到 4,000 万，目标损失放宽到 2.00；讲义的参考实现在 M4 Max 上处理 $32\times5000\times256$ 个 token，CPU 约 1 小时 22 分、MPS 约 36 分钟，第 5000 步验证损失 1.80。MPS 上不要开 TF32；CPU 上可用 `torch.compile(model)`，MPS 上可用 `backend="aot_eager"`。调试先用 TinyStories 验证集训练 tokenizer（讲义按约 2.2 万篇计；本书使用的 V2 验证文件有 27,630 篇、22.5 MB），确认瓶颈一致后再上全量。

## 28.2 A2 系统：性能分析、FlashAttention-2、分布式

**要交付什么。** 端到端基准脚本与 nsys 分析；混合精度累加实验与 `autocast` 基准；显存分析；梯度检查点；PyTorch 注意力与 `torch.compile` 基准；FlashAttention-2 的纯 PyTorch 前向、Triton 前向（含 causal）、PyTorch 反向与基准；分布式通信基准、朴素 DDP、扁平化梯度 DDP、计算通信重叠的 DDP；优化器状态分片；FSDP；以及一组纸面计算题（数据并行、FSDP、张量并行、二维并行、环形 all-reduce 变体）。排行榜比的是两张 B200、batch 2 时一个完整训练步的墙钟时间，从空缓存开始 10 分钟内必须完成。

**模型尺寸表。** 除排行榜外，作业使用词表 10,000、batch 4，上下文长度随题目变化，五个尺寸：small（768/3072/12 层/12 头）、medium（1024/4096/24/16）、large（1280/5120/36/20）、xl（2560/10240/32/32）、10B（4608/12288/50/36）。用 8.8 节的方法先估算每个尺寸的参数、显存和 FLOPs，就知道哪些配置会 OOM，基准表里应该标出来而不是跳过。

**最容易错的地方。**

- 计时不同步：每次计时前后 `torch.cuda.synchronize()`，或用 `triton.testing.do_bench`；warmup 步不计入（9.4 节）。
- nsys 里分不清前向、反向和优化器：用 NVTX range 标记，再用 `--capture-range=nvtx --nvtx-capture=measure` 配合具名 `measure` 范围截取 warmup 之后（9.4 节）。
- FlashAttention：grid 是 (query tile 数, batch)，kernel 内只有一个 key 循环；片上 $m,\ell,O$ 用 FP32；作业约定的 causal 屏蔽值为 $-10^6$（有限负数是该测试设置的近似；通用精确 mask 通常使用 $-\infty$ 并处理全空行）；保存 $L$ 供反向，反向先算 $D=\mathrm{rowsum}(O\circ dO)$（10.3–10.5 节）。没有 GPU 时用 `TRITON_INTERPRET=1` 在 CPU 上核对下标与 mask；讲义提醒解释器有 bug，本书实测 Triton 3.6 解释器下 BF16 输入的 `tl.dot` 结果错误（第 10.4 节），BF16 路径只能在 GPU 上验证。
- 排行榜的模型比尺寸表大得多：$d=4096$、$d_{ff}=11008$、34 层、32 头、词表 151,936、上下文 32,768、BF16、batch 2，在两张 B200 上计时。讲义说明它是故意设计得难以放进显存的，实现也必须是自己的。
- DDP 不等长 batch：本地损失要按全局有效 token 数归一化（11.2、11.10 节）。
- 优化器状态分片：按参数大小平衡所有权，更新后由拥有者广播（11.4 节）。

**本书 demo。** `attention.py` 演示在线 softmax、softmax 反向恒等式与 Triton 行 softmax；`systems.py` 演示自定义 RMSNorm 反向、张量并行反向与低精度累加；`distributed.py` 验证 DDP、collective 等价与优化器状态分片。这些例子没有实现完整二维分块注意力前向与分块反向。CPU demo 验证数值与通信语义，GPU 性能需要实测。

### A2：怎样判断通信真的阻塞了训练

先固定模型、dtype、全局 batch 和有效 token 数，分别测单卡、同步 DDP、bucketed DDP 与 overlap 版本。计时前预热并同步；多卡迭代时间取最慢 rank，而不是只看 rank 0。记录均值/中位数、尾部时间、通信张量字节数、bucket 大小、软件和网络拓扑。

| 时间线现象 | 优先验证的假设 | 有辨识力的对照 |
|---|---|---|
| backward 后排着一长串 collective | 梯度就绪后才统一同步 | 提前启动 bucket，检查最后一个梯度后的通信尾部 |
| 大量短 collective，GPU 之间有空隙 | 延迟主导或 bucket 太小 | 扫消息/bucket 大小；记录调用数而非只看总字节 |
| collective 开始时间跨 rank 差很多 | 计算、取数或长度不均衡 | 对齐 NVTX 标签，比较各 rank 的数据/计算耗时 |
| 通信与计算重叠但步时没下降 | 二者竞争带宽/SM 或关键路径未缩短 | 对照无 overlap；用真实端到端步时确认 |

通信总耗时之和可能大于迭代墙钟时间，因为它与计算重叠，甚至不同流互相重叠。应测**关键路径上暴露的通信尾部**，并配合 $\alpha+S/W$ 判断延迟还是带宽限制。增加 batch 后更快也可能只是提高计算通信比，不等于算法统计效率不变。跨 rank 的设备时钟不能未经对齐直接相减。

### A2：activation checkpointing 的验收

这里的“检查点”是**激活重计算**，不是保存恢复训练文件。选择若干层/子图边界保留输入，反向时重跑内部计算。先以相同输入、权重和随机状态比较不开/开启时 loss 与梯度，再测真实峰值显存和完整训练步时。

重算必须重复相同的随机行为；dropout、MoE 随机路由、可变控制流都需检查。重算函数也不应重复有副作用的计数、缓存追加或状态更新。PyTorch 的 reentrant 与 non-reentrant checkpoint 行为不同，应显式配置 `use_reentrant`，而非依赖默认值。[PyTorch v2.11 checkpoint 说明](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/utils/checkpoint.py)

实验至少有“全不重算、按 block 重算、只重算高显存子图”三组，在固定 batch 下测内存/时间，再在各自最大可行 batch 下比较 tokens/s。第二种比较多改变了 batch，应另报到目标验证损失的成本。

### A2：FSDP 实现要点

先在两 rank 小网络上对照单步梯度与更新，再增加预取与重叠。核对：所有 rank 参数分片顺序相同；padding 分片不会被当作真实参数；前向和反向需要时聚集完整权重；梯度 reduce-scatter 的 sum/average 与 loss 分母一致；临时全量权重释放时仍保留反向需要的信息；更新仅作用于本地分片。

优化器应连接实际持久保存的参数分片，不能继续更新一个已被替换的旧张量。预取增加同时驻留的完整层，不能只用理想 $N/P$ 状态账判断是否放得下。测试还应覆盖共享参数、无梯度分支、恢复后的 optimizer state 与所有 rank 的 collective 顺序。[官方 FSDP2 源码](https://github.com/pytorch/pytorch/tree/v2.11.0/torch/distributed/fsdp)

以上是实现和验收方法，不提供评分题的完整 FSDP 代码；具体接口与作业规定以讲义为准。

## 28.3 A3 缩放律：用有限查询预算做决策

**要交付什么。** 第一部分用给定的 IsoFLOP 数据（`isoflops_curves.json`）复现 Chinchilla 的方法：每个预算取最低损失对应的 $N$，拟合 $N_*(C)$ 与 $D_*(C)$，外推到 $10^{23}$ 和 $10^{24}$ FLOPs。第二部分是重头戏：通过训练 API 在 12 B200-小时内查询任意配置的验证损失，为 48 B200-小时的最终训练选出最优模型尺寸与超参数，并预测其最终损失，提交到 API；成绩部分取决于你预测的配置实际训练出来有多好。

**API 规则要读仔细。** 排队和运行中的实验按你设定的最长运行时间**预扣**预算，完成后按实际用时返还，超时按上限扣；超出 12 小时后 API 拒绝新请求。训练数据是 DCLM 分词后的固定顺序数据，没有重复 epoch；模型种子只影响初始化；架构与 A1 基本一致，上下文 512，32K 词表。API 支持 AdamW 与 SGD，都用 warmup 加余弦的学习率日程；讲义示例的设置是 weight decay 0.01、裁剪 1.0、warmup 5%、末端学习率 10%，但这些都是可调的超参数。非 embedding 参数按 $12n_{layer}d^2$ 估算。预算按 B200 墙钟时间计，一次运行中途记录的 `val_losses` 也不是完成退火的短训练，这两点对实验设计的影响见第 13.5 节。

**策略要点。**

1. 先花一小部分预算确认学习率和 batch 在小模型上的合理范围，否则后续 IsoFLOP 曲线会被调参不足污染（13.5 节）。
2. 每个计算预算放 4–6 个模型尺寸，以上一预算的预测为中心；用抛物线拟合顶点而不是只取最低点：稀疏网格下，只取最低点的 $N_*$ 中位误差约 41%，拟合顶点约 4%（13.4 节的模拟）。
3. 检查最低观测点是否落在网格边缘，是就补点。13.4 节的模拟里，真最优点在网格内，但粗网格把最小预算的最低点选到了右端，抬高拟合左端、压低指数；多花 1 次查询补点后，抛物线外推的最差误差从 21% 降到 9%。实际需预留补点预算，并记录上限耗尽后仍未确定的曲线。
4. 留一个较大预算不参与拟合，用来检验外推；报告预测区间而不是单个数字。
5. 最终提交的超参数（学习率、batch、宽深比）也要有依据，通常来自小规模的趋势外推。

`python -m examples.scaling` 在已知真值的合成曲面上比较了这些设计选择，可以先在上面练习"怎么分配查询"。

### A3：时间预算先用实测吞吐换算

若本地可以运行同配置模型，用包含前向、反向、更新与同步的稳定窗口测有效 token 吞吐：

$$q=\frac{\text{窗口内有效训练 token}}{\text{窗口墙钟秒数}},\qquad\widehat D(t)=\widehat q\,t.$$

词表、上下文、batch、并行数、精度和重算方式都可能改变 q。若 API 不提供单步 benchmark，可通过少量短查询的已处理 token 和实际用时估计；排队时间是否计费、超时规则需按 API 单独处理。不能用硬件峰值或另一种模型的 MFU 代替实测 q。

以秒计的预算与 FLOPs 预算只有在已知吞吐/成本关系时才能互换。对不同尺寸分别校准 q，并预留验证、保存和故障开销；预热长短也会让短查询的平均吞吐偏低。

### A3：损失预测区间怎样落地

保存每次独立运行的 N、D、日程、seed、最终退火完成后的 loss 和实际用时。拟合后按**运行/seed**重采样，重新拟合一组参数；在每个样本中带入目标配置与采样出的吞吐估计，得到目标损失分布。留出更大预算检查覆盖情况，再报告分位数、外推倍数和残差结构。

均值置信区间只反映已选模型族中平均 loss 的估计不确定性；**下一次训练的预测区间**还需加入独立训练的残差、随机性及吞吐影响。同一次运行的多个 checkpoint 不是独立重复。bootstrap 也不能自动覆盖模型族错设与跨规模变化，留出校验失败时应扩大实验或缩小结论。

`numerical_tools.py` 用简单幂律和独立合成噪声展示均值区间与新观测区间，并由解析梯度/概率等对照检查代码；它不生成这门课最终评分配置或预测答案。

该示例在 log-loss 空间拟合；指数还原的趋势是条件中位数，不能直接叫算术均值。代码用残差的经验指数均值（smearing）修正原尺度均值，再分别给出 bootstrap 均值区间与加入独立残差的新运行区间。具体数字保存于 `book/review-update-results.json`；这是指定模型族与噪声下的教学结果，不是实际 LM API 的预测，也没有证明该方法能覆盖任意远距离外推。

## 28.4 A4 数据：从 Common Crawl 到可训练的 token

**要交付什么。** 查看 WARC/WET；用 Resiliparse 抽取正文；fastText 语言识别；邮件、电话、IP 的 PII 掩码；NSFW 与仇恨言论分类器过滤；Gopher 规则；以 Wikipedia 引用链接为正例训练质量分类器；精确行去重；MinHash + LSH 文档去重；最后组合成流水线，过滤已按 fastText 英文概率不低于 0.7 预过滤过的 WET 文件（原始为 2,500 个），用 GPT-2 tokenizer 编码为 `uint16`，在固定训练脚本下训练 GPT-2 small 形状的模型，目标是最小化 Paloma C4-100-domains 验证损失。训练代码和超参数不能改。

**最容易错的地方。**

- 每个过滤器的阈值都要靠抽样审计决定，并报告每一步删掉了多少文档（14.2 节）。
- IPv4 要校验每段 0–255；电话号码格式多样；替换成统一占位符后要监测占位符比例（14.2 节）。
- 行去重删除的是"在整个语料中出现超过一次"的行，两遍扫描，用哈希作键（14.3 节）。
- MinHash 找到的只是候选，必须复核真实 Jaccard；并查集聚类会产生传递闭包（14.4 节）。
- 每篇文档末尾加 `<|endoftext|>`；`uint16` 足以容纳 GPT-2 的 50,257 个 ID（5.10 节）。
- 该作业允许用 Paloma 验证数据训练过滤器，但不能把验证文本放进训练集；在这样的迭代中它承担开发集角色，独立研究还需另外的最终留出评估（14.10 节）。

**本书 demo。** `data.py` 用于理解第 14 章的规则过滤、PII 掩码、行去重和 MinHash 去重，接口和细节与作业不同，不能当作作业代码提交或改写；作业的这些函数请对照讲义自己实现，再用自己的 WET 样本做消融。语言识别和质量分类器不在 demo 中。

## 28.5 A5 对齐：从 GRPO 到估计量变体

**要交付什么。** 用 vLLM 评估 OLMo-2-0425-1B 在 GSM8K 上的三种提示（`question_only`、`r1_zero`、`r1_zero_three_shot`）；推导 baseline 方差；实现分词与 response mask、逐 token log-prob 与熵、奖励计算、组归一化、on-policy 损失、按序列聚合、带梯度累积的 GRPO 训练步；跑通标准 GRPO（4 个种子平均验证准确率至少 25%），扫学习率、做提示消融；然后实现 Dr. GRPO（去掉标准差、常数归一化）、RFT、MaxRL 并在受控条件下比较；再实现离策略的 token 级重要性加权与 GSPO 序列级比率并比较；最后必须提出并测试一个自己的估计量（`try_your_own`，10 分），相对已讲过的某个方法只改动一处。比较变体时还要实现 GRPO_constant（标准 GRPO 改用常数归一化），中间穿插推导难度权重、讨论 RFT 与长度归一化等书面题。

**建议超参数**（作业给出）：学习率 $10^{-5}$，每步 256 条 rollout（32 题 × 每题 8 条），梯度累积 32，裁剪 1.0，AdamW $(0.9,0.95)$ 无 weight decay，采样温度 1.0、最长 512 token，共 200 个 rollout 步，每 10 步评估至少 1,024 道验证题。

**最容易错的地方。**

- response mask 必须只覆盖模型生成的部分，包括停止串；prompt 与 padding 不进入损失（16.2、16.9 节）。
- 按序列平均时，微批损失乘以“本微批序列数 / 总序列数”再反向；若每条序列都用同一个固定 $L_{max}$ 作分母且仍按序列平均，该微批权重保持不变。若改成全 batch 有效 token 平均，权重才随各微批有效 token 数变化。应先写出完整 batch 的分子与分母，再推微批系数（5.3 节）。
- 组均值与标准差只在同一题的 8 条回答内计算；全对或全错的组 advantage 为 0（18.3 节）。
- old log-prob 必须 detach；padding 位置先置安全值再做指数（18.11 节）。
- 各变体沿用标准 GRPO 调好的学习率（讲义的协议，预算所限），并用多个种子画均值与区间。结论首先限于这个共同协议；若变体在不偏向它的 LR 下仍有超出不确定性的收益，证据更强，但变体较差也可能来自 LR 不适配，不能直接解释为算法本身更差（18.16 节）。
- 奖励解析器可能冤枉模型：基线阶段就要人工检查"格式错但其实答对"的样本。

**本书 demo。** `posttrain.py` 的 `grpo_arithmetic_demo` 是同一结构的缩小版：采样、打分、组 advantage、on/off-policy 两轮 clipped 更新、微批累积、熵与混合组比例日志，CPU 上几分钟跑完；`scripts/run_grpo_demo.py` 用 3 个种子比较三种 advantage。demo 的约定（标准差估计、数值保护、裁剪范围等）是为讲解选的，与作业规定并不一致；demo 不能作为作业代码，作业的约定一律以讲义为准，自己实现。

### A5：完整核对清单

| 环节 | 必须固定/核对 | 建议证据 |
|---|---|---|
| 数据与划分 | prompt 分组、train/dev/test、重复题与难度 | 按 ID/近重复检查；开发集用于调参 |
| 模板与分词 | system/chat template、prompt/response 边界、移位 | 人工展开一条 token、label、mask 对照 |
| EOS/PAD/停止 | 有效结束 token、停止串、截断类型 | EOS=PAD 时仍保留真实 EOS 监督 |
| 采样 | temperature/top-p、最大长度、独立 group | 原始 reward、长度、全对/全错比例 |
| 策略角色 | 当前、实际采样 old、reference | old/ref detach；记录权重版本和模型哈希 |
| 引擎一致性 | 同权重同前缀的 logits/logps、精度、路由 | 更新前 replay rollout，查看 log-ratio 分位数 |
| 概率支持 | top-p 支持与概率重归一化 | 保存实际采样 mask；禁止只用原始 logits 代替截断概率 |
| reward/解析器 | 正确性、格式、单位与等价表达 | 人工核对成功/失败/格式错样本；空结果不默默计正常 |
| advantage | 同题 group、std correction、epsilon、长度权重 | 固定手工奖励组与零方差组的数值对照 |
| loss 与累积 | token/sequence/prompt 分母、跨卡全局权重 | 完整 batch 与 microbatch 单步梯度对照 |
| 优化 | unscale→clip→step、有效 step 数、scheduler | 梯度有限、参数确有更新、溢出跳步单独计数 |
| 权重同步/缓存 | 更新后 rollout 引擎加载、旧 KV 失效 | 同一版本 replay；旧数据重用次数与 policy lag |
| 评估 | 独立题、固定预算、seed、checkpoint 选择 | 准确率+区间+长度/成本；禁止按测试分选模型 |

所有异常日志要能追到触发 prompt、token、样本版本及 last-good checkpoint。一次 loss 有限不保证整个循环正确；训练 reward 提高也不替代独立验证。测试覆盖本书 demo，不代表通过官方作业全部验收。

## 28.6 推荐的完成顺序与时间分配

| 阶段 | 内容 | 先读 | 先跑 |
|---|---|---|---|
| 第 1–2 周 | A1 tokenizer 与模型组件，单元测试全部通过 | 第 01–05 章 | `examples.tokenizer`、`tests/test_course.py` |
| 第 3 周 | A1 训练、学习率扫描、消融 | 第 05、20 章 | `scripts/run_tinystories.py` |
| 第 4–5 周 | A2 基准、FlashAttention、DDP/分片 | 第 08–11 章 | `examples.attention`、`examples.distributed` |
| 第 6 周 | A3 先做纸面拟合，再规划 API 预算 | 第 13 章 | `examples.scaling` |
| 第 7 周 | A4 过滤与去重流水线 | 第 14 章 | `examples.data` |
| 第 8–9 周 | A5 GRPO 及变体 | 第 16–18 章 | `examples.posttrain` |

每完成一组作业，用第 20.10 节的报告模板写一页记录：改了什么、怎样验证正确、结果的不确定性有多大、哪些结论只在当前规模成立。

## 28.7 把作业知识迁移成自己的研究实验

课程 handout 与测试接口属于特定年度；独立研究应重新定义数据、预算和验收，而不以“通过作业测试”代替方法结论。可以按以下顺序扩大：

| 阶段 | 最小对照 | 通过后再扩大什么 |
|---|---|---|
| 目标与 mask | 展开一条 token/label/mask；完整与微批梯度对照 | 变长 batch、packing、多卡 |
| 数值与内存 | FP64 小矩阵、块边界、全空/极端输入 | BF16/FP8、实际 GPU kernel |
| 架构 | 固定预算的基线与单项改动 | 两种规模、足够的 LR 调参 |
| 数据 | 固定验证、域比例、重复曝光与去重簇 | 更多来源及长尾任务 |
| 后训练 | 固定 rollout 复算概率、reward 和一次更新 | 在线采样、异步版本、多轮环境 |
| 交付 | 恢复/取消/失败样例、独立评估与成本 | 长运行和真实服务负载 |

新增机制实验见第 20.14 节。它们帮助检验公式，完整的 CCE GPU、MLA/SSM 工业实现、真实验证器与多机 RL 仍需相应系统；先列清缺失部件与实验条件，再做性能或能力结论。

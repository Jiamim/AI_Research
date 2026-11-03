# 08｜资源核算：先算清楚，再运行实验



## 8.1 张量布局是资源分析的起点

PyTorch tensor 不只有 shape，还有 dtype、stride、storage offset 和实际 storage。`transpose` 常返回共享存储的 view；`contiguous()` 可能分配新内存。`reshape` 有时是 view，有时需要复制，取决于布局能否满足新的 stride。

对于 $X:[B,T,d]$，`numel()*element_size()` 给出逻辑元素字节数，但多个 view 可能共享存储，不能把它们全部相加当成独立占用。峰值显存还包括梯度、优化器、临时算子 workspace、通信缓冲和分配器保留空间。

PyTorch 的 reserved 通常已经包含 allocated：它是分配器向设备申请后持有的内存，不能把两者再相加。可以报告 allocated 峰值、reserved 峰值及设备总占用，并解释差额；三者在不同时间达到峰值，也不应拼成一个并不存在的同时峰值。

`einsum` 的价值是让维度语义清楚，例如 `btd,df->btf`。可读的表示并不保证编译出的内核最快；跨维度排列、是否形成 contiguous 张量和矩阵形状仍会影响实际速度。各年第二讲通过 PyTorch/einops 把这些细节与 FLOPs 分析连接起来。[2026 第二讲](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_02.py)

## 8.2 参数量逐项相加

对无 bias、无权重共享、GQA+SwiGLU 的 decoder，在这些结构假设完整列出且不含额外模块时，参数量为

$$N=2Vd+L(2d^2+2dd_{kv}+3df+2d)+d.$$

其中 $2Vd$ 是 embedding 与输出头，$2d^2+2dd_{kv}$ 是 Q/K/V/O 投影，$3df$ 是 MLP，$2d$ 是块内两个 RMSNorm，最后 $d$ 是末尾归一化。若共享输入/输出 embedding，减去 $Vd$。RoPE 固定频率不算可训练参数。

这个公式应与 `sum(p.numel() for p in model.parameters())` 对照。差异通常揭示了 bias、权重共享或配置假设不一致。

## 8.3 FLOPs 的主项与被忽略的项

矩阵乘 $[m,k]@[k,n]$ 约需 $2mkn$ FLOPs。每个训练 token 经过的 dense 权重若为 $N$，前向线性层约为 $2N$ FLOPs；对输入和权重的梯度各增加一次同量级矩阵乘，所以训练主项约为 $6ND$。

但 $6ND$ 不是精确的宇宙常数。注意力 score 与 value 聚合额外贡献每层前向约 $4BT^2d$ FLOPs；反向、causal tile 跳过、重计算、词表头、MoE active 参数及未使用参数都会改变实际数字。

判断 $6ND$ 是否合适，应先检查序列长度是否足以让 $T^2d$ 项显著。短序列下线性层常占主导；极长序列下忽略注意力运算会严重低估成本。

## 8.4 一个训练预算算例

假定 dense 模型 $N=10^9$，训练 $D=2\times10^{10}$ token，近似 $C=1.2\times10^{20}$ FLOPs。若全部设备合计实测持续训练吞吐为 $2\times10^{14}$ FLOP/s，则仅主计算约需 $6\times10^5$ 秒，即 6.94 天。

这不是某张 GPU 的性能承诺；吞吐数字是人为设定的算例。数据读取、checkpoint、失败重跑与验证还需额外时间。把理论峰值直接放进分母会低估训练周期。

模型 FLOPs utilization（MFU）是无量纲比率，必须计入时间。若一步的算法 FLOPs 为 $F_{model}$、平均步时为 $t_{step}$、设备数为 $P$、单卡理论峰值为 $F_{peak}$ FLOP/s，则

$$\mathrm{MFU}=\frac{F_{model}}{t_{step}\,P\,F_{peak}}.$$

HFU 的分子可计入重计算等实际执行运算，因此可能高于同口径 MFU。报告时要给出公式、dtype、设备数、是否使用稀疏峰值、是否计入 attention 与重计算，以及时间是否包含数据加载和通信。

## 8.5 显存账单必须说明 dtype 假设

一个常见假设：BF16 模型参数 2 bytes、BF16 梯度 2 bytes、FP32 master 参数 4 bytes、FP32 Adam 一二阶状态各 4 bytes，总计 16 bytes/参数。10 亿参数的这部分约 16 GB，尚未包括激活。

若梯度为 FP32，则变为 18 bytes/参数；若直接持有 FP32 参数与梯度而没有额外 master copy，参数 4、梯度 4、Adam 8，也为 16。总数碰巧一样不代表训练状态结构一样。

传统注意力保存 $P:[B,h_q,T,T]$ 时，单层占 $sBh_qT^2$ bytes。取 $B=1,h_q=32,T=8192,s=2$，单个概率矩阵就是 4 GiB。FlashAttention 避免保存该矩阵，但模型其他激活仍随 $BTd$、层数和 checkpoint 策略增长。

KV cache 属于推理状态，通常为

$$M_{KV}=2LBTh_{kv}d_hs.$$

因子 2 来自 K 和 V，$s$ 是每元素字节数。这是未计分页碎片、额外元数据、量化 scale 和共享前缀的基础账。

## 8.6 重计算用算力购买容量

普通反向保存中间激活。checkpoint 只保留部分边界，反向时重跑局部前向。若按 $L$ 层链条每 $k$ 层设一个边界，简化峰值激活约与 $L/k+k$ 成正比，取 $k\approx\sqrt L$ 可得到 $O(\sqrt L)$ 的层数依赖。真实 Transformer 的不等层开销、算子融合和调度使最优点更复杂。

重计算增加 FLOPs，却可能允许更大 batch、更少分片或更好的通信重叠，最终缩短墙钟时间。应同时测量 tokens/s 和峰值显存，而不是只盯着 FLOPs 增加。

## 8.7 让参数账与 FLOPs 账真正对上

令 $N_{proj}=L(2d^2+2dd_{kv}+3df)$，统计各层投影矩阵的参数；输出头为 $Vd$。忽略 norm、激活函数、bias 和词表 softmax 的低阶项，一批序列的前向约为

$$F_{forward}\approx2BT(N_{proj}+Vd)+4LBT^2d.$$

注意 embedding lookup 没有对所有 $Vd$ 权重做一次 dense matmul。未共享输入/输出 embedding 时，直接拿总参数 $N$ 代入 $2BTN$ 会把输入 embedding 当成每 token 的密集计算；在小模型大词表场景中误差明显。权重共享又会让“一份参数”承担不同算子角色，所以最准确的账始终按操作而不是按存储份数累加。

在可近似把前向/反向主要矩阵乘按 1:2 计数时，训练一步约为

$$F_{train}\approx6BT(N_{proj}+Vd)+12LBT^2d.$$

对照论文时要先对齐注意力项的约定。PaLM 定义 MFU 时每 token 用 $6N+12LTd$，与上式一致（不计 causal mask 省掉的一半）。Kaplan 等的 $C\approx6N+6Ln_{ctx}d$ 只给每 token 注意力计入一个 $T^2$ 项量级，即上式注意力项的一半。FlashAttention 等内核报告的 TFLOP/s 又常按 causal 只算一半 tile。同一次训练用不同约定，算出的 MFU 可以差出几个百分点，"我的 FLOPs 对不上论文"往往只是约定不同。

这是未跳过未来 tile 的全矩阵约定；真正利用因果稀疏性的内核能减少 score/value 部分运算，但不会把所有线性投影也减半。optimizer update、重计算、padding 和通信另计。

变长 batch 要按各自长度计算。若隔离文档且内核真正跳过无效配对，注意力主项随 $\sum_iT_i^2$ 增长；若补齐到最长 $T_{max}$ 后仍做稠密计算，则按 $B T_{max}^2$ 付费。例如长度 1024 与 8192 两条序列，稠密 padding 的配对数约为各自单独计算的 1.97 倍。attention mask 保证语义正确，不自动让执行成本变为有效 token 数的函数。

对普通 MHA、$f=8d/3$ 的 SwiGLU，$N_{proj}/L\approx12d^2$，每层注意力配对项与投影项的比值约为 $T/(6d)$。例如 $d=4096,T=32768$，比值约 1.33，注意力已不能被 $6ND$ 主项轻率忽略。这是单层主项的数量级判断，不是某个 FlashAttention 内核的实测速度比。

## 8.8 一个完整算例：Llama 2 形状的模型

基础作业要求对 GPT-2 形状的模型做完整的参数、FLOPs、显存与训练时间核算，那几道题请自己完成。这里用作业没有考的 Llama 2 7B 演示同样的方法：$V=32000$、$T=4096$、$L=32$、$d=4096$、32 个头、$d_{ff}=11008$，SwiGLU、RMSNorm、不共享 embedding。[examples/accounting.py](../../examples/accounting.py) 把下列数字全部算出，`python -m examples.accounting` 可复算。

**参数。** 每层 $4d^2+3dd_{ff}+2d=202{,}383{,}360$，32 层共 6.48B；embedding 与输出头各 $Vd=131.1$M；再加末端 RMSNorm，合计 **6,738,415,616**，与公开的 Llama 2 7B 参数量一致。BF16 仅加载权重就需 13.5 GB。

**前向 FLOPs。** 只计矩阵乘，每条序列取满上下文（统一取 4,096）：

| 模型 | 每条序列（TFLOPs） | Q/K/V/O 投影 | 注意力 $QK^\top$ 与 $PV$ | SwiGLU | 输出头 |
|---|---:|---:|---:|---:|---:|
| Llama 2 7B | 62.9 | 28.0% | 14.0% | 56.4% | 1.7% |
| Llama 2 13B | 119.0 | 28.9% | 11.5% | 58.5% | 1.1% |
| Llama 2 70B（GQA，8 个 KV 头） | 606.9 | 16.3% | 7.2% | 76.1% | 0.3% |
| Llama 2 7B，上下文 32,768 | 996.0 | 14.1% | 56.5% | 28.5% | 0.9% |

模型变大时，MLP 占比上升、输出头占比下降，因为 $Ld^2$ 项增长快于 $Vd$；70B 用 GQA 把 K/V 投影缩小到 1/8，投影占比明显下降。上下文从 4K 增到 32K，每条序列的 FLOPs 增加约 15.8 倍而不是 8 倍：随 $T^2$ 增长的注意力项从 14% 变成 57%。这正是 8.3 节"先检查 $T^2d$ 项是否显著"的定量版本。

**显存。** 混合精度 AdamW 的模型状态按 8.5 节的 16 字节/参数计，7B 为 107.8 GB，一张 80 GB 显卡放不下，必须分片（第 11 章）。激活用 Korthikanti 等人推导的每层 $sbh(34+5as/h)$ 字节估算（16 位、无模型并行；$s$ 为序列长度、$b$ 为 batch、$h$ 为隐藏宽度、$a$ 为头数）：[Reducing Activation Recomputation](https://arxiv.org/abs/2205.05198)

| 策略 | 每条 4,096 token 序列的激活 |
|---|---:|
| 不重计算 | 104.2 GB |
| 历史公式中的选择性重计算（移去二次注意力保存项） | 18.3 GB |
| 每层重算：仅已保存的层输入边界 | 1.07 GB |

$5as/h$ 一项来自注意力分数、softmax 与 dropout，$s=4096$ 时它等于 160，远大于 34，所以长序列下最值得省的是注意力部分。原公式按 4h GeLU MLP 加 dropout 的 GPT 块推导，SwiGLU 块的常数不同，这里只作量级估算。表中的 1.07 GB 只数边界保存量，不是训练峰值：反向仍有当前层重建激活、QKV、梯度与 workspace。FlashAttention 的具体保存张量也不等于简单移去旧公式的一项；它与选择性重算需分别测量。

**训练时间。** Llama 2 7B 用 2.0T token 训练。按 8.7 节的 $F_{train}$（含注意力项，不跳过未来 tile）约 $9.2\times10^{22}$ FLOPs，只用 $6ND$ 则是 $8.1\times10^{22}$。论文报告 7B 用了 184,320 个 A100-80GB GPU 小时；A100 的 BF16 峰值为 312 TFLOP/s，由此反推的有效利用率约 45%（按 $6ND$ 计约 39%）。[Llama 2](https://arxiv.org/abs/2307.09288) 核算的用处就在这里：把一个公开数字拆成"总 FLOPs ÷ 峰值 ÷ 利用率"，判断它是否合理。

不同的人对"哪些张量算激活"的约定可能不同，结果会差一些；关键是把约定写清楚，并能指出主导项。

## 8.9 社区进展：把公式对到真实 checkpoint

第 29 章把本章的参数公式推广到 GQA、MLA、MoE、MTP 和 Gated DeltaNet 与 Lightning Attention 混合层，用 14 个开源模型的 `config.json` 计算，再与官方 safetensors 元数据对照，全部精确相等。对齐过程暴露了几类本章公式没有覆盖的张量：

- DeepSeek-V3（13.46B）、GLM-4.5、Qwen3-Next 的 checkpoint 带有多 token 预测（MTP）模块，DeepSeek-V3 报告的 671B 不含它；
- GLM-4.5 的 Q/K/V 投影带 bias（`attention_bias=true`），输出投影却没有，按"四个投影都有 bias"计会多算；
- Kimi K2 每层存了一个 rotary `inv_freq` 缓冲。

"激活参数"也需要定义：gpt-oss 模型卡计入输出头、不计输入 embedding，按这个口径复算的 5.13B 和 3.61B 与模型卡一致。

## 8.10 FlashAttention 之后，词表 logits 可能成为新峰值

设有效输入位置数 $M=BT$，输出词表 V，完整 logits 有 MV 个元素。$M=32768,V=131072$ 时，单个 BF16 logits 就是 8 GiB，FP32 为 16 GiB；随后 log-softmax、梯度或临时 cast 可能再增加峰值。FlashAttention 只解决注意力矩阵，不解决这一输出层。

交叉熵实际只需要每行的 logsumexp 与目标 logit。可以把词表切成块，维护行统计，反向重算各块概率；第 10.10 节给出精确推导，Cut Cross-Entropy 将这类思路映射到融合内核。[Cut Your Losses](https://arxiv.org/html/2411.09009v1)

只把 logits 按 token 分成小块后逐个相加 loss，若普通 autograd 仍保留每个块的 softmax，累积保存量未必下降到一个块；必须检查保存张量的生命周期。相反，分块重算减少的是 logits 激活，不会消除 $Vd$ 权重、其梯度和 Adam 状态。把“峰值少了 8 GiB”和“词表层全部省掉”混在一起，会再次算错容量。

## 练习

**1．** 参数已经可以放入显存，为什么仍会在第一次 `optimizer.step()` OOM？

**解答。** 许多优化器惰性创建动量状态。前向和反向通过只说明模型、激活与梯度能容纳，首次更新还要分配状态和可能的临时缓冲。

**2．** Llama 2 7B 共享输入与输出 embedding 后参数减少多少？前向 FLOPs 是否减少？

**解答。** 参数减少 $Vd=131.1$M（约 1.9%）。前向 FLOPs 不变：embedding lookup 本来就不是矩阵乘，输出头的 $2TdV$ 仍然要算。

**3．** 把 Llama 2 7B 改成 GQA（32 个 query 头、8 个 KV 头），每层投影参数减少多少？对上下文 4096 的前向 FLOPs 影响大吗？

**解答。** K/V 投影从 $2d^2$ 降到 $2d\cdot d/4$，每层减少 $1.5d^2\approx25.2$M，32 层约 805M。投影 FLOPs 减少 37.5%，总前向 FLOPs 约减少 10.5%；注意力分数与加权求和的 FLOPs 不变，因为 query 头数没变。GQA 的主要收益在推理时的 KV cache，见第 12 章。

**4．** 若以 BF16 参数与梯度、FP32 主权重与 Adam 状态训练 Llama 2 7B，模型状态需要多少显存？与全 FP32（参数、梯度、两个动量）相比呢？

**解答。** $16$ bytes/参数（2+2+4+4+4），约 107.8 GB，与全 FP32 相同。混合精度节省的是激活和带宽，不是这部分模型状态；减少它要靠 ZeRO/FSDP 分片，见第 11 章。

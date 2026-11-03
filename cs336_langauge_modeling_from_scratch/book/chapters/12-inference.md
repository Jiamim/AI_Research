# 12｜推理系统：延迟、吞吐和生成质量



## 12.1 prefill 与 decode 是两种工作负载

prefill 一次处理整个输入，通常有较大的矩阵乘，权重可在很多 token 间复用。decode 每步只生成一个新 token，需要反复读取模型权重和已有 KV。batch 很小时，线性层接近矩阵向量乘，往往受内存带宽约束。

这解释了为什么同一个模型训练时利用率很高，交互式生成却可能很低。不能用训练 tokens/s 预测在线服务延迟。服务指标至少包括首 token 延迟 TTFT、后续 token 间隔 TPOT、请求吞吐、输出 token 吞吐和 p95/p99 延迟。

## 12.2 KV cache 省掉了什么

过去位置在因果网络中的隐藏表示不会因新 token 而改变，所以过去各层 K/V 可以复用。每步只计算新 token 的 Q/K/V，并对所有历史 KV 做注意力。cache 省去重复计算过去前缀，但仍需读取相关历史 KV，单步 attention 成本仍随历史长度增长。

取 $L=32,h_{kv}=8,d_h=128,T=8192,B=1$，BF16 下缓存约为

$$2\times32\times8\times128\times8192\times2=1\ \mathrm{GiB}.$$

若改用 32 个 KV 头，变为 4 GiB。多并发、长输出和长 prompt 会迅速吃满显存。paged cache 与共享前缀能够减少碎片/重复，但不会改变独立唯一 KV 的基本信息量。

**减小 KV 的几条架构路线**（2025/2026 推理讲义的整理）：GQA/MQA 让多个 query 头共享 KV（第 04 章）；MLA 只缓存低维 latent，DeepSeek-V2 把每 token 每层需要缓存的维度从 MHA 的 $2\times16384$（K 与 V 各 $128$ 头 $\times128$ 维）压到 $512+64=576$（后 64 维留给 RoPE，见第 07 章）；跨层注意力（CLA）让相邻层共享 KV；局部（滑动窗口）注意力与全局注意力交替，局部层只保留最近 $w$ 个位置的 KV。目标是在不明显损失质量的前提下缩小 KV：DeepSeek-V2 报告 MLA 在其设置下甚至略优于 MHA，局部注意力单独使用则可能掉点。这些结论都应在自己的长上下文任务上复核。

## 12.3 采样控制分布，不是修复模型

temperature 用 $p_i\propto\exp(z_i/\tau)$，$\tau$ 越低通常越集中；贪心取 argmax 是独立分支，不能直接除以零。top-k 只保留概率最大的 k 个，top-p 保留累积概率达到阈值的最小前缀并重新归一化。

过滤顺序、temperature 放置位置与停止 token 都是生成协议的一部分。报告任务分数必须固定这些设置。降低温度可能改善某些格式任务，却降低候选多样性，影响 pass@k 和自一致性投票。

训练时不要把 top-p 截断后的概率误用作原策略的 log probability。RL 的采样分布若与训练中使用的策略概率不同，重要性比率和目标解释都需要相应调整。

## 12.4 continuous batching 与分页缓存

传统 batch 等待最长请求完成，短请求结束后资源空闲。continuous batching 在每个调度轮次移除完成请求并加入新请求，提高利用率，但需要维护各请求的长度、cache 页表和位置。

PagedAttention 把 KV cache 分成固定大小物理块，通过页表映射到每个请求的逻辑序列，缓解必须预分配连续最大长度缓存的问题。它改善内存管理，不等于改变 attention 的数学定义。[PagedAttention](https://arxiv.org/abs/2309.06180)

块表还让**前缀共享**变得自然：多个请求共享同一个 system prompt 时，填满的前缀块只存一份，按引用计数释放。SGLang 的 RadixAttention 把所有请求的前缀组织成基数树，在多轮对话和 agent 工作负载里复用率很高。[inference.py](../../examples/inference.py) 的 `PagedKVCache` 是一个最小的块分配器：4 个请求共享 48 token 的系统提示、各自再有 25 个 token，块大小 16 时共占 11 个物理块，不共享需要 20 个；若按最大长度 2048 连续预留则需要 512 个。

大型 prefill 可能阻塞 decode，引发延迟抖动。chunked prefill 把长输入切段与 decode 交错；prefill/decode 分离还可使用不同设备池，但增加 KV 传输和调度复杂度。选择要基于真实请求长度分布与服务延迟目标。

## 12.5 量化为什么有时省内存却不加速

均匀对称量化可写作 $q=\operatorname{clip}(\operatorname{round}(w/s),q_{min},q_{max})$，反量化 $\hat w=sq$。逐 tensor scale 简单，逐通道/分组 scale 更能适应不同范围，但增加元数据和反量化开销。

weight-only 量化主要减少权重读写和存储；若没有匹配的融合内核，先解压回 FP16 再普通矩阵乘可能抵消速度收益。激活 outlier、KV 量化误差、校准数据分布和 accumulation dtype 都影响质量。

在没有 clipping 的标量均匀量化中，rounding 误差满足 $|w-\hat w|\le s/2$；发生 clipping 后不再有这个界。对线性层，$\Delta y=(\hat W-W)x$，所以 $\|\Delta y\|\le\|\hat W-W\|\,\|x\|$。同样的权重误差遇到不同激活分布会产生不同输出误差，这解释了为什么用真实校准输入衡量输出重建误差，可能比只最小化权重 MSE 更有效。

稀疏剪枝也类似：删掉数值不代表硬件会跳过计算。非结构化零值若仍用 dense GEMM，FLOPs 与时间不一定减少。参数存储、理论乘加数和实际执行路径应分别报告。

## 12.6 speculative decoding 的精确接受规则

draft 模型从分布 $q$ 提议 token $x$，target 分布为 $p$。接受概率为 $\min(1,p(x)/q(x))$；若拒绝，从归一化残差分布 $[p-q]_+$ 采样替代。它不是简单地“target 觉得够好就接受”。

为什么正确？接受路径对 token $x$ 贡献 $q(x)\min(1,p(x)/q(x))=\min(p(x),q(x))$。所有拒绝的总概率等于 $\sum_x[p(x)-q(x)]_+$，乘残差条件分布后贡献 $[p(x)-q(x)]_+$，两条路径相加恰为 $p(x)$。

多 token 版本让 target 并行验证一段 draft，遇到第一次拒绝时截断后续提议；全部接受时还可以利用 target 已算出的下一位置分布多生成一个 token。收益依赖接受率、draft 成本、可并行验证长度和内存瓶颈，并非 draft 越小越好。Medusa 在 target 上加多个预测头直接并行出草稿（它常配合 typical acceptance，不保证与 target 分布完全一致；Medusa-2 还会微调主干），EAGLE 让草稿模型读取 target 的高层特征，多 token prediction 训练目标也可提供草稿能力。

`examples/inference.py` 的 `speculative_step` 实现一轮完整的"起草 $k$ 个、验证、拒绝时从残差采样、全部接受时补一个"。对随机生成的 5 词表、$k=3$ 的分布，两万轮模拟中首 token 的经验分布与 target 分布的最大偏差约 0.004（两万次抽样的统计误差量级），平均每轮产出约 2.1 个 token。分布完全一致是这个算法的正确性标准，每轮 token 数才是速度收益。

接受与修正规则中的 $p,q$ 必须是实际目标和草稿的采样分布，包括各自 temperature/top-p 处理；还要使用同一 token 事件空间。若只是比较原始 logits，而采样经过另一套处理，上述证明不再适用。拒绝概率为零时无需构造残差分布，否则会在 $p=q$ 的情况下除以零。[Speculative Decoding 原论文](https://arxiv.org/abs/2211.17192)

多 token 验证还要求每一行条件于**同一条实际草稿前缀**：第 i 行的 target 与 draft 都看到输入前缀加草稿的前 i−1 个 token。在第 i 处拒绝、换成修正 token 后，后续草稿行的条件已经过时，必须丢弃并按新前缀重算，KV 也要回滚到接受部分。逐位置的正确性由这个条件关系归纳到完整序列，并非只检查首 token 就足够。`speculative_step` 是一轮概率操作，调用方仍负责 EOS、输出预算和 KV 回滚；它返回的修正/额外 token 也不属于“被接受的草稿”。配套两 token 小例子枚举联合分布，另测中途拒绝时不会使用过时的后续行。

概率以 FP16/BF16 保存后，总和可能因舍入略偏离 1，检查时应先高精度求和，再允许有依据的小误差。对非负的正规数，逐项相对舍入误差按概率质量加权后仍约为单位舍入误差，而不是词表大小乘以该误差；只有次正规数的绝对误差项会随元素个数累加。直接把容差设为 `vocab_size * eps`，在 512 词表的 BF16 中就会容忍总和为 0.5，掩盖输入问题。本书参考实现将两种误差分开，只在合法舍入范围内重新归一化。

## 12.7 一个服务实验应该怎样报告

至少固定输入/输出长度分布、并发数、采样策略、模型权重、量化设置与 cache 精度。区分饱和吞吐实验和满足延迟约束的负载实验。吞吐高的配置若让 p99 延迟超过应用要求，仍然可能不可用。

**练习。** 一种 batching 策略让输出 tokens/s 翻倍，却让首 token 延迟翻三倍，是否更好？

**解答。** 取决于应用。离线批处理可能接受，实时交互可能无法接受。性能比较应在明确的服务目标约束下进行，而不是只最大化一个计数。

服务负载还需说明请求到达过程。只等前一个请求完成再提交下一个的闭环测试，会在服务变慢时自动降低到达率，容易掩盖排队尾部；固定到达率的开环测试则应记录排队、超时和被拒请求。TTFT 从提交到首 token，包含排队、预填充与调度；TPOT 是逐 token 间隔的汇总，均值不能替代 p95/p99。离线饱和 tokens/s 与满足延迟目标的有效吞吐需要分别报告。

若请求生成 n 个 token、首 token 时刻为 $t_1$、末 token 为 $t_n$，常见逐请求 TPOT 为 $(t_n-t_1)/(n-1)$，仅在 $n>1$ 时定义。先对每个请求算 TPOT 再平均，给请求等权；汇总所有间隔再平均，给长输出更多权重，两者可能不同。流式接口还可能一次发送多个 token，所以客户端收到的 chunk 间隔不一定是模型逐 token 延迟。

稳定负载下，Little 定律给出平均在途请求数 $\bar N=\lambda\bar T$，其中 $\lambda$ 为完成请求率、$\bar T$ 为平均系统停留时间；排队与服务都要计入。这个约束有助于核对并发、吞吐、延迟是否自洽，但持续过载时队列增长，不能把短暂窗口的数字套成稳态容量结论。

## 12.8 用带宽下界估计 decode 的延迟与吞吐

只考虑每轮必须读取的 dense 线性权重，参数量为 $N_w$、每参数 $s$ bytes、并发 batch 为 $B$，每轮计算约 $2BN_w$ FLOPs，读权重约 $sN_w$ bytes，算术强度近似 $I\approx2B/s$，BF16 下约为 $B$。H100 的 ridge point 约 295，仅靠权重复用要接近它，batch 需要到数百，而不是 1–4 个请求。

但每个请求还有自己的 KV cache，它不在请求之间共享。2026 推理讲义用一个简单模型把两者放在一起：假设计算与访存完美重叠、忽略其他开销，每一步解码都要读一遍全部权重和所有请求的 KV，

$$\text{latency}\approx\frac{sN_w+B\cdot M_{KV/seq}}{W},\qquad \text{throughput}=\frac{B}{\text{latency}}.$$

以 Llama 2 13B（40 层、$d=5120$、40 个头、头宽 128）在 H100（3.35 TB/s、80 GB）上、上下文 1024、BF16 为例，`python -m examples.accounting` 给出：

| 配置 | 显存（GB） | 每步延迟（ms） | 吞吐（token/s） |
|---|---:|---:|---:|
| MHA，batch 1 | 26.9 | 8.0 | 125 |
| MHA，batch 64 | 79.7 | 23.8 | 2,689 |
| MHA，batch 256 | 240.8 | — | 放不下 |
| GQA 8 个 KV 头，batch 64 | 33.4 | 10.0 | 6,417 |
| GQA 8 个 KV 头，batch 256 | 65.6 | 19.6 | 13,068 |

读法：增大 batch 用更差的单请求延迟换更高的总吞吐，因为读权重的成本被摊薄了；但每个请求的 KV（MHA 下每序列 0.84 GB）随 batch 线性增长，很快放不下。GQA 把 KV 缩小 5 倍，同样 batch 下延迟更低、能开更大的 batch（表中 GQA 行同时少了 1.68B 的 K/V 投影权重，参数从 13.02B 降到 11.34B，所以这一列的差别不全来自 KV）。这是吞吐的理论上界、延迟的理论下界，真实系统还要扣除 kernel 开销、调度和非重叠部分。

无排队且预热完成时，首 token 延迟（TTFT）的主要计算通常是 prefill；在线还包括等待与调度。prefill 是大矩阵乘，适合小 batch 以降低 TTFT；decode 适合大 batch 以提高吞吐。prefill/decode 分离的目的之一是让两类阶段各自选择 batch，另外也是为了消除 prefill 对 decode 的干扰，并让两者独立配置资源与并行方式。

带宽模型默认 GPU 的全部 SM 都在读 KV，这在长上下文、小 batch 的 decode 里不一定成立。一些 decode 内核按 batch 与头组织线程块，并行度可能不足。实际块数取决于使用 query 头还是 KV 头、GQA 分组和 tile 划分，不能仅从 8 个 KV 头推出只有 8 个块；小 batch 下仍应检查 SM 利用率和实际带宽。Flash-Decoding（split-KV）因此把 KV 的序列维也切成若干段并行计算，每段得到局部输出和 logsumexp，再用在线 softmax 的合并规则（第 10.2 节）归约。它增加了沿序列的并行度，有机会提高带宽利用率；主 KV 数据量不变，但多了部分输出、logsumexp 和归约的读写，不能保证达到理论带宽。短上下文或大 batch 时并行度已足够，这些额外开销可能抵消收益。

教学 cache 每步用 `torch.cat` 扩大 K/V，既分配新张量又复制旧内容；它验证语义，但还增加了本可避免的缓存搬运。预分配 cache 按位置写入、分页 cache 按页追加，才更接近服务实现。相同的 cache 数学不等于相同的缓存管理成本。

带宽项还应与计算下界比较：理想重叠时 $t_{step}\ge\max(F_{step}/F_{peak},M_{read}/W)$，实际还受同步、通信与 kernel 限制。上表仅展示内存带宽模型，不是测出的服务速度；高 batch 进入计算受限区后，不能继续仅用 bytes/W 外推。MoE 的权重读取还取决于该批次覆盖了哪些专家，不能直接把总参数或单 token active 参数当作每轮唯一读取量。

## 12.9 剪枝加蒸馏：换一个更小的模型

量化保持结构、降低每个数的位数；另一条路线是直接得到更小的模型。课程介绍的流程是：在大模型上用校准数据估计各层、各头、MLP 通道或 embedding 维度的重要性，剪掉不重要的部分，再以原模型为教师做蒸馏，恢复剪枝造成的损失，必要时迭代。结构化剪枝（删整层、整头、整通道）得到稠密的小矩阵，有机会使用常规内核加速；新形状的对齐、占用率与启动开销仍可能抵消收益；非结构化稀疏则依赖专门内核。与从头训练同尺寸小模型相比，这条路线的卖点是用少得多的 token 达到相近质量，但需要在目标任务上验证。 两个代表性的公开实践：Minitron 对 Nemotron-4 做深度、宽度、注意力与 MLP 的结构化剪枝，再用不到原训练数据 3% 的 token 做蒸馏重训，把模型压缩 2–4 倍；Sheared-LLaMA 把 LLaMA2-7B 剪到指定目标形状，并按各数据域的 loss 动态调整重训数据配比。[Minitron](https://arxiv.org/abs/2407.14679)、[Sheared-LLaMA](https://arxiv.org/abs/2310.06694)

## 12.10 社区进展：KV 设计与权重精度决定并发

第 29 章的练习 1 用真实配置算了一笔账。在一张 80 GB 显卡上服务 32K 上下文的请求：

- Qwen3-30B-A3B 以 BF16 部署，约 5.9 个请求，即最多同时服务 5 个；
- gpt-oss-20b 用 MXFP4 权重，约 82 个。

两者激活参数相近，理想显存容量比约为 14 倍（82 对 5.9）。这个差距由两部分相乘：空闲显存差约 3.5 倍（两者权重差约 47 GB，其中 MXFP4 相对 BF16 省下约 28 GB）；每请求的 KV 差约 4 倍。两者每层每 token 的 KV 其实一样大（都是 1,024 个数），KV 的差别来自 gpt-oss 层数只有一半、其中一半又是 128 token 的窗口。MLA 每层只缓存 576 个数，与头数无关。DeepSeek-V3 的 MTP 模块可以作为 speculative decoding 的草稿头（12.6 节）。长上下文推理成本的架构应对见第 30.4 节。

## 12.11 草稿模型的接受率与加速边界

单位置接受率可化为

$$\alpha=\sum_vq_v\min(1,p_v/q_v)=\sum_v\min(p_v,q_v)=1-\tfrac12\|p-q\|_1.$$

它由实际采样分布的 total variation 距离决定；在相同题上“答案准确率接近”不能替代逐前缀概率匹配。草稿非常小但分布偏差大，可能比稍大的高接受率草稿更慢。[Speculative Decoding](https://arxiv.org/html/2211.17192v2)

设一轮草稿 k 个 token 花 $t_d(k)$，target 验证花 $t_v(k)$，平均输出 $\mathbb E[L]$，则每输出 token 的平均轮次成本约为 $(t_d(k)+t_v(k))/\mathbb E[L]$，需低于普通 decode 的 $t_v(1)$ 才有收益。接受率随位置和前缀变化、EOS 提前终止、KV 回滚、调度和共享设备争用都会影响这张账。不能把“平均吐出 3 个 token”直接叫作“三倍加速”。

## 12.12 前缀缓存的键与失败恢复

KV 是“模型在某个前缀上的中间计算”，缓存键至少应覆盖模型/适配器版本、全部祖先 token、位置规则，以及多模态输入身份；相同末块 token 配不同历史不产生相同 KV。生产设计常将父块哈希、本块 token 和额外身份组合，且只复用已填满的不可变块；共享未填满块后继续写入需要 copy-on-write 或其他隔离。[vLLM 前缀缓存设计](https://docs.vllm.ai/en/latest/design/prefix_caching/)

本书的 allocator 仅代表同一固定模型、文本前缀的块语义，不存实际 K/V。请求入场先算需要多少新块，再一次性更改状态；容量不足就保持原状态，避免“前面几个块已分配、最后一个 OOM”泄露引用。测试验证失败入场、重复 request ID 与释放后块数。真实服务还需要取消、抢占、并发锁及缓存失效。

## 12.13 分离 prefill/decode 为什么依赖网络

分离设备可降低大 prefill 对 decode 的干扰，并分别调整并行布局，但要把已计算的 KV 送到 decode 池。传输下界约为 $M_{KV,prompt}/W_{network}$，还包括排队与布局转换；长 prompt、大 KV 或慢互连可能抵消收益。按延迟目标统计 goodput，即在约定 TTFT/TPOT 等阈值内完成的请求速率，才能判断分离是否有价值。[DistServe](https://arxiv.org/html/2401.09670v1)

同样 tokens/s 下，某系统大量超时，另一系统稳定完成，两者可用吞吐不同。按请求长度分层报告排队、传 KV、prefill、decode 与取消率，比只给一个总吞吐数字更能解释瓶颈。

## 练习

**1．** 用 12.8 节的模型估算：Llama 2 13B 在 H100 上，batch 64 时 MHA 改成 MQA（1 个 KV 头），显存和每步延迟大约变成多少？

**解答。** 每序列 KV 从 0.84 GB 降到 0.021 GB，batch 64 共 1.3 GB；K/V 投影变小，权重降到约 21.9 GB。总计约 23.3 GB，延迟约 7.0 ms，吞吐约 9,200 token/s。代价是质量可能下降，GQA 是折中。

**2．** speculative decoding 的单位置接受率为 $\alpha$、每轮起草 $k$ 个 token，若各位置独立，期望每轮产出多少 token？

**解答。** 前 $i$ 个草稿全部被接受的概率为 $\alpha^i$，加上拒绝或全部接受后多出的那一个：$\mathbb E=\sum_{i=0}^{k}\alpha^i=(1-\alpha^{k+1})/(1-\alpha)$。$\alpha=0.54,k=3$ 时约 2.0，与 demo 的 2.1 同量级（demo 各位置的接受率不同）。

本章以 [2025 推理讲义](https://github.com/stanford-cs336/spring2025-lectures/blob/b98b08a98d9d47a69bbdcb4e96a58aa48ee4d13b/lecture_10.py) 和 [2026 推理讲义](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_10.py) 为主线。

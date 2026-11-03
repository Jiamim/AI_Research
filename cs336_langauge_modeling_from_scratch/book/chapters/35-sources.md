# 35｜逐年覆盖表、资料索引与核验记录
本书以 2026-10-26 可访问的官方材料为依据。三年课程表、51 份核心讲义、15 份主作业和一份沿用两年的安全补充作业均已获取；manifest 共 71 条记录，其中安全补充保留两个页面来源。讲义按 Python 文本及 PDF 抽取文本阅读核对，未声称逐个观看视频，也没有把所有课件图片作为已读文本处理。
## 35.1 年份变化与合并决策
| 主题 | 2024 | 2025 | 2026 | 本书处理 |
|---|---|---|---|---|
| 系统内核 | RMSNorm 自定义内核 | FlashAttention-2 | FlashAttention-2、checkpoint/FSDP 扩展 | 保留 RMSNorm 推导，展开在线注意力 |
| 数据 | 四次独立数据课 | 两次数据课并吸收其他主题 | 数据来源、过滤、混合、合成 | 保留 DSIR、Bloom、后缀数组 |
| 推理服务 | 分散于架构/系统 | 独立第十讲 | 保留并更新 | 独立第 12 章 |
| 后训练 | SFT、偏好、RLHF/DPO | 数学 SFT、EI、GRPO | 归一化、RFT、MaxRL、GSPO | 分开解释偏好与可验证奖励 |
| 其他架构 | MoE | MoE | 注意力替代与 MoE | 单列第 07 章 |
| 第十七讲 | 评估 | RL 代码讲解 | 多模态 | 三个主题分别保留 |

表中的年度差异是根据课程表、讲义与作业内容作的比较，不代表某主题此前完全没有被提及。
## 35.2 逐讲覆盖
### 2024
| 讲次 | 官网主题 | 本书章节 | 已核验来源 |
|---|---|---|---|
| 1 | Overview, tokenization (Percy) | 00–02 | [讲义](https://github.com/stanford-cs336/spring2024-lectures/blob/2eab561d4fc5f0bd77ab68afc85dd563a0b107ba/lecture_01.py) |
| 2 | Pytorch, resource accounting (Percy) | 01、05、08 | [讲义](https://github.com/stanford-cs336/spring2024-lectures/blob/2eab561d4fc5f0bd77ab68afc85dd563a0b107ba/lecture_02.py) |
| 3 | Architectures, hyperparameters (Tatsu) | 03–05 | [讲义](https://github.com/stanford-cs336/spring2024-lectures/blob/2eab561d4fc5f0bd77ab68afc85dd563a0b107ba/nonexecutable/Lecture%203%20-%20architecture.pdf) |
| 4 | Mixture of experts (Tatsu) | 06 | [讲义](https://github.com/stanford-cs336/spring2024-lectures/blob/2eab561d4fc5f0bd77ab68afc85dd563a0b107ba/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) |
| 5 | GPUs (Tatsu) | 08–10 | [讲义](https://github.com/stanford-cs336/spring2024-lectures/blob/2eab561d4fc5f0bd77ab68afc85dd563a0b107ba/nonexecutable/Lecture%205%20-%20GPUs.pdf) |
| 6 | Kernels, Triton (Percy) | 09–10、21 | [讲义](https://github.com/stanford-cs336/spring2024-lectures/blob/2eab561d4fc5f0bd77ab68afc85dd563a0b107ba/lecture_06.py) |
| 7 | Parallelism (Tatsu) | 11 | [讲义](https://github.com/stanford-cs336/spring2024-lectures/blob/2eab561d4fc5f0bd77ab68afc85dd563a0b107ba/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) |
| 8 | Parallelism (Percy) | 11 | [讲义](https://github.com/stanford-cs336/spring2024-lectures/blob/2eab561d4fc5f0bd77ab68afc85dd563a0b107ba/lecture_08.py) |
| 9 | Scaling laws (Tatsu) | 13 | [讲义](https://github.com/stanford-cs336/spring2024-lectures/blob/2eab561d4fc5f0bd77ab68afc85dd563a0b107ba/nonexecutable/Lecture%209%20-%20Scaling%20laws%20basics.pdf) |
| 10 | Scaling laws (Tatsu) | 05、13 | [讲义](https://github.com/stanford-cs336/spring2024-lectures/blob/2eab561d4fc5f0bd77ab68afc85dd563a0b107ba/nonexecutable/Lecture%2010%20-%20Scaling%20details.pdf) |
| 11 | Data (Percy) | 14 | [讲义](https://github.com/stanford-cs336/spring2024-lectures/blob/2eab561d4fc5f0bd77ab68afc85dd563a0b107ba/lecture_11.py) |
| 12 | Data (Percy) | 14 | [讲义](https://github.com/stanford-cs336/spring2024-lectures/blob/2eab561d4fc5f0bd77ab68afc85dd563a0b107ba/lecture_12.py) |
| 13 | Data (Percy) | 14 | [讲义](https://github.com/stanford-cs336/spring2024-lectures/blob/2eab561d4fc5f0bd77ab68afc85dd563a0b107ba/lecture_13.py) |
| 14 | Data (Percy) | 04、14、16、18 | [讲义](https://github.com/stanford-cs336/spring2024-lectures/blob/2eab561d4fc5f0bd77ab68afc85dd563a0b107ba/lecture_14.py) |
| 15 | Alignment (Tatsu) | 16 | [讲义](https://github.com/stanford-cs336/spring2024-lectures/blob/2eab561d4fc5f0bd77ab68afc85dd563a0b107ba/nonexecutable/Lecture%2015%20-%20Alignment%20by%20SFT.pdf) |
| 16 | Alignment (Tatsu) | 17 | [讲义](https://github.com/stanford-cs336/spring2024-lectures/blob/2eab561d4fc5f0bd77ab68afc85dd563a0b107ba/nonexecutable/Lecture%2016%20-%20Alignment%20by%20RLHF.pdf) |
| 17 | Evals (Tatsu) | 15 | [讲义](https://github.com/stanford-cs336/spring2024-lectures/blob/2eab561d4fc5f0bd77ab68afc85dd563a0b107ba/nonexecutable/Lecture%2017%20-%20Evals.pdf) |
| 18 | Guest lecture by Ce Zhang | 资料边界，未推断嘉宾口述 | [课程表](https://cs336.stanford.edu/spring2024/)；未列独立讲义 |
| 19 | Guest lecture by Aakanksha Chowdhery | 资料边界，未推断嘉宾口述 | [课程表](https://cs336.stanford.edu/spring2024/)；未列独立讲义 |
### 2025
| 讲次 | 官网主题 | 本书章节 | 已核验来源 |
|---|---|---|---|
| 1 | Overview, tokenization (Percy) | 00–02 | [讲义](https://github.com/stanford-cs336/spring2025-lectures/blob/b98b08a98d9d47a69bbdcb4e96a58aa48ee4d13b/lecture_01.py) |
| 2 | PyTorch, resource accounting (Percy) | 05、08 | [讲义](https://github.com/stanford-cs336/spring2025-lectures/blob/b98b08a98d9d47a69bbdcb4e96a58aa48ee4d13b/lecture_02.py) |
| 3 | Architectures, hyperparameters (Tatsu) | 03–05 | [讲义](https://github.com/stanford-cs336/spring2025-lectures/blob/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) |
| 4 | Mixture of experts (Tatsu) | 06 | [讲义](https://github.com/stanford-cs336/spring2025-lectures/blob/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) |
| 5 | GPUs (Tatsu) | 08–10 | [讲义](https://github.com/stanford-cs336/spring2025-lectures/blob/b98b08a98d9d47a69bbdcb4e96a58aa48ee4d13b/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) |
| 6 | Kernels, Triton (Tatsu) | 09–10 | [讲义](https://github.com/stanford-cs336/spring2025-lectures/blob/b98b08a98d9d47a69bbdcb4e96a58aa48ee4d13b/lecture_06.py) |
| 7 | Parallelism (Tatsu) | 11 | [讲义](https://github.com/stanford-cs336/spring2025-lectures/blob/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) |
| 8 | Parallelism (Percy) | 11 | [讲义](https://github.com/stanford-cs336/spring2025-lectures/blob/b98b08a98d9d47a69bbdcb4e96a58aa48ee4d13b/lecture_08.py) |
| 9 | Scaling laws (Tatsu) | 13 | [讲义](https://github.com/stanford-cs336/spring2025-lectures/blob/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) |
| 10 | Inference (Percy) | 04、12 | [讲义](https://github.com/stanford-cs336/spring2025-lectures/blob/b98b08a98d9d47a69bbdcb4e96a58aa48ee4d13b/lecture_10.py) |
| 11 | Scaling laws (Tatsu) | 05、13 | [讲义](https://github.com/stanford-cs336/spring2025-lectures/blob/00191bba00d6d64621dc46ccaed9122681413a24/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf) |
| 12 | Evaluation (Percy) | 15 | [讲义](https://github.com/stanford-cs336/spring2025-lectures/blob/b98b08a98d9d47a69bbdcb4e96a58aa48ee4d13b/lecture_12.py) |
| 13 | Data (Percy) | 04、14、16 | [讲义](https://github.com/stanford-cs336/spring2025-lectures/blob/b98b08a98d9d47a69bbdcb4e96a58aa48ee4d13b/lecture_13.py) |
| 14 | Data (Percy) | 14 | [讲义](https://github.com/stanford-cs336/spring2025-lectures/blob/b98b08a98d9d47a69bbdcb4e96a58aa48ee4d13b/lecture_14.py) |
| 15 | Alignment - SFT/RLHF (Tatsu) | 16–17 | [讲义](https://github.com/stanford-cs336/spring2025-lectures/blob/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) |
| 16 | Alignment - RL (Tatsu) | 18 | [讲义](https://github.com/stanford-cs336/spring2025-lectures/blob/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) |
| 17 | Alignment - RL (Percy) | 17–18、20 | [讲义](https://github.com/stanford-cs336/spring2025-lectures/blob/b98b08a98d9d47a69bbdcb4e96a58aa48ee4d13b/lecture_17.py) |
| 18 | Guest Lecture by Junyang Lin | 资料边界，未推断嘉宾口述 | [课程表](https://cs336.stanford.edu/spring2025/)；未列独立讲义 |
| 19 | Guest lecture by Mike Lewis | 资料边界，未推断嘉宾口述 | [课程表](https://cs336.stanford.edu/spring2025/)；未列独立讲义 |
### 2026
| 讲次 | 官网主题 | 本书章节 | 已核验来源 |
|---|---|---|---|
| 1 | Overview, tokenization [Percy] | 00–02 | [讲义](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_01.py) |
| 2 | PyTorch (einops), resource accounting (FLOPs, memory, arithmetic intensity) [Percy] | 05、08–09 | [讲义](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_02.py) |
| 3 | Architectures, hyperparameters [Tatsu] | 03–05 | [讲义](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_03.pdf) |
| 4 | Attention alternatives and mixture of experts [Tatsu] | 06–07 | [讲义](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_04.pdf) |
| 5 | GPUs, TPUs [Tatsu] | 08–10 | [讲义](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_05.pdf) |
| 6 | Kernels, Triton [Percy] | 09–10 | [讲义](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_06.py) |
| 7 | Parallelism [Percy] | 11 | [讲义](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_07.py) |
| 8 | Parallelism [Tatsu] | 11 | [讲义](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_08.pdf) |
| 9 | Scaling laws [Tatsu] | 13 | [讲义](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_09.pdf) |
| 10 | Inference [Percy] | 04、12 | [讲义](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_10.py) |
| 11 | Scaling laws [Tatsu] | 05、13 | [讲义](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_11.pdf) |
| 12 | Evaluation [Percy] | 15 | [讲义](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_12.py) |
| 13 | Data (sources, datasets) [Percy] | 14 | [讲义](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_13.py) |
| 14 | Data (filtering, deduplication, mixing, synthetic data) [Percy] | 14、16 | [讲义](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_14.py) |
| 15 | Mid/post-training (SFT/RLHF) [Tatsu] | 16–17 | [讲义](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_15.pdf) |
| 16 | Post-training - RLVR [Tatsu] | 18 | [讲义](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_16.pdf) |
| 17 | Alignment - multimodality [Percy] | 19 | [讲义](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_17.py) |
| 18 | Guest lecture: Daniel Selsam | 资料边界，未推断嘉宾口述 | [课程表](https://cs336.stanford.edu/)；未列独立讲义 |
| 19 | Guest lecture: Dan Fu | 资料边界，未推断嘉宾口述 | [课程表](https://cs336.stanford.edu/)；未列独立讲义 |

2024 年评估讲义注明嘉宾课没有录像；2025/2026 官网未为嘉宾行提供独立讲义链接。本书不能保证涵盖这些课独有的口述、互动或最新观点，读者不应把这部分当成已经完整核验。
## 35.3 五组作业的知识覆盖
| 作业 | 学习重点 | 本书章节 | 本书实际可运行内容 |
|---|---|---|---|
| A1 | tokenizer、模型、优化、训练与生成 | 01–05、08、20、28 | 按作业约定的 byte BPE（预切分、特殊 token、增量合并、流式编码）、带消融开关的解码器、AdamW/日程/裁剪、TinyStories CPU 实验 |
| A2 | profiler、内核、通信、分片 | 08–11、28 | 在线 softmax 与 softmax 反向恒等式、Triton 行 softmax kernel（CPU 解释器可运行）、DDP/collective/优化器状态分片双进程检查、低精度累加（作业评分的 FlashAttention 实现不提供） |
| A3 | IsoFLOP 与预算外推 | 13、28 | 已知真值曲面上的预算受限实验设计、最低点与抛物线两种估计、多种子误差分析 |
| A4 | 抽取、过滤、去重、消融 | 14、28 | Gopher 式规则、联系方式替换、行级精确去重、MinHash+LSH 去重的参考实现（接口与作业不同） |
| A5 | SFT、偏好、RLVR 及变体 | 16–18、28 | SFT/蒸馏/DPO/SimPO/GRPO/GSPO 目标、在小语言模型上的完整 GRPO 循环与变体比较、离散策略的 baseline 方差枚举（例子不同于作业） |

这些实现用于建立知识与验证机制；官方作业的全量测试适配、竞赛结果、GPU 内核、互联网规模清洗和大模型训练运行不在已完成实验声明内。
| 年份/作业 | PDF 页数 | 固定历史版本 |
|---|---:|---|
| A2024-1 | 34 | [官方 PDF](https://github.com/stanford-cs336/spring2024-assignment1-basics/blob/dae79d035bf866a71d81a02b0c023980d537b16d/cs336_spring2024_assignment1_basics.pdf) |
| A2024-2 | 30 | [官方 PDF](https://github.com/stanford-cs336/spring2024-assignment2-systems/blob/6849ecb981c83e8a979b1ccca9a0e26949adcb9a/cs336_spring2024_assignment2_systems.pdf) |
| A2024-3 | 6 | [官方 PDF](https://github.com/stanford-cs336/spring2024-assignment3-scaling/blob/3f16a055d73e8626f1491c0548d7fb339703df15/cs336_spring2024_assignment3_scaling.pdf) |
| A2024-4 | 16 | [官方 PDF](https://github.com/stanford-cs336/spring2024-assignment4-data/blob/a5cd5b75d2b6a17b96b6e1c3c7eb267bec8e1291/cs336_spring2024_assignment4_data.pdf) |
| A2024-5 | 21 | [官方 PDF](https://github.com/stanford-cs336/spring2024-assignment5-alignment/blob/fcbfe2b3b72cb9e5536b93ba12265d04b8ca4cca/cs336_spring2024_assignment5_alignment.pdf) |
| A2025-5-safety | 19 | [官方 PDF](https://github.com/stanford-cs336/assignment5-alignment/blob/eb2c562e05802d3c9ee4c4ec2eec43b104d77e9e/cs336_spring2025_assignment5_supplement_safety_rlhf.pdf) |
| A2025-1 | 50 | [官方 PDF](https://github.com/stanford-cs336/assignment1-basics/blob/430e2c844e29f8aad8f9330e8706db9cb508241f/cs336_spring2025_assignment1_basics.pdf) |
| A2025-2 | 37 | [官方 PDF](https://github.com/stanford-cs336/assignment2-systems/blob/e495ed740080661ab084914674d3e10048778889/cs336_spring2025_assignment2_systems.pdf) |
| A2025-3 | 6 | [官方 PDF](https://github.com/stanford-cs336/assignment3-scaling/blob/09d205bde59e5c533368c0209dc86a4d5e4323ea/cs336_spring2025_assignment3_scaling.pdf) |
| A2025-4 | 17 | [官方 PDF](https://github.com/stanford-cs336/assignment4-data/blob/5a5f890cd9b72733e6273a596c40b696df3aa9df/cs336_spring2025_assignment4_data.pdf) |
| A2025-5 | 38 | [官方 PDF](https://github.com/stanford-cs336/assignment5-alignment/blob/eb2c562e05802d3c9ee4c4ec2eec43b104d77e9e/cs336_spring2025_assignment5_alignment.pdf) |
| A2026-5-safety | 19 | [官方 PDF](https://github.com/stanford-cs336/assignment5-alignment/blob/eb2c562e05802d3c9ee4c4ec2eec43b104d77e9e/cs336_spring2025_assignment5_supplement_safety_rlhf.pdf) |
| A2026-1 | 47 | [官方 PDF](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf) |
| A2026-2 | 48 | [官方 PDF](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf) |
| A2026-3 | 8 | [官方 PDF](https://github.com/stanford-cs336/assignment3-scaling/blob/03e9372992e913061b9e78b5cfcb62ad8a87de35/cs336_assignment3_scaling.pdf) |
| A2026-4 | 16 | [官方 PDF](https://github.com/stanford-cs336/assignment4-data/blob/0555bea66369872d912652debf10b115ca0688c8/cs336_assignment4_data.pdf) |
| A2026-5 | 39 | [官方 PDF](https://github.com/stanford-cs336/assignment5-alignment/blob/c2734a26308710949fe13226960a1e8cece94b7e/cs336_spring2026_assignment5_alignment.pdf) |

2025 年旧文件在当前 main 上失效，已从官方 git 历史恢复。选取的是 2026-01-01 之前最后一次提交，并在 `sources/manifest.json` 记录固定 commit、下载 URL、SHA-256 与恢复说明。2026 安全补充仍指向 2025 资料，不视作一份独立的新内容。
## 35.4 关键原始论文
论文标题与摘要已核验；正文主要解释依据课程讲义和独立推导，表中链接供进一步追溯，不表示每篇论文均逐页精读。
| 论文 | 原始来源 |
|---|---|
| Attention Is All You Need | [arXiv 1706.03762](https://arxiv.org/abs/1706.03762) |
| RoFormer: Enhanced Transformer with Rotary Position Embedding | [arXiv 2104.09864](https://arxiv.org/abs/2104.09864) |
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | [arXiv 2205.14135](https://arxiv.org/abs/2205.14135) |
| FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | [arXiv 2307.08691](https://arxiv.org/abs/2307.08691) |
| Scaling Laws for Neural Language Models | [arXiv 2001.08361](https://arxiv.org/abs/2001.08361) |
| Training Compute-Optimal Large Language Models | [arXiv 2203.15556](https://arxiv.org/abs/2203.15556) |
| Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer | [arXiv 2203.03466](https://arxiv.org/abs/2203.03466) |
| ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | [arXiv 1910.02054](https://arxiv.org/abs/1910.02054) |
| Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism | [arXiv 1909.08053](https://arxiv.org/abs/1909.08053) |
| Direct Preference Optimization: Your Language Model is Secretly a Reward Model | [arXiv 2305.18290](https://arxiv.org/abs/2305.18290) |
| QLoRA: Efficient Finetuning of Quantized LLMs | [arXiv 2305.14314](https://arxiv.org/abs/2305.14314) |
| DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | [arXiv 2402.03300](https://arxiv.org/abs/2402.03300) |
| Understanding R1-Zero-Like Training: A Critical Perspective | [arXiv 2503.20783](https://arxiv.org/abs/2503.20783) |
| Group Sequence Policy Optimization | [arXiv 2507.18071](https://arxiv.org/abs/2507.18071) |
| LoRA: Low-Rank Adaptation of Large Language Models | [arXiv 2106.09685](https://arxiv.org/abs/2106.09685) |
| Learning Transferable Visual Models From Natural Language Supervision | [arXiv 2103.00020](https://arxiv.org/abs/2103.00020) |
| Sigmoid Loss for Language Image Pre-Training | [arXiv 2303.15343](https://arxiv.org/abs/2303.15343) |
| Visual Instruction Tuning | [arXiv 2304.08485](https://arxiv.org/abs/2304.08485) |
| Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution | [arXiv 2409.12191](https://arxiv.org/abs/2409.12191) |
| Mamba: Linear-Time Sequence Modeling with Selective State Spaces | [arXiv 2312.00752](https://arxiv.org/abs/2312.00752) |
| Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality | [arXiv 2405.21060](https://arxiv.org/abs/2405.21060) |
| DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | [arXiv 2405.04434](https://arxiv.org/abs/2405.04434) |
| Muon is Scalable for LLM Training | [arXiv 2502.16982](https://arxiv.org/abs/2502.16982) |

## 35.5 实际验证记录
本次检查环境：Python 3.12.7，PyTorch 2.11.0+cpu，CPU，未使用 CUDA。
- 最近一次机器验证记录：299 项测试，失败 0，错误 0；整体成功：True。见 `book/validation.json`。覆盖作业约定（BPE 示例、调度、初始化、核算数字）、数值边界（精度、padding、目标归一化、KL 梯度）、恢复、注意力、分布式与统计工具。
- 两个 CPU/Gloo 进程：等长与不等长本地 batch 的平均梯度与单进程全局目标一致；reduce-scatter+all-gather 等于 all-reduce；优化器状态分片的 AdamW 与不分片一致。
- TinyStories CPU 实验（第 20.4 节）：200 MiB 训练文本、10K BPE、8.3M 参数模型、每个完整运行 6.1M 训练 token；6 个学习率、7 个消融配置，另有 3 个基线重复种子。结果见 `book/tinystories-results.json`。
- 合成算术小模型：120 步时 100 道题答对 10%，训练到 1000 步答对 100%；在其上运行的 GRPO 循环见第 18.16 节与 `book/grpo-results.json`。
- 缩放实验为明确标注的合成曲面，用于比较实验设计，不是真实模型的缩放律。
- 前沿章节（第 29–34 章）：14 个开源模型的参数公式与官方 safetensors 元数据逐个精确相等，YaRN 频率与 transformers 参考实现逐元素对照，gpt-oss-120b 的 checkpoint 大小由元数据复算为 60.8 GiB；合成数据蒸馏实验在 CPU 上用 5 个种子运行。结果见 `book/frontier-results.json`，由 `scripts/run_frontier_experiments.py` 重算。
- CUDA/Triton 性能、多机扩展、真实大模型 RLVR 和多模态训练未在当前 CPU 环境中运行；Triton kernel 只在解释器模式下验证了正确性。

验证命令与运行范围见仓库根目录 `README.md`。

## 35.6 进阶章节的依据与边界
第 22–27 章是为独立研究与应用增加的扩展，不归为某年课程的内容：研究设计、数据适配、RAG、推理时计算、压缩、机制可解释性。原始论文的标题摘要和重点相关全文片段已核对，访问与哈希记录见 `sources/advanced-sources.json`。教学公式和代码独立编写，不把简化更新视为完整 DoReMi/GPTQ 等方法复现。
| 进阶资料 | 原始论文 |
|---|---|
| DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining | [arXiv 2305.10429](https://arxiv.org/abs/2305.10429) |
| Don't Stop Pretraining: Adapt Language Models to Domains and Tasks | [arXiv 2004.10964](https://arxiv.org/abs/2004.10964) |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | [arXiv 2005.11401](https://arxiv.org/abs/2005.11401) |
| Dense Passage Retrieval for Open-Domain Question Answering | [arXiv 2004.04906](https://arxiv.org/abs/2004.04906) |
| Lost in the Middle: How Language Models Use Long Contexts | [arXiv 2307.03172](https://arxiv.org/abs/2307.03172) |
| GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | [arXiv 2210.17323](https://arxiv.org/abs/2210.17323) |
| AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | [arXiv 2306.00978](https://arxiv.org/abs/2306.00978) |
| SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models | [arXiv 2211.10438](https://arxiv.org/abs/2211.10438) |
| Locating and Editing Factual Associations in GPT | [arXiv 2202.05262](https://arxiv.org/abs/2202.05262) |
| Towards Best Practices of Activation Patching in Language Models: Metrics and Methods | [arXiv 2309.16042](https://arxiv.org/abs/2309.16042) |
| Training Verifiers to Solve Math Word Problems | [arXiv 2110.14168](https://arxiv.org/abs/2110.14168) |
| Let's Verify Step by Step | [arXiv 2305.20050](https://arxiv.org/abs/2305.20050) |
| Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters | [arXiv 2408.03314](https://arxiv.org/abs/2408.03314) |

## 35.7 前沿章节的依据与边界
第 29–34 章整理 2024–2025 年社区公开的模型、训练方法和数据配方，同样不归为某年课程的内容。模型配置按固定版本保存在 `sources/model-configs/`，下表的总参数取自 Hugging Face safetensors 元数据，与 `examples/architectures.py` 的公式逐个精确相等。各技术报告中引用的数字均对照原文核对过；它们是各团队在各自设定下的结果，本书没有复现这些大规模训练。
| 模型 | 固定版本 | safetensors 总参数 |
|---|---|---:|
| [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json) | `b968826d9c46` | 8,190,735,360 |
| [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json) | `ad44e777bcd1` | 30,532,122,624 |
| [Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/8efa61729e24bd65b1d152b5ab5409052aa80e65/config.json) | `8efa61729e24` | 235,093,634,560 |
| [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/config.json) | `e815299b0bcb` | 684,489,845,504 |
| [moonshotai/Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/config.json) | `fd1984e2b7a3` | 1,026,408,235,864 |
| [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b/blob/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json) | `b5c939de8f75` | 116,829,156,672 |
| [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/config.json) | `6cee5e81ee83` | 20,914,757,184 |
| [allenai/OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B/blob/a1847dff35000b4271fa70afc5db10fd29fedbdf/config.json) | `a1847dff3500` | 1,484,916,736 |
| [allenai/OLMo-2-1124-7B](https://huggingface.co/allenai/OLMo-2-1124-7B/blob/7df9a82518afdecae4e8c026b27adccc8c1f0032/config.json) | `7df9a82518af` | 7,298,617,344 |
| [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/9c7f2fbe84465e40164a94cc16cd30b6999b0cc7/config.json) | `9c7f2fbe8446` | 81,324,862,720 |
| [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/a07cc9a04f16550a088caea529712d1d335b0ac1/config.json) | `a07cc9a04f16` | 3,075,098,624 |
| [mistralai/Mistral-Small-24B-Base-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501/blob/b0a2e4ed093c26997495ae625528f81ea04b749f/config.json) | `b0a2e4ed093c` | 23,572,403,200 |
| [zai-org/GLM-4.5](https://huggingface.co/zai-org/GLM-4.5/blob/cbb2c7cfb52fa128a9660cb1a7a78e017899e115/config.json) | `cbb2c7cfb52f` | 358,337,791,296 |
| [MiniMaxAI/MiniMax-M1-80k](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k/blob/8d1494b1a260e22040d5b9b2eb332eb44500b34d/config.json) | `8d1494b1a260` | 456,089,655,296 |

配置获取日期：2026-10-27。

| 前沿资料 | 原始论文 | 章节 | 核对范围 |
|---|---|---|---|
| Extending Context Window of Large Language Models via Positional Interpolation | [arXiv 2306.15595](https://arxiv.org/abs/2306.15595) | 30 | 摘要 |
| YaRN: Efficient Context Window Extension of Large Language Models | [arXiv 2309.00071](https://arxiv.org/abs/2309.00071) | 30 | 摘要 |
| Efficient Streaming Language Models with Attention Sinks | [arXiv 2309.17453](https://arxiv.org/abs/2309.17453) | 29, 30 | 摘要与正文相关段落 |
| Microscaling Data Formats for Deep Learning | [arXiv 2310.10537](https://arxiv.org/abs/2310.10537) | 34 | 摘要与正文相关段落 |
| Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing | [arXiv 2406.08464](https://arxiv.org/abs/2406.08464) | 33 | 摘要与正文相关段落 |
| DataComp-LM: In search of the next generation of training sets for language models | [arXiv 2406.11794](https://arxiv.org/abs/2406.11794) | 33 | 摘要 |
| The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale | [arXiv 2406.17557](https://arxiv.org/abs/2406.17557) | 33 | 摘要与正文相关段落 |
| FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | [arXiv 2407.08608](https://arxiv.org/abs/2407.08608) | 34 | 摘要与正文相关段落 |
| How to Train Long-Context Language Models (Effectively) | [arXiv 2410.02660](https://arxiv.org/abs/2410.02660) | 30 | 摘要 |
| Scaling Laws for Precision | [arXiv 2411.04330](https://arxiv.org/abs/2411.04330) | 34 | 摘要与正文相关段落 |
| Tulu 3: Pushing Frontiers in Open Language Model Post-Training | [arXiv 2411.15124](https://arxiv.org/abs/2411.15124) | 33 | 摘要与正文相关段落 |
| Gated Delta Networks: Improving Mamba2 with Delta Rule | [arXiv 2412.06464](https://arxiv.org/abs/2412.06464) | 29 | 摘要 |
| Phi-4 Technical Report | [arXiv 2412.08905](https://arxiv.org/abs/2412.08905) | 33 | 摘要与正文相关段落 |
| DeepSeek-V3 Technical Report | [arXiv 2412.19437](https://arxiv.org/abs/2412.19437) | 29, 33, 34 | 摘要与正文相关段落 |
| 2 OLMo 2 Furious | [arXiv 2501.00656](https://arxiv.org/abs/2501.00656) | 29, 33 | 摘要与正文相关段落 |
| DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | [arXiv 2501.12948](https://arxiv.org/abs/2501.12948) | 31, 33 | 摘要与正文相关段落 |
| Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention | [arXiv 2502.11089](https://arxiv.org/abs/2502.11089) | 30 | 摘要 |
| DAPO: An Open-Source LLM Reinforcement Learning System at Scale | [arXiv 2503.14476](https://arxiv.org/abs/2503.14476) | 31 | 摘要与正文相关段落 |
| Gemma 3 Technical Report | [arXiv 2503.19786](https://arxiv.org/abs/2503.19786) | 29, 30 | 摘要与正文相关段落 |
| Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? | [arXiv 2504.13837](https://arxiv.org/abs/2504.13837) | 31 | 摘要与正文相关段落 |
| Qwen3 Technical Report | [arXiv 2505.09388](https://arxiv.org/abs/2505.09388) | 29, 33, 34 | 摘要与正文相关段落 |
| The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models | [arXiv 2505.22617](https://arxiv.org/abs/2505.22617) | 31 | 摘要与正文相关段落 |
| ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models | [arXiv 2505.24864](https://arxiv.org/abs/2505.24864) | 31 | 摘要 |
| MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention | [arXiv 2506.13585](https://arxiv.org/abs/2506.13585) | 29, 30, 31, 32, 34 | 摘要与正文相关段落 |
| Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs | [arXiv 2506.14245](https://arxiv.org/abs/2506.14245) | 31 | 摘要 |
| Kimi K2: Open Agentic Intelligence | [arXiv 2507.20534](https://arxiv.org/abs/2507.20534) | 29, 32, 33 | 摘要与正文相关段落 |
| gpt-oss-120b & gpt-oss-20b Model Card | [arXiv 2508.10925](https://arxiv.org/abs/2508.10925) | 29, 30, 32, 34 | 摘要与正文相关段落 |

## 35.8 数值工具与模型进展的原始来源
新增作业诊断与验收、第 22–34 章综合自测、DPO 共享参数反例、RL 训练/推理一致性、NVFP4、BLT/H-Net、Muon 更新尺度、QAT/2:4、SAE/steering 与学习率重新升温。章节保持 37 章；相关来源和状态如下，教学数值见 `book/review-update-results.json`。工业训练结果仅引用论文，不声明复现。
| 原始来源 | 核对状态 |
|---|---|
| [scaler](https://arxiv.org/html/2510.13786v1) | targeted_text_checked |
| [dsv32](https://arxiv.org/html/2512.02556v1) | targeted_text_checked |
| [fp16](https://arxiv.org/html/2510.26788v1) | targeted_text_checked |
| [kimi_linear](https://arxiv.org/html/2510.26692v1) | targeted_text_checked |
| [olmo2](https://arxiv.org/html/2501.00656v3) | targeted_text_checked |
| [muon](https://arxiv.org/html/2502.16982v1) | targeted_text_checked |
| [blt](https://arxiv.org/html/2412.09871v1) | targeted_text_checked |
| [hnet](https://arxiv.org/html/2507.07955v1) | targeted_text_checked |
| [rewarm](https://arxiv.org/html/2403.08763v1) | targeted_text_checked |
| [sae](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) | targeted_text_checked |
| [steering](https://arxiv.org/abs/2308.10248) | targeted_text_checked |
| [nvfp4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) | targeted_text_checked |
| [qat](https://pytorch.org/blog/quantization-aware-training/) | targeted_text_checked |
| [sparse24](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/) | targeted_text_checked |
| [checkpoint](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/utils/checkpoint.py) | targeted_text_checked |
| [fsdp](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/distributed/fsdp/_fully_shard/_fully_shard.py) | targeted_text_checked |
| [m2_config](https://huggingface.co/MiniMaxAI/MiniMax-M2/raw/04312f0cfbdeefb912575595b7c045512a29293c/config.json) | targeted_text_checked |
| [m2_code](https://huggingface.co/MiniMaxAI/MiniMax-M2/raw/04312f0cfbdeefb912575595b7c045512a29293c/modeling_minimax_m2.py) | targeted_text_checked |

## 35.9 理论边界与实现契约的依据
第 00–34 章正文中的理论边界与实现契约由已获取资料和小规模数值实验支撑。`book/chapter-review-results.json` 保存 NoPE、PPO 等受控反例，`chapter_invariants` 字段记录分块 softmax 合并、Delta rule、CLIP 跨批负例、非均匀 MoE 路由、优化器零梯度语义、预条件梯度与 YaRN 频率计数，可用 `python scripts/run_chapter_review.py` 重算。独立关系检查见 `tests/test_chapter_invariants.py`，手算题见第 21.10 节；这些结果不代表完整模型训练或 GPU 性能验证。核验范围未包括无公开讲义的口述，也未重跑工业规模训练。
`interfaces` 字段补充 UTF-8 流式解码、独立文档 packing、线性注意力低精度、按簇配对 bootstrap、终止/截断 GAE、校准与密度比反例，以及直接由函数计算的 RoPE 频率曲线；相应测试见 `tests/test_learning_interfaces.py`。这些是接口与数学关系的教学核验，原有训练实验仍保留各自的日期与运行条件。

| 原始来源 | 核验范围 |
|---|---|
| [nsys](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | 官方 CLI 中 capture-range 与具名 NVTX 范围参数 |
| [llava](https://llava-vl.github.io/blog/2024-01-30-llava-next/) | LLaVA-NeXT 的 AnyRes 官方介绍 |
| [nope](https://arxiv.org/html/2305.19466v2) | 第 5 节因果 NoPE 的位置表示条件 |
| [rope](https://arxiv.org/abs/2306.15595) | 位置插值原论文摘要；频率分析由公式独立推导 |
| [entropy](https://arxiv.org/abs/2505.22617) | 熵机制论文摘要；通用熵微分独立推导并测梯度 |
| [harmony](https://raw.githubusercontent.com/openai/harmony/main/README.md) | 官方格式示例中的通道与接收者字段 |
| [fineweb](https://arxiv.org/html/2406.17557v2) | 全文中的 6,000 H100 GPU 小时；70B 主项成本独立复算 |

## 35.10 深入机制的原始资料
以下资料用于补充词表损失、MLA/扫描、LoRA、后训练、检索、压缩与低精度。核验日期 2026-10-31；读取范围按表列出，不等于每篇逐页精读或复现原论文。数学教学实现见 `examples/mechanisms.py`，数值结果见 `book/mechanism-results.json`。

| 原始资料 | 本书章节 | 核对内容 |
|---|---|---|
| [CS336 Spring 2026](https://cs336.stanford.edu/) | 00、28 | 官网课程范围与 AI 工具条款；链接的完整 Google 文档本次未能访问 |
| [Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/html/2411.09009v1) | 08、10 | 交叉熵分解、前后向重算和梯度过滤 |
| [Cut Cross-Entropy 作者实现](https://raw.githubusercontent.com/apple/ml-cross-entropy/main/README.md) | 10 | 不同累加/过滤实现模式与精确参考 |
| [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/pdf/2312.03732) | 16 | PDF 第 3 节 rank 缩放与假设 |
| [DeepSeek-V2](https://arxiv.org/html/2405.04434v5) | 07、29 | §2.1.2–2.1.3 latent K/V、投影吸收与解耦 RoPE |
| [Transformers are SSMs](https://arxiv.org/html/2405.21060v1) | 07 | 结构化矩阵、递推、scan/chunk 的区别 |
| [Muon is Scalable for LLM Training](https://arxiv.org/html/2502.16982v1) | 05 | 正交化、形状尺度与分布式边界 |
| [Megatron-LM](https://arxiv.org/html/1909.08053v4) | 11 | 输出词表并行与交叉熵通信 |
| [DistServe](https://arxiv.org/html/2401.09670v1) | 12 | prefill/decode 分离、传输成本与 goodput |
| [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/) | 12 | 父块哈希、额外身份与只缓存完整块 |
| [PyTorch v2.11 SDPA 源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/nn/functional.py) | 04、10 | dropout_p、mask 和非方阵 causal 约定 |
| [PyTorch v2.11 AMP 示例](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/docs/source/notes/amp_examples.rst) | 05、11 | 梯度累积、unscale、step 与 scale 更新边界 |
| [PyTorch FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) | 11 | 重分片、梯度同步与版本接口区别 |
| [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | 05 | 摘要；累积衰减公式由本文独立推导 |
| [DoReMi](https://arxiv.org/html/2305.10429v4) | 23 | min-max 目标、逐 token 正超额损失与算法 1 |
| [Data Selection for Language Models via Importance Resampling](https://arxiv.org/abs/2302.03169) | 14 | 摘要；过滤后的域分布由条件概率独立推导 |
| [How to Train Long-Context Language Models (Effectively)](https://arxiv.org/html/2410.02660v1) | 30 | 长文档、短数据混合与训练/评估长度 |
| [Self-RAG](https://arxiv.org/html/2310.11511v1) | 24 | 自适应检索与反思信号的训练/推理角色 |
| [ColBERT](https://arxiv.org/html/2004.12832v2) | 24 | late interaction 与 MaxSim 聚合 |
| [Let’s Verify Step by Step](https://arxiv.org/html/2305.20050v1) | 25 | 步骤监督与 outcome 监督的区别 |
| [Tülu 3](https://arxiv.org/html/2411.15124v4) | 17、33 | 提示池、去污染与后训练协议 |
| [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) | 27 | 替代模型、干预验证、缺失 attention circuits 的局限 |
| [QuaRot](https://arxiv.org/html/2404.00456v1) | 26 | 正交旋转、不变量与归一化/非线性位置 |
| [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/html/2303.15343v3) | 19 | 算法 1 的归约与 bias/负例先验 |
| [Byte Latent Transformer](https://arxiv.org/html/2412.09871v1) | 02 | incremental patching、next-byte 熵与局部/全局模块 |
| [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/html/2507.07955v1) | 02 | 动态边界、因果扩展与压缩率训练 |
| [Transformer Engine FP8/FP4 Primer](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html) | 34 | NVFP4 二维权重尺度、随机舍入与 Hadamard 配方 |
| [Policy Invariance under Reward Transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) | 32 | PDF 第 1、3、4 页：潜势塑形定理与终止前提（已查看图像） |
| [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438) | 17 | PDF 的 GAE 构造、价值误差与偏差/方差 |
| [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/html/2211.17192v2) | 12 | 接受率与实际轮次成本 |
| [RULER: What’s the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/html/2404.06654v1) | 30 | 有效上下文、合成任务及评估边界 |
| [Rethinking the Effective Sample Size](https://arxiv.org/abs/1809.04129) | 18、31 | 摘要：常见 ESS 近似的多重假设与局限；书中将其限定为权重集中度诊断 |
| [transformers v4.57.0: Llama 4 modeling and configuration (no_rope_layers, chunked attention, attention temperature tuning)](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/llama4/modeling_llama4.py) | 29、30 | 源码：默认每 4 层 1 层 NoPE；RoPE 层用 8,192 块内分块因果掩码；NoPE 层 query 乘 1+0.1·ln(1+floor((t+1)/8192))。模型权重需授权，未读取 |
| [Scalable-Softmax Is Superior for Attention](https://arxiv.org/abs/2501.19399) | 30 | 摘要：softmax 最大值随候选数增加趋于 0、注意力随上下文变平；用于说明 Llama 4 NoPE 层温度修正的出处 |
| [Negative Results for SAEs On Downstream Tasks and Deprioritising SAE Research (GDM Mech Interp Team Progress Update #2, 2025-03-26)](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28) | 27 | 全文相关段落（2026-11-01 浏览器读取）：有害意图检测任务的数据规模、稠密线性探针与 SAE 探针的分布内/OOD 对比、重建后探针的 AUROC 下降、降低 SAE 研究优先级的结论与保留意见；非同行评审的活动网页，站点拒绝脚本下载，未记录哈希 |
| [Position: Use Sparse Autoencoders to Discover Unknowns](https://arxiv.org/abs/2506.23845) | 27 | 摘要：区分"对已知概念的操作"与"发现未知概念"，用于调和 SAE 的负面与正面结果 |

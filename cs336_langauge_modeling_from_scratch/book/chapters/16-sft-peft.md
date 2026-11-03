# 16｜中期训练、指令微调与参数高效适配

## 16.1 每个训练阶段在改变什么

预训练在广泛文本上学习分布；中期训练把数据向代码、数学、长上下文或任务形式倾斜；SFT 用示范教模型在给定指令下产生期望回答。三者不一定有严格边界，常见差别是数据分布、损失掩码、学习率与序列结构。[2026 第十五讲](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_15.pdf)

继续预训练可以吸收领域知识，SFT 可激活已有能力、约束输出方式，也可能学习新模式。不能把“SFT 只教格式、永远不学知识”当成定律。实际效果受示范覆盖、模型容量、重复次数和训练步数影响。

## 16.2 response-only loss 的正确对齐

对于 prompt $x$ 和 response $y$，条件似然为

$$\mathcal L_{SFT}=-\sum_{t=1}^{|y|}\log\pi_\theta(y_t\mid x,y_{<t}).$$

将对话串成完整 token 序列后，input/target 相差一位。掩码应标记目标 token 是否属于 assistant response，包括约定的结束 token；它不是简单标记输入位置是否属于 assistant。

```text
完整串：  用户标记  问题  助手标记  答案1  答案2  EOS
输入：    用户标记  问题  助手标记  答案1  答案2
目标：    问题      助手标记  答案1  答案2  EOS
loss mask: 0       0         1      1      1
```

注意 `encode(prompt)+encode(response)` 不一定等于 `encode(prompt+response)`，因为分词可能跨边界合并。聊天模板若用专门控制 token 提供稳定边界会更容易对齐；否则需要完整编码后的 offsets 或明确的边界处理策略。

## 16.3 归一化定义了训练偏好

按所有 response token 平均，会让长回答贡献更多总权重；先逐序列平均再逐样本平均，则每个样本总权重相同。两种目标都可合理，但效果不同。多轮对话应明确训练所有 assistant 回合，还是只训练最后一轮。

packing 后的注意力 mask 与 loss mask 相互独立。前者规定看什么，后者规定哪些预测接受监督。仅把 prompt 的 loss 设为零，不会阻止 response 读取 prompt，这正是条件生成所需行为。

## 16.4 示范数据从哪里来

2026 年第十五讲比较了几类常见指令数据：FLAN 把大量现成 NLP 任务改写成指令格式，覆盖广但回答简短、像"在答题"而不是在对话；Alpaca 用强模型按种子指令生成，风格更像助手，但继承了教师的错误；ShareGPT/Vicuna 来自真实用户与 ChatGPT 的对话；OpenAssistant 由志愿者撰写和标注；Tulu 3 等则是多来源混合并精心过滤。数据来源决定了模型的"口吻"：同样的知识，训练在 FLAN 上与训练在对话数据上，模型说话的方式可能明显不同；具体程度应由固定模板和任务的对照验证。

以下是讲义中基于特定研究的经验提醒，不推翻 16.1 节 SFT 也能学习新信息的边界：SFT 最擅长**提取**预训练已有的行为，而不是加入新行为；加入模型不知道的知识，即使内容正确，有时反而有害，因为它在教模型"自信地说出没有依据的内容"；少量正确类型的数据（安全、指令遵循、风格）影响很大，例如约 500 条安全示范就能显著改善安全行为，但长尾能力仍需要更多数据。

## 16.5 示范质量与幻觉

长度、口吻、引用形式和拒绝策略都可以被 SFT 快速学到。如果示范鼓励每题都回答，即使模型缺少相关知识也可能学习自信补全。若示范包含虚构来源，模型会模仿来源格式而不是自动验证其真实性。

构建数据时要区分能力示范、事实知识、表达风格和边界处理；按任务分层评估。用少量高质量数据的成功案例不能证明数据量总不重要，尤其对新能力、长链推理或多语言迁移。

安全 SFT 需要覆盖正常请求、风险请求与边界请求，同时衡量误拒和遗漏。人工标注指南、评审一致性与分歧应成为数据记录的一部分。2024 年课程对人工示范、偏好标注和风格偏差的讨论值得与新版一起保留。[2024 第十五讲](https://github.com/stanford-cs336/spring2024-lectures/blob/2eab561d4fc5f0bd77ab68afc85dd563a0b107ba/nonexecutable/Lecture%2015%20-%20Alignment%20by%20SFT.pdf)

## 16.6 LoRA 从低秩更新出发

对 $W\in\mathbb R^{d_{out}\times d_{in}}$，冻结 $W$，学习

$$W'=W+\frac{\alpha}{r}BA,\quad A\in\mathbb R^{r\times d_{in}},\quad B\in\mathbb R^{d_{out}\times r}.$$

新增参数为 $r(d_{in}+d_{out})$，远小于全矩阵。典型初始化 A 随机、B 为零，让初始 $\Delta W=0$。此时第一步对 A 的梯度为零，但 B 可收到非零梯度，之后两者都能学习；若两者都为零且没有其他更新项，对 BA 的任务梯度会同时为零，无法开始学习。

LoRA 主要节省可训练参数的梯度和优化器状态；冻结权重仍需加载，反向传播仍需经过网络并保存/重算激活。部署时在适当精度下可以合并 $BA$ 到 $W$，但多适配器切换、量化权重和精度误差会影响实际方案。[LoRA](https://arxiv.org/abs/2106.09685)

rank 限制的是单个更新矩阵，而不是整个模型输出行为只有 rank-r 的简单自由度。不同层、不同投影的低秩更新经过非线性组合可产生复杂变化。rank、目标模块和训练预算应一起调。

量化后的合并还要再检查一次函数。QLoRA 训练时实际使用的是反量化基座 $\widetilde W$ 加高精度更新 $\Delta W$；合并后若重新量化，推理权重变成 $Q(\widetilde W+\Delta W)$，一般不等于 $\widetilde W+\Delta W$。例如间距 0.5 的量化网格，基座为 1、更新为 0.1，重新量化可能又得到 1，适配更新被抹掉。合并前后应比较固定输入的 logits 与任务指标，而不只检查保存的参数名称。

## 16.7 QLoRA 省的内存在哪里

QLoRA 将冻结基础权重量化存储，计算时反量化到合适 dtype，LoRA 因子保持可训练精度。NF4 是针对近似正态权重分布设计的非均匀量化编码，double quantization 进一步压缩分组 scale 等量化常数。[QLoRA](https://arxiv.org/abs/2305.14314)

**量化常数的账。** QLoRA 用块大小 64 的 NF4：每 64 个权重共享一个 absmax 缩放常数。若常数存成 FP32，每参数额外开销是 $32/64=0.5$ bit；double quantization 把这些常数再按 256 个一组量化成 8 位、每组再存一个 FP32 常数，开销降到 $8/64+32/(64\times256)\approx0.127$ bit。7B 模型的冻结权重因此约为 $7\times10^9\times(4+0.127)/8\approx3.6$ GB。反量化后用 BF16 计算，LoRA 因子和它们的 AdamW 状态按可训练参数量另算。

4-bit 权重不意味着完整训练状态每参数只需 0.5 bytes；还有 scales、LoRA 权重、梯度、优化器、激活和临时反量化缓冲。是否使用 paged optimizer 会影响内存峰值与 CPU/GPU 传输。

## 16.8 一个适配实验的最小对照

固定基础权重、训练数据、模板、预算与评估协议，对比零样本、full SFT、不同 rank LoRA。分别记录峰值显存、训练时间、保存体积、任务分数与通用能力回退。不要仅以“可训练参数少”推断质量相同或速度更快。

**练习。** 用户 prompt 部分不计算 loss，为什么它仍影响参数梯度？

**解答。** response 的预测通过注意力读取 prompt 的隐藏表示，链式法则会沿这些依赖传播。mask 移除了直接预测 prompt 的监督项，并没有切断 prompt 对回答的条件作用。

## 16.9 一个 token 边界就能改变监督含义

EOS 与 PAD 可能共用同一个 ID，此时不能用 `labels != pad_id` 构造 loss mask，否则真实回答的 EOS 也被删掉，模型会学不到正确停止。应使用真实长度、角色边界或独立 attention/loss mask 来区分“有意义的结束标记”和“填充占位”。

监督的第一个回答 token 对应 prompt 最后一个 token 位置的 logits。最后一个 EOS 对应回答末 token 的 logits。把 mask 按输入位置而非目标位置对齐，会多训练一个模板位置或漏掉一个回答 token。`examples/objectives.py` 明确要求 logits、labels、mask 已经移位对齐，并分别实现按 token 与按序列平均的 SFT loss。

## 16.10 蒸馏：学习一个答案，还是学习一整个分布

硬标签蒸馏把教师生成的轨迹当作 SFT 示范。软标签蒸馏在同一上下文、同一 token 空间上，让学生分布逼近教师分布。给定温度 $\tau$，定义

$$q_\tau=\operatorname{softmax}(z_{teacher}/\tau),\quad p_\tau=\operatorname{softmax}(z_{student}/\tau),$$

$$L_{KD}=\tau^2D_{KL}(q_\tau\Vert p_\tau).$$

较高温度暴露非首选 token 之间的相对关系；$\tau^2$ 用于补偿温度对梯度量级的影响，是常见约定而非必须常数。教师分布要 detach。可以与原始标签交叉熵混合，但必须明确定义两个损失的权重与 token mask。[知识蒸馏原论文](https://arxiv.org/abs/1503.02531)

对未缩放的学生 logit $z_j$，梯度为 $\partial L_{KD}/\partial z_j=\tau(p_{\tau,j}-q_{\tau,j})$。不乘 $\tau^2$ 时则为 $(p_{\tau,j}-q_{\tau,j})/\tau$。当温度远大于两侧 logit 差异，中心化后 $p_{\tau,j}-q_{\tau,j}\approx(\widetilde z_{student,j}-\widetilde z_{teacher,j})/(V\tau)$，才看出这个补偿使梯度近似不随温度缩小；有限温度下它不保证相同更新，温度同时改变了被拟合的概率分布。

不同 tokenizer 的词表位置不对应，不能直接对 logits 作 KL。可使用文本轨迹蒸馏，或另行构造共同事件空间。软标签方法通常还要计算整个词表分布，代价不同于只保存教师最终文本。

[objectives.py](../../examples/objectives.py) 的 `distillation_loss` 实现同词表、按有效 token 平均的这一目标，测试检查教师无梯度、padding 无贡献、学生提高教师偏好的类别概率。它不是跨 tokenizer 蒸馏算法。

蒸馏也有上下文分布问题：教师在自己的成功轨迹上给出的分布，不等于学生犯错后实际遇到的状态。on-policy distillation 让教师评价学生产生的前缀，更贴近学生状态分布，但需要额外教师调用；不能把“更多教师 token”隐藏在相同训练步数的比较里。

**练习 2．** 对一个 $4096\times4096$ 的投影矩阵做 rank 16 的 LoRA，新增多少参数？若模型有 32 层、每层对 Q/K/V/O 四个投影都加 LoRA，总共多少？

**解答。** 每个矩阵 $16\times(4096+4096)=131{,}072$；$32\times4\times131{,}072\approx16.8$M，约为 7B 模型的 0.24%。可训练参数少，但前向与反向仍要经过全部冻结权重。

**练习 3．** 一批 SFT 数据中，回答长度为 10 和 90 token 的两个样本，按 token 平均与按序列平均时，短样本每个 token 的梯度权重各是多少？

**解答。** 按 token 平均：每个 token 权重 $1/100$。按序列平均：短样本每个 token 为 $\frac12\times\frac1{10}=1/20$，长样本为 $\frac12\times\frac1{90}=1/180$。后者让短回答的每个 token 权重大 9 倍。

## 16.11 社区进展：蒸馏成了小模型的主路径

DeepSeek-R1 的蒸馏小模型只做 SFT、不做 RL，Qwen3 的小模型则先做离策略蒸馏、再做对齐教师 logits 的在策略蒸馏，具体数字见第 33.4 节。后者正是 16.10 节"教师应在学生的前缀上给出分布"的大规模实践。

第 33.5 节用一个能完全验证的玩具实验比较了教师采样、拒绝采样过滤和软标签蒸馏。agent 轨迹的 SFT 掩码（只对模型生成的段计算损失）见第 32.1 节。

## 16.12 LoRA 的 rank 为什么也是优化尺度

写 $\Delta W=sBA$，初始 B=0，A 的元素独立、零均值、方差 $\sigma_A^2$。设任务对 W 的梯度为 G，单步 SGD 有 $B_1=-\eta sGA^\top$，因而

$$\Delta W_1=-\eta s^2GA^\top A,\qquad \mathbb E[\Delta W_1]=-\eta s^2r\sigma_A^2G.$$

若固定 $\alpha$ 使用 $s=\alpha/r$，该理想化首步函数更新随 $1/r$ 缩小；$s=\alpha/\sqrt r$ 则在此假设下不随 r 缩小。这是 rsLoRA 的 rank stabilization 动机。它不保证所有 Adam、初始化、数据规模下最优超参数不变，也不等于 rank 越大越好。[rsLoRA](https://arxiv.org/abs/2312.03732)

冻结 W 只去掉 $dW$ 的保存/更新，不自动去掉传播给更早层的梯度（沿用本节列向量约定，$g_x=W^\top g_y$）。若在冻结主干上包 `no_grad()`，位于更早层的 adapter 也可能断掉梯度。应逐模块检查哪些参数可训练，以及 loss 对它们是否非零；共享权重、多 adapter 与量化合并要用合并前后 logits 对照。

**自测。** r 从 16 增到 64、$\alpha,\eta$ 不变，在上述首步 SGD 模型中，两种缩放分别如何变化？

**解答。** $\alpha/r$ 的期望有效更新变为 1/4；$\alpha/\sqrt r$ 保持同量级。若只比较固定 LR 的最终分数，可能混入更新尺度差异。

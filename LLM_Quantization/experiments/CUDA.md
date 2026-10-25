# Colab SSH：T4 / L4 / A100 CUDA 实验

这组实验面向通过 SSH 登录 Colab、拉取仓库后直接运行的场景。使用合成张量，无需下载模型。默认工作量按 T4 设计，L4/A100 使用同一命令；每次记录实际 GPU、时钟上限、L2 大小和 PyTorch/CUDA/Triton 版本。

当前代码已通过 CPU 格式检查、数值参照和运行记录测试。Triton 内核还用 `TRITON_INTERPRET=1` 在 CPU 上逐值核对，并离线编译到 SM75/80/89/90。**这些检查都不涉及性能，T4/L4/A100 还没有实测。** 结论以你自己的报告和跳过/失败记录为准。它与原有 27 个 FP64 逐章实操分开计数。

## 拉取与安装

首次使用时用自己的仓库地址下载项目；已有项目时，在项目根目录运行 `git pull --ff-only` 更新。以下命令都在仓库根目录执行：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import triton; print(triton.__version__)"   # Linux CUDA 版 PyTorch 通常自带
python -m pip install -r requirements-cuda.txt
python -B experiments/cuda_lab.py --list
python -B experiments/cuda_lab.py --profile smoke --write-results --output-dir runs/cuda
```

`requirements-cuda.txt` 要求 PyTorch 2.3+ 和 NumPy；版本满足时保留 Colab 已有的 CUDA PyTorch。如果上面的 CUDA 可用性为 `False`，先确认 SSH 终端用的是 Colab GPU 运行时对应的 Python。不要套用 CI 的 CPU PyTorch 安装命令。

bitsandbytes 是可选依赖，未安装时相关路径会明确跳过，其他实验照常运行。`int8` 和 `decode_stack` 中的 LLM.int8()/NF4 路径需要它：

```bash
python -m pip install bitsandbytes
python -B experiments/cuda_lab.py --case int8 decode_stack --profile smoke --write-results --output-dir runs/cuda
```

安装和首次 CUDA 初始化不计入速度结果，Triton 首次编译在预热和配置选择阶段完成。依赖的实际版本会写进报告；升级版本或更换 GPU 后应重新跑 smoke。

## 十组实验

`--case` 可以一次给多个名称，按列出的顺序执行；`all` 执行全部。

| `--case` | 对照内容 | 重点观察 | 依赖 |
|---|---|---|---|
| `bandwidth` | 远大于 L2 的缓冲区上做读、写、复制 | 本轮可达显存带宽，作为其他路径“带宽比例”的分母 | 无 |
| `gemm` | FP32、FP16；SM80+ 另测 BF16 和请求 TF32 的 FP32 | 小 M 与大 M 的吞吐差异、精度误差、计算类型 | 无 |
| `int4` | FP16 原权重、提前解码的 FP16、每次先解码再 GEMM；group 32/128 | 实际字节数、尺度成本、解码耗时、临时显存 | 无 |
| `w4a16` | cuBLAS FP16、**同模板 Triton FP16**、**Triton 融合 W4A16**、先解码再 GEMM，扫描 M | decode→prefill 时 4 bit 收益如何随 M 消失；融合与先解码的差别 | Triton |
| `lowp_gemm` | FP16、`torch._int_mm` INT8；SM89+ 另测 `torch._scaled_mm` FP8 | 原生低精度计算的吞吐，以及在线激活量化的成本 | 无 |
| `attention` | 显式 FP32 logits，与强制 MATH、EFFICIENT、FLASH 的 SDPA | prefill/decode 的延迟与数值差异，后端是否适配 | 无 |
| `kv` | FP16 缓存、提前解码的缓存、先解码 INT4 再 Attention；recent 0/128 | K 沿时间分组、V 沿特征分组、最近窗口成本、整体解码开销 | 无 |
| `kv_fused` | SDPA、**同模板 Triton FP16 split-K**、**Triton 融合 INT4 KV**、先解码再 SDPA，扫描长度 | 长上下文 decode 中 KV 读取字节对延迟的影响 | Triton |
| `int8` | bitsandbytes INT8 GEMM、输出反量化、在线激活量化，逐级计时 | 纯 GEMM 与完整线性路径的差别 | bitsandbytes |
| `decode_stack` | Qwen3-0.6B 形状 28 层线性骨架：FP16、W4A16、LLM.int8()、NF4；eager 与 CUDA Graph | 一次 decode 步中，启动开销、权重字节和量化实现各占多少 | Triton；bitsandbytes 可选 |

### 新增实验分别回答什么

**`bandwidth`：先测出这块卡的实际上限。** 第 09 章的 roofline 估算需要“有效带宽”。厂商峰值和实际可达值差别很大，T4 还可能受功耗和时钟限制。其他路径的 `bandwidth_fraction` 都以本轮 `read_sum` 为分母，用来判断某条路径是否已接近带宽上限。

**`w4a16`：weight-only 4 bit 什么时候真的更快。** 原有 `int4` 只测先把 INT4 解回完整 FP16 矩阵再调用 cuBLAS 的路径，它读写的字节比 FP16 原权重还多，任何 M 下都会更慢。`w4a16` 增加了在寄存器内解码的 Triton 内核，读取的字节约为 FP16 的 28%（g128）。它同时提供一个**同模板 FP16 内核**，分块、网格、累加方式完全相同，只把权重读取换成 FP16，两者的差异就只来自字节数和解码开销。M=1 时预期接近带宽上限并显示 4 bit 收益；M 增大后，计算和解码开销占主导，收益会缩小甚至反转。拐点出现在哪个 M、是否出现，由实测决定。报告中的 `tensor_core_mma` 读取编译后的 PTX：T4 上 Triton 的 `tl.dot` 不生成 `mma` 指令（SM75 走 FMA 路径），因此 T4 的大 M 结果不能代表 Tensor Core 能力。

**`lowp_gemm`：不依赖第三方库的原生低精度计算。** `torch._int_mm` 调用 cuBLASLt INT8（T4/A100/L4 都支持）。L4、H100 等 SM89+ 设备再测 FP8 E4M3 的 `torch._scaled_mm`。两条路径都分别报告“仅 GEMM”“GEMM+反量化”和“含在线激活量化”。如果 W8A8 只在纯 GEMM 上快，加上 eager 量化就变慢，说明收益依赖算子融合。INT8 结果中的一块完整归约会与 CPU int64 精确核对。

**`kv_fused`：长上下文 decode 读的是缓存。** split-K decode attention 把序列切成多段，每段在线 softmax，最后用第 08 章推导的分段合并公式（按 `exp(m_s - max m)` 重新加权）得到结果。INT4 前缀在内核内解码，最近 128 个 token 保留 FP16，作为额外一段参与合并。同模板 FP16 内核给出只改变读取字节时的对照，SDPA 是现成实现的参照。

**`decode_stack`：把单层收益放回一次 decode 步。** 用 Qwen3-0.6B 的 7 个线性层形状（hidden 1024、MLP 3072、Q 2048、KV 1024）组成 28 层骨架，加入 RMS 归一化、SiLU 门控和残差，省略 Attention、RoPE、KV 写入、embedding 与 lm_head。它直接对应 [2026-10-15 T4 记录](archive/colab_t4_2026-10-15/analysis.md)：当时 LLM.int8() 比 FP16 慢 73%，NF4 慢 27%，分析中提出“小模型下额外算子调度开销超过节省的权重读取”，但没有证据。这里可以把三类开销拆开看：
- `fp16_eager` 与 `fp16_cuda_graph` 的差值是 Python/启动开销；
- `triton_w4a16_cuda_graph` 与 `fp16_cuda_graph` 的比值是去掉启动开销后的字节收益；
- `bnb_int8_eager`、`bnb_nf4_eager` 与 `fp16_eager` 的比值，对应历史记录中的减速是否在线性层上复现。

骨架只覆盖线性层，结果不能直接换算成整模型 token/s，完整模型仍用 Qwen3 Notebook 测量。

### 实现边界

`int4` 和 `kv` 真正把两个编码打包进一个 uint8，并存储 FP16 组尺度；执行时先用 PyTorch CUDA 解码回 FP16，再做浮点计算。`w4a16` 和 `kv_fused` 读取同一份物理编码，在 Triton 寄存器中解码，**仍然以 FP16/FP32 计算，不是原生 INT4 Tensor Core 指令**。GEMV 以及与组对齐的 GEMM 分块先用整数码求和，再乘一次组尺度，这是常见 W4A16 内核的做法。与先解码的权重相比，差异只在 FP16 舍入量级，报告中 `error_vs_decoded` 记录这一差异。小 M 时可选 split-K：各段以 FP32 原子累加，累加顺序不固定，同一输入的重复结果可能在最后一位上不同。

Triton 的分块配置在正式计时外短测选出，结果写在每行的 `config` 和 `tuning` 中，淘汰原因也会保留。这是固定候选集上的选择，**不代表 Marlin、AWQ、GPTQ 等生产内核能达到的水平**。比较时应首先看同模板对照。

`kv` 和 `kv_fused` 测的都是静态缓存快照，不包含流式追加、分页或 RoPE；流式状态的正确性见第 08 章实操。

`int8` 使用 bitsandbytes 的 INT8 输入和 INT32 累加结果，并取一小块完整归约与 CPU int64 结果精确核对。形状满足该后端的归约对齐要求，并拒绝它报告的 FP32 回退，确切指令仍需 profiler 确认。该实验把离群阈值设为 0、关闭离群分流，**不是完整 LLM.int8() 质量复现**；完整的阈值 6.0 路径在 `decode_stack` 中。

T4 对 BF16/TF32 路径记录跳过，非 SM89+ 设备对 FP8 记录跳过。Attention 能否运行由当前 PyTorch 的设备/形状能力检查决定。指定的 SDPA 后端单独启用，不允许换成另一个后端冒充成功。能力检查通过后出现的执行错误记为失败，不会被隐藏成“不支持”。TF32 只是向后端提出的精度请求，实际指令需结合 profiler 确认。

### 计时与 L2 缓存

GPU 的 L2 缓存会让小输入的重复调用看起来比显存带宽还快：T4 的 L2 为 4 MB，A100 为 40 MB。所以涉及字节对比的实验都在**每次计时调用前清空 L2**（`l2_mode=flush`）：写一块至少 4 倍 L2 的缓冲区，清空操作排在计时事件之前、不计入计时，每次调用各有一对 CUDA Event。这种模式不记录 Wall 时间。`bandwidth`、`decode_stack` 使用连续调用（`l2_mode=warm`）：前者的缓冲区本身远大于 L2；后者 28 层的总权重约 0.9 GB，下一次调用开始时 L2 中的数据已被替换，而连续调用正好保留了 CUDA Graph 和 eager 的启动开销差别。

### 工作量与运行命令

| 设置 | `smoke` | `full` |
|---|---|---|
| 每条路径预热 / 计时轮数 / 每轮调用数 | 3 / 5 / 5 | 5 / 7 / 20 |
| 带宽缓冲区 | max(256 MiB, 16×L2) | max(1 GiB, 16×L2) |
| GEMM `M,N,K` | M=1/128，N=K=1024 | M=1/32/512，N=K=4096 |
| INT4 权重 `N,K` | 512×1024 | 2048×4096 |
| W4A16 `N×K`；M；group | 2048×2048；1/16/128；128 | 4096×4096 与 11008×4096；1/4/16/64/256；32/128 |
| INT8/FP8 `M,N,K` | 32/512 × 2048 × 2048 | 32/512/2048 × 4096 × 4096 |
| Attention `B,H,T,D` | 1×8×256×64 | 1×8×1024×64 |
| KV 缓存 `B,H,T,D` | 1×8×512×64 | 1×8×4096×64 |
| 融合 KV `B,H,D`；长度 | 1×8×128；1024/4096 | 1×8×128；2048/8192/32768 |
| decode 骨架 tokens | 1 | 1/16 |

新增实验的形状集中在 `cuda_cases.py` 的 `SIZES` 表中，修改一处即可。

```bash
# 先分别跑通需要的组
python -B experiments/cuda_lab.py --case bandwidth w4a16 --profile smoke --write-results
python -B experiments/cuda_lab.py --case kv_fused decode_stack --profile smoke --write-results

# 跑通后扩大同一轮全部实验；需要更久，不宣称固定耗时
python -B experiments/cuda_lab.py --profile full --write-results --output-dir runs/cuda-full

# 单项 profiler：额外的一次调用，不混入正式计时；trace 文件可能较大
python -B experiments/cuda_lab.py --case w4a16 --trace --write-results

# 验证原有章节的 CUDA 数值路径
python -B experiments/chapter_demo_lab.py --device cuda --write-results --output-dir runs/chapters
python -B -m unittest discover -s tests -p test_demo_torch_kernels.py -v

# 新增实验的格式检查、Triton 解释器核对和 GPU smoke 集成测试
python -B -m unittest discover -s tests -p test_cuda_lab.py -v
```

用 `--device cuda:1` 可以指定第二张卡，`--seed 43` 可以改变合成输入。不同 GPU 或版本不保证随机流逐位相同，因此比较 T4/L4/A100 时，要同时保留种子、形状、依赖和误差。

## SSH 断开与取回结果

如果服务器装有 tmux，可以先进入会话再运行：

```bash
tmux new -s quant-cuda
python -u -B experiments/cuda_lab.py --profile full --write-results --output-dir runs/cuda-full
# Ctrl-b 然后 d：退出终端但保留会话；重新登录后恢复
tmux attach -t quant-cuda
```

tmux 能保留 SSH 会话中的进程，但不能阻止 Colab 回收运行时。每完成一条测量都会更新 JSON，中断时已有结果会保留。运行结束后及时把日志和结果目录取回本地：

```bash
# 在本地执行；替换主机别名和实际远程路径
scp -r colab:/content/你的仓库/runs/cuda-full ./colab-results
```

## 如何读报告

不加 `--write-results` 时只打印。加上后默认保存在 `runs/cuda/<时间戳>_<run_id>/`。`--output-dir` 指定的是父目录，每次运行自动创建新目录，避免覆盖上一次的成功记录。

| 文件 | 内容 |
|---|---|
| `cuda_results.json` | 环境、参数、每轮 CUDA/Wall 时间、误差、存储、峰值、Triton 配置、关键对照、跳过与失败原因 |
| `summary.csv`、`summary.md` | 扁平对照表、关键对照比值和阅读摘要 |
| `sources/` | 本次执行对应的源码、说明与测试快照 |
| `manifest.json` | 文件 SHA-256；内部一致性不等于真实性或质量保证 |
| `traces/` | 仅 `--trace` 时生成的 Chrome profiler 文件 |

运行状态有 `completed`、`completed_with_skips`、`skipped`、`partial`、`failed`、`interrupted` 六种。`completed_with_skips` 表示适用的路径都成功，但有未执行的项目；只运行了缺少后端的可选实验时为 `skipped`。任何一条路径失败都会使状态变为 `partial` 或 `failed`，并返回非零退出码，不能把“目录存在”当作成功。强制杀掉进程时状态可能停留在 `running`，以 JSON 为准。

`summary.md` 的“关键对照”列出同一实验内成对路径的中位数比值（参照/候选，大于 1 表示候选更快），例如同模板下的 INT4 与 FP16 读取、CUDA Graph 与 eager。读结果时先看这些成对比值，再看绝对毫秒数。

CUDA Events 给出设备时间线上的调用耗时，Wall 时间（仅 warm 模式）还包含主机提交和同步。每个样本是多次调用的均值，预热后打乱路径的计时顺序。这些是 eager 或 CUDA Graph 算子实验，不是单请求尾延迟或服务吞吐。所有路径都包含自己的输出与临时张量分配，pack 路径还包含输入检查和格式转换。随机输入创建、跨方案的 dtype 准备、Triton 配置选择、输出正确性对照和 profiler 都在正式计时之外。

`storage_bytes` 是声明路径的 payload＋尺度等张量账本。`bytes_moved` 是该路径至少要读写的字节数，`effective_gbps` 由它除以中位时间得到，`bandwidth_fraction` 再除以本轮实测读带宽。先解码的路径按“读 INT4＋写 FP16＋读 FP16”计数，因此是下界。这些都是按公式算的账本，**不是 profiler 计数**，需要精确的 DRAM 流量时用 Nsight Compute。

进程内同时保留了参照与候选输入，所以 `resident_allocated_bytes` 不能当作该路径单独部署时的占用。`incremental_peak_bytes` 才是单次调用超过当前驻留量的分配峰值，它仍不包含 CUDA 库在 PyTorch allocator 之外分配的内存。TFLOPS 按声明的 FLOP 数除以时间计算，含转换的路径给出的是有效吞吐，不等于硬件峰值。

原始接口参考：[PyTorch CUDA 语义与计时](https://docs.pytorch.org/docs/stable/notes/cuda.html)、[CUDA Graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)、[SDPA 后端选择](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)、[Triton 文档](https://triton-lang.org/main/index.html)、[bitsandbytes INT8 函数](https://huggingface.co/docs/bitsandbytes/reference/functional)。实验会记录实际依赖版本，兼容性与执行结果以本轮验证为准。

[返回实验指南](README.md) · [返回学习首页](../README.md)

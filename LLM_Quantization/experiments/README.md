# 实验指南

所有命令在项目根目录执行。Python 3.10+；CPU 参考环境使用 Python 3.12。各入口默认只运行，不覆盖发布结果。

| 目标 | 入口 | 依赖与范围 |
|---|---|---|
| 手算与基础机制 | `python experiments/quant_lab.py` | NumPy，14 组实验、38 项检查 |
| 单项技术 | `python experiments/technique_lab.py --technique gptq` | NumPy，93 项示例 |
| 正文推导与反例 | `python experiments/deep_dive_lab.py` | NumPy，20 项 |
| LLM.int8 分解与阈值 | `python experiments/llm_int8_lab.py` | NumPy；导图需要 Matplotlib |
| 逐章动手练习 | `python experiments/chapter_demo_lab.py --device cpu` | PyTorch，27 个 CPU/CUDA 实操 |
| T4/L4/A100 算子测量 | `python experiments/cuda_lab.py --profile smoke` | CUDA，10 组实验：带宽、GEMM、INT4、Triton 融合 W4A16、INT8/FP8、Attention、KV、融合 INT4 KV、bitsandbytes INT8、Qwen3 形状 decode 骨架 |
| 成本与容量算例 | `python experiments/outlook_scenarios.py` | NumPy，4 个假设情景 |
| 真实模型量化 | [Qwen3 Notebook](notebooks/colab_qwen3_quantization.ipynb) | 兼容 NVIDIA GPU；按 Notebook 安装依赖 |

这些入口验证的层次不同。小矩阵 QDQ、物理打包、真实低比特内核和模型质量分别验证，不能互相替代。

## Colab SSH CUDA 实验

仓库拉取到服务器后，在项目根目录执行：

```bash
python -m pip install -r requirements-cuda.txt
python -B experiments/cuda_lab.py --profile smoke --write-results --output-dir runs/cuda
```

该入口依次运行显存带宽、GEMM、INT4 打包与解码、Triton 融合 W4A16、原生 INT8/FP8、Attention、KV、融合 INT4 KV decode、可选 bitsandbytes INT8，以及 Qwen3-0.6B 形状的 decode 骨架。用 `--case` 选择一项或多项，`--profile full` 扩大工作量，`--trace` 保存 profiler。涉及字节对比的实验在每次计时调用前清空 L2，报告给出带宽比例和成对对照比值。逐次时间、误差、显存和失败原因保存在独立的运行目录中。具体范围、T4/L4/A100 的差异和结果取回方式见 [CUDA 实验指南](CUDA.md)。

## NumPy 数值实验

```bash
python -m pip install -r requirements.txt
python experiments/quant_lab.py
python experiments/technique_lab.py --list
python experiments/technique_lab.py --technique gptq
python experiments/technique_lab.py --check-docs
python experiments/deep_dive_lab.py
python experiments/llm_int8_lab.py
python experiments/outlook_scenarios.py
```

发布输出见[基础实验](results/实验结果.md)、[逐项技术实验](results/逐项技术实验结果.md)、[趋势算例](results/未来趋势_情景算例.md)。趋势算例还配有[交互式成本计算器](../figures/outlook_cost_calculator.html)。

## 逐章实操

安装 `python -m pip install -r requirements-demos.txt` 后运行。


<a id="chapter-demos"></a>

17 章主线、数学附录、LLM.int8 专题和科研进阶各提供一个完整实操，加上新GPU架构的7个专题实操，共 **27 个带中文逐步注释的 demo**。每章的“本章实操”包含实践目的、完整源码、运行命令、预期观察、动手修改建议和边界；代码既能在正文展开，也能单独打开修改。技术索引页只负责导航，无需另写一个算法实验。

这些 demo 使用 Python 3.10+、NumPy 和 PyTorch，全部支持 CPU/CUDA；无需下载模型。`--device auto` 优先选择可用 GPU，`--device cuda` 强制使用 GPU，不可用时报错，`--device cpu` 使用 CPU。训练示例实际执行小型梯度更新，存储示例实际完成临时文件保存与重载，性能/科研示例使用明确标注的合成数据。它们是章节练习，独立于 93 项机制示例和 20 个推导反例计数，不等于 27 篇论文复现。

```bash
python experiments/chapter_demo_lab.py --list
python experiments/chapter_demo_lab.py --chapter 02 --device cuda
python experiments/chapter_demo_lab.py --chapter 07 --device cuda
python experiments/chapter_demo_lab.py --chapter research --device cuda
python experiments/chapter_demo_lab.py --device cuda
# 本机没有 GPU 时
python experiments/chapter_demo_lab.py --device cpu
```

| 章节 | 实操内容 |
|---|---|
| [00](../docs/00_基础与全景.md#chapter-demo) | 模型参数、分组元数据与 KV 容量 |
| [01](../docs/01_均匀量化与误差.md#chapter-demo) | INT4 编码、字节打包、解码与输出误差 |
| [02](../docs/02_GPTQ与二阶误差补偿.md#chapter-demo) | 逐列补偿、逆矩阵消元与穷举对照 |
| [03](../docs/03_离群值与缩放.md#chapter-demo) | 缩放搜索与独立输入测试 |
| [04](../docs/04_可学习校准与HQQ.md#chapter-demo) | 舍入变量优化与 HQQ offset 更新 |
| [05](../docs/05_旋转与非相干化.md#chapter-demo) | 旋转搜索、Hessian 变换与非线性反例 |
| [06](../docs/06_码本_稀疏与混合精度.md#chapter-demo) | 加权码本拟合、索引打包与成本摊销 |
| [07](../docs/07_QAT_QLoRA与原生低比特.md#chapter-demo) | 手写 QAT/STE、冻结基座 LoRA 训练与合并 |
| [08](../docs/08_KV缓存与Attention量化.md#chapter-demo) | 流式缓存、分段 softmax 与失败回滚 |
| [09](../docs/09_浮点格式与推理系统.md#chapter-demo) | 打包层保存、独立重载及数值验证 |
| [10](../docs/10_实验_评估与选型.md#chapter-demo) | PPL 聚合、配对 bootstrap 与 goodput |
| [11](../docs/11_练习与答案.md#chapter-demo) | 注入 scale 错位并逐阶段排错 |
| [12](../docs/12_新进展_变换与编码.md#chapter-demo) | Trellis 动态规划、穷举和 bit 流解码 |
| [13](../docs/13_新进展_FP4与误差修正.md#chapter-demo) | FP4 尺度搜索与残差误差诊断 |
| [14](../docs/14_新进展_训练与MoE.md#chapter-demo) | 固定路由下的专家位宽分配 |
| [15](../docs/15_新进展_KV与推理系统.md#chapter-demo) | MLA 双路径误差与循环状态小更新 |
| [16](../docs/16_未来趋势与研究路线.md#chapter-demo) | 质量约束下的成本选择与回本量 |
| [数学附录](../docs/附录A_推导所需的数学.md#chapter-demo) | KKT、线性求解与二阶统计核验 |
| [LLM.int8 专题](../docs/专题_LLM_int8_原理与实测.md#chapter-demo) | 向量尺度、阈值与 batch 依赖 |
| [科研进阶](../docs/科研进阶_高质量量化论文路线与清单.md#chapter-demo) | 同格式搜索、隔离测试与不确定性 |

新增架构专题可直接用同一命令选择：

| demo ID | 机制与章节 |
|---|---|
| `gpu_formats` | [GPU能力与MXFP4/6/8](../docs/09_浮点格式与推理系统.md#demo-gpu_formats) |
| `mr_gptq` | [微旋转、FP4固定网格与补偿](../docs/13_新进展_FP4与误差修正.md#demo-mr_gptq) |
| `four_over_six` | [4/6实际尺度候选与选择偏差](../docs/13_新进展_FP4与误差修正.md#demo-four_over_six) |
| `training_transpose` | [1D/2D量化与转置一致性](../docs/14_新进展_训练与MoE.md#demo-training_transpose) |
| `gradient_rounding` | [SR与MS-EDEN尺度条件期望](../docs/14_新进展_训练与MoE.md#demo-gradient_rounding) |
| `oscillation` | [OsciReset与训练状态成本](../docs/14_新进展_训练与MoE.md#demo-oscillation) |
| `scale_layout` | [UE8M0转换、repack与分块乘法](../docs/15_新进展_KV与推理系统.md#demo-scale_layout) |

```bash
python experiments/chapter_demo_lab.py --chapter four_over_six --device cuda
python experiments/chapter_demo_lab.py --chapter training_transpose --device cpu
```

以上仍是PyTorch数值模拟。匹配的Blackwell/TE 2.8可另行运行[原生NVFP4探针](native_nvfp4_probe.py)；其限制和验证层次见[执行后端](../docs/15_新进展_KV与推理系统.md#architecture-backends)。新论文与架构来源见[原始资料索引](../references/原始资料索引.md#gpu-architecture-sources)。

默认运行不保存结果；第 09 章只写入自动清理的临时目录。需要记录个人运行时显式指定：

```bash
python experiments/chapter_demo_lab.py --chapter 08 --device cuda --write-results --output-dir runs/ch08
```

发布的整套运行记录见 [chapter_demo_results.json](results/chapter_demo_results.json)。修改时编辑 `experiments/demos/` 中的源码，再执行 `python experiments/build_technique_docs.py` 将代码及注释同步回章节；`python experiments/check_project.py` 会检查正文与源码一致并执行全部 demo。直接复制正文代码时也在项目根目录运行，便于导入公共数值函数；此时 `QUANT_DEMO_DEVICE=cuda` 可强制选GPU，默认auto。

个人练习修改源码后，可加 `--practice` 直接运行，无需重建正文。该模式仍执行数学断言和设备核验，报告明确标记正文一致性未检查；保存时必须指定个人目录。改变题设时，应重新推导相应预期结果。发布内容继续使用上面的严格生成与检查流程。

```bash
python experiments/chapter_demo_lab.py --chapter 03 --device cuda --practice --write-results --output-dir runs/practice
```

个人结果目录中的 `chapter_demo_results.json` 表示最近一次尝试；每次运行还在 `runs/<run_id>/` 保存独立报告和 `sources/` 源码/正文快照。报告逐章原子更新，状态包括 `running`、`partial`、`completed`、`failed`；失败时保留已完成章节、异常与失败前输出。只有 `completed` 表示所选章节全部通过。下载的源码快照在项目根目录用相同参数运行；练习记录需保留 `--practice`。

### 在 Colab GPU 上运行逐章 demo

<a id="colab-chapter-demos"></a>

打开 [逐章 GPU Notebook](notebooks/colab_chapter_demos.ipynb)，上传到 Colab，选择“运行时 → 更改运行时类型 → GPU”，按单元顺序运行。Notebook **自带本次源码快照**，无需仓库地址或额外上传工程；它会校验内嵌文件哈希、解压到 `/content`，保留 Colab 已有的 CUDA 版 PyTorch，再检查 GPU。Colab 的硬件和额度随服务配置变化，见[官方说明](https://research.google.com/colaboratory/faq.html)。

Notebook 中 `CHAPTER="all"` 可运行全部27个demo，改成 `"07"`、`"math"`、`"int8"` 或 `"research"` 可单独运行；`DEVICE="cuda"` 是默认设置。练习时在左侧文件浏览器编辑解包后的 `experiments/demos/`，保存后设置 `PRACTICE=True`，重跑执行单元即可，不要重新解包。每次执行使用新结果目录；下载ZIP包含结果JSON、执行源码、测试日志和已运行的原生探针结果，失败后也可直接下载。修改了本地发布源码后运行 `python experiments/build_colab_chapter_demos.py` 重建内嵌快照。

已有完整项目时，可直接在 Colab 运行：

```python
# %cd 到实际上传的项目根目录；安装需求不会指定CPU专用torch源。
%pip install -r requirements-demos.txt
!python experiments/chapter_demo_lab.py --device cuda --write-results --output-dir /content/chapter_results
```

GPU版的量化、矩阵乘、训练更新、缓存和统计都用目标设备张量计算，运行结果记录实际设备、GPU型号、PyTorch/CUDA版本以及张量设备核验数。随机fixture在CPU生成后一次搬到设备，文件I/O和打印也在CPU；原生探针在Notebook中默认关闭，需要SM100/SM103与指定TE版本；默认FP64数值demo适合小矩阵数学核验，尤其在T4上**不用于测速或声称比CPU更快**。INT8点积用FP64承载精确整数和后转INT32，FP4用码本模拟，均不要求T4支持原生FP4。

发布的全部27个demo已在PyTorch CPU上执行；CUDA集成测试需在兼容GPU运行，当前没有该套demo的GPU实测记录。原有93个NumPy机制示例保留作为独立参照。真实模型量化仍使用 [Qwen3 Colab Notebook](notebooks/colab_qwen3_quantization.ipynb)，按[模型实验验收](../docs/10_实验_评估与选型.md#model-validation-milestones)推进。

## 真实模型与评测协议

<a id="model-experiments"></a>

想在真实小模型上练习，可以打开 [Colab：Qwen3-0.6B 量化实验](notebooks/colab_qwen3_quantization.ipynb)。上传到 Colab 后选择 GPU，或用 Google 官方 Colab VS Code 扩展连接 Notebook 内核。默认自动比较 8 个配置：FP16、LLM.int8()、NF4、RTN INT4 分组 32/64/128，以及 GPTQ/AWQ INT4 分组 128。GPTQ/AWQ 使用 GPTQModel 7.6.0，从固定模型快照和独立 train 校准数据开始，保存并重新加载打包权重后评测；手写 RTN 是 FP16 伪量化，不计 INT4 加速比。

`smoke` 使用 2048 个输入 token、32 条 × 512 token 校准数据排错；`full` 使用 32768 个输入 token、重叠上下文和 128 条校准数据。结果包含 PPL、相对 FP16 的 logits KL、逐组 RTN 权重误差、显存、速度、准备耗时、CSV 和 PNG。中文回答在 HTML 中并排展示，内容正确性、格式遵循和是否触及输出长度上限分别记录，无法可靠自动判定的内容需人工核对。失败方法写入 `failures.json` 并标记 `partial`，不混入成功结果。大型 GPTQ/AWQ checkpoint 单独保存在 `/content/quant_checkpoints`，不包含在小型报告 ZIP 中。

[2026-10-15 T4 实测](archive/colab_t4_2026-10-15/analysis.md)完成了 **6/8 个配置**；GPTQ 在线程池初始化、AWQ 在权重打包阶段失败。[原执行 Notebook](archive/colab_t4_2026-10-15/notebook_executed.ipynb)已原样归档。当前 Notebook 为 **协议 v4**：固定模型/数据快照、修正 GPTQ 父线程池最小容量、分离 AWQ 打包与推理后端，默认分别测 prefill/decode。修订代码已做 CPU 检查，**修订后的完整 GPU 运行仍待 Colab 验证**。

每次运行保存模型/数据提交号、依赖、执行代码、逐次计时及文件哈希清单。下载后用 `python experiments/inspect_run.py /path/to/report.zip` 核对原始结果包；历史提取报告缺少原始包，不能补造逐次计时。不同协议或不同 prompt 的性能结果不能直接拼成同一对照。后端配置与验证范围见[实验与选型](../docs/10_实验_评估与选型.md#colab-backends)。
## 保存个人结果

<a id="saving-results"></a>

```bash
python experiments/quant_lab.py --write-results --figures --output-dir runs/baseline
python experiments/technique_lab.py --check-docs --write-results --output-dir runs/techniques
python experiments/deep_dive_lab.py --write-results --output-dir runs/derivations
python experiments/outlook_scenarios.py --write-results --output-dir runs/outlook
python experiments/llm_int8_lab.py --write-results --output-dir runs/int8
```

`--output-dir` 必须和 `--write-results` 一起使用。个人输出放在被 Git 忽略的 `runs/` 下；JSON 记录参数、依赖版本与实验源码 SHA-256。`--write-results` 不指定目录时更新仓库发布结果，位置见下表；请在维护发布内容时使用。

| 内容 | 发布位置 |
|---|---|
| 基础、93 项技术、20 项推导、27 个实操、趋势算例 | [results/](results/) |
| LLM.int8 数值结果与图表 | [LLM.int8 图表](../figures/llm_int8_deep_dive/) |
| 历史 T4 执行证据 | [archive/colab_t4_2026-10-15/](archive/colab_t4_2026-10-15/analysis.md) |

源码指纹标识某次执行；修改源码后应实际重跑，不能手改结果中的哈希。刷新基础实验图表需同时加 `--figures`。历史 T4 Notebook 保持原字节，当前源码的 CPU 测试不能成为历史 GPU 结果的替代证据。

完整的生成、检查与发布步骤见[贡献指南](../CONTRIBUTING.md)。

[返回学习首页](../README.md)

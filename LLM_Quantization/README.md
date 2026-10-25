# LLM 量化：原理、图解与代码

从手算量化到研究实验的中文教程。用数学推导、图示、可运行代码和反例，理解每种方法优化什么、误差从哪里来，以及它增加哪些存储与计算成本。

项目包含 **17 章主线、93 项机制示例、27 个逐章实操、20 个推导反例和 250 道练习**，另有数学附录、LLM.int8 专题及科研指南。技术来源最近增补至 **2026-10-20**，各资料保留自己的版本与核查日期。

[开始阅读](docs/00_基础与全景.md) · [技术速查](docs/技术图解与代码索引.md) · [实验指南](experiments/README.md) · [论文导航](references/论文与博客导航.md) · [参与维护](CONTRIBUTING.md)

## 快速运行

需要 Python 3.10+。下载项目后，在项目根目录执行；基础实验使用小矩阵，无需模型或 GPU。

```bash
python -m pip install -r requirements.txt
python experiments/quant_lab.py
python experiments/technique_lab.py --technique gptq
```

<a id="chapter-demos"></a>

带中文逐步注释的实操使用 PyTorch：

```bash
python -m pip install -r requirements-demos.txt
python experiments/chapter_demo_lab.py --chapter 02 --device cpu
# 有可用 CUDA 时
python experiments/chapter_demo_lab.py --chapter 02 --device cuda
```

运行全部章节、修改练习、记录结果及 7 个架构专题见[逐章实操指南](experiments/README.md#chapter-demos)。默认运行不覆盖发布结果；个人实验用 `--write-results --output-dir runs/my-experiment`。

<a id="colab-chapter-demos"></a>

| Notebook | 用途 |
|---|---|
| [逐章实操](experiments/notebooks/colab_chapter_demos.ipynb) | 自带源码快照，在 Colab GPU 上执行 27 个小矩阵实操 |
| [Qwen3-0.6B 量化](experiments/notebooks/colab_qwen3_quantization.ipynb) | 比较 FP16、LLM.int8、NF4、RTN、GPTQ、AWQ 的模型指标 |

通过 SSH 使用 Colab T4/L4/A100 时，拉取仓库后即可运行 CUDA 算子实验，不需要 Notebook，也不下载模型：

```bash
python -m pip install -r requirements-cuda.txt
python -B experiments/cuda_lab.py --profile smoke --write-results --output-dir runs/cuda
```

覆盖显存带宽、GEMM 精度与吞吐、INT4 打包与解码、Triton 融合 W4A16 与 INT4 KV decode 内核、原生 INT8/FP8、Attention 后端，以及 Qwen3-0.6B 形状的 decode 骨架（FP16、W4A16、LLM.int8()、NF4，含 CUDA Graph 对照）。安装、SSH 会话、完整测试与结果解读见 [CUDA 实验指南](experiments/CUDA.md)。目前通过的是 CPU 格式检查、记录测试和 Triton 解释器核对，T4/L4/A100 的性能结果需要实际运行才能得到。

## 阅读顺序

第一次学习可按 00–01 → 02–05 → 06–08 → 09–11 → 12–16 推进。矩阵、梯度或 SVD 不熟悉时，先读[数学附录](docs/附录A_推导所需的数学.md)。


| 章节 | 你要解决的问题 | 建议 |
|---|---|---|
| [00 基础与全景](docs/00_基础与全景.md) | 模型里到底什么占显存？W4A16 是什么？ | 第一次必读 |
| [01 均匀量化](docs/01_均匀量化与误差.md) | scale、zero point、分组、裁剪怎么算？ | 手算例子全部做一遍 |
| [02 GPTQ](docs/02_GPTQ与二阶误差补偿.md) | 为什么舍入一个权重，要修改别的权重？ | 跟着推到补偿公式 |
| [03 离群值与缩放](docs/03_离群值与缩放.md) | AWQ 与 SmoothQuant 为何看似相同又不同？ | 理解同一个等价变换的不同目标 |
| [04 可学习的校准](docs/04_可学习校准与HQQ.md) | 如何优化裁剪、舍入与无数据双轴尺度？ | 接上优化思路 |
| [05 旋转量化](docs/05_旋转与非相干化.md) | 不改变模型输出，能把离群值打散吗？ | 先看四维 Hadamard 例子 |
| [06 码本与混合精度](docs/06_码本_稀疏与混合精度.md) | 2 bit 为什么还可能可用？ | 理解“编码方式”也是自由度 |
| [07 训练与微调](docs/07_QAT_QLoRA与原生低比特.md) | QAT、QLoRA、BitNet 与离散码字微调有何区别？ | 注意哪些参数真正被训练 |
| [08 KV 与 Attention](docs/08_KV缓存与Attention量化.md) | 长上下文为何吃显存？缓存与算子量化有何区别？ | 先推缓存公式与误差传播 |
| [09 格式与硬件](docs/09_浮点格式与推理系统.md) | NF4、FP4、INT4 与查表/反量化执行有何区别？ | 部署前必读 |
| [10 实验与选型](docs/10_实验_评估与选型.md) | 怎样亲手验证，怎样选择方法？ | 按实验编号运行 |
| [11 练习与答案](docs/11_练习与答案.md) | 如何分章自测、组织课程和完成综合项目？ | 先做后看答案，按评分要点复习 |
| [12 新进展：变换与编码](docs/12_新进展_变换与编码.md) | DuQuant、ParoQuant、VPTQ、QTIP、HIGGS 等增加了哪些自由度？ | 补齐重要变换、码本和组级位宽路线 |
| [13 新进展：FP4 与误差修正](docs/13_新进展_FP4与误差修正.md) | FP4 尺度、分布漂移、零位剪枝和低秩补偿如何改进？ | 包含 2026 年 10 月新论文 |
| [14 新进展：训练与 MoE](docs/14_新进展_训练与MoE.md) | QuEST、ParetoQ、OnlineQAT 和专家结构量化有何不同？ | 关注训练预算、路由和共享谱基 |
| [15 新进展：KV 与推理系统](docs/15_新进展_KV与推理系统.md) | KV 预算、MLA、循环状态与低比特服务如何落地？ | 区分量化对象、数据生命周期与实际执行路径 |
| [16 未来趋势与研究路线](docs/16_未来趋势与研究路线.md) | 未来 1–3 年哪些方向值得投入，如何验证？ | 区分证据与预测，结合成本、行为和 KV 预算算例 |
| [技术速查](docs/技术图解与代码索引.md) | 我想直接找某种方法 | 进入包含原理、图示与代码的完整小节 |
| [论文与博客导航](references/论文与博客导航.md) | 每项技术的原论文、博客和官方说明在哪里？ | 按技术直接打开，区分论文、博客、项目与规范 |
| [来源索引](references/原始资料索引.md) | 每个结论应回到哪篇论文核对？ | 按编号检索 |

需要集中练习时，使用[课程考核与答案](docs/11_练习与答案.md#course-assessment-guide)。准备研究时，阅读[科研进阶指南](docs/科研进阶_高质量量化论文路线与清单.md)，其中已有固定量化格式下的校准泛化文献对照与实验设计。

## 实验状态与范围

| 层次 | 当前证据 |
|---|---|
| 数学与代码 | 14 组基础实验、93 项技术、20 个推导、27 个实操及回归测试经过 CPU 验证 |
| 逐章 CUDA 实操 | 实现了目标设备计算与对照测试，完整 GPU 运行仍待验证 |
| 历史真实模型 | [2026-10-15 T4 记录](experiments/archive/colab_t4_2026-10-15/analysis.md)完成 6/8 配置；GPTQ/AWQ 准备失败 |
| 当前真实模型 Notebook | 协议 v4 已修订并经过 CPU 逻辑检查，完整 GPU 校准、保存、重载和评测仍待执行 |

机制示例与实操使用明确标注的教学简化；PyTorch 实操默认 FP64，不能用于声称原生 INT4/FP4 加速。真实模型 Notebook 中的 RTN 是反量化回 FP16 的路径。详细条件与后续验收见[实验与选型](docs/10_实验_评估与选型.md#model-validation-milestones)。

## 项目结构

```text
docs/                       教程、题库、专题与科研指南
experiments/                数值函数、运行入口与文档生成器
  demos/                    27 个逐章与专题实操源码
  notebooks/                两个可运行的 Colab Notebook
  results/                  发布的数值结果与报告
  archive/                  历史 GPU 实验原始证据
figures/                    正文引用的图表与交互式计算器
references/                 来源索引、阅读映射与技术覆盖
  provenance/               来源版本、哈希及检索核查记录
tests/                      数学、文档、Notebook 与结果回归测试
```

## 检查与维护

```bash
python -m pip install -r requirements-dev.txt -c constraints-cpu.txt
python -B experiments/check_project.py --notebook-cpu
```

检查默认不修改发布文件，也不下载模型。文档中的部分代码由源码生成，修改位置和构建顺序见[贡献指南](CONTRIBUTING.md)。

## 全书约定

- 输入 $X\in\mathbb R^{n\times d}$，权重 $W\in\mathbb R^{m\times d}$，输出 $Y=XW^T$。
- $q$ 是低比特编码，$\widehat x$ 是解码后的近似值；$Q(x)$ 在声明的上下文中表示量化—反量化。
- 位宽、分组元数据、未量化层和实际执行类型分别计算；权重误差、任务质量、显存与速度分别评估。
- 来源编号及所读版本见[原始资料索引](references/原始资料索引.md)，教学代码的适用范围见各示例说明。

# 贡献与维护

欢迎改进原理讲解、修正数值实现、增加有明确边界的实验，或报告可复现的问题。阅读和运行入口见[首页](README.md)与[实验指南](experiments/README.md)。

## 在哪里修改

| 内容 | 维护位置 | 同步方式 |
|---|---|---|
| 原理、推导、练习与答案 | `docs/` 对应章节 | 直接编辑手写正文，保留区域标记与显式锚点 |
| 93 项技术示例、代码解释、图示步骤 | `experiments/technique_catalog.py`、`latest_techniques.py` | 生成器回填正文与 SVG |
| 27 个完整实操 | `experiments/demos/` | 生成器将完整源码嵌入对应章节 |
| 实操目的、观察与修改建议 | `experiments/chapter_demo_catalog.py` | 与实操源码一并生成 |
| 公共数值函数 | `quant_lab.py`、`technique_kernels.py`、`demo_torch_kernels.py` | 检查对应 NumPy/PyTorch 路径 |
| 来源与阅读链接 | `references/reading_links.json`、来源索引与 `provenance/` | 生成论文导航和覆盖表 |
| 真实模型评测 | `experiments/notebooks/colab_qwen3_quantization.ipynb` | 直接编辑当前 Notebook；保留协议与状态说明 |
| SSH CUDA 实验 | `experiments/cuda_lab.py`、`cuda_cases.py`、`cuda_runtime.py`、`cuda_kernels.py`、`cuda_triton.py` | CPU 核验格式与记录，用 Triton 解释器核对融合内核；在 T4/L4/A100 跑 smoke 和 GPU 集成测试 |

不要直接修改 `inline-technique`、`chapter-demo` 等生成区域来修复代码，下一次构建会覆盖它们。正文中的 `depth-example` 与 `outlook-example` 则是直接执行的推导与情景源码，应在正文修改。

`experiments/notebooks/colab_chapter_demos.ipynb` 是自带源码的生成产物，用于离线上传 Colab；修改实操或正文后需要重建。`figures/` 中的 PNG 用于正文展示，SVG 用于缩放与导出，保留两种格式。`experiments/archive/` 中的原执行 Notebook 和提取数据用于追溯历史结果，应保持原字节。

## 本地检查

建议在独立虚拟环境安装依赖，CPU 参考环境使用 Python 3.12：

```bash
python -m pip install -r requirements-dev.txt -c constraints-cpu.txt
python -B experiments/build_technique_docs.py
python -B experiments/build_colab_chapter_demos.py
python -B experiments/check_project.py --notebook-cpu
```

仅核对生成产物、无需写文件时：

```bash
python -B experiments/build_technique_docs.py --check
python -B experiments/build_colab_chapter_demos.py --check
```

统一检查包含实际展示的代码与源码一致性、本地链接、题库结构、数学断言、失败回滚、保存重载及 Notebook CPU 逻辑。不要加 `-O`，它会移除数值断言。CPU 检查不下载模型；CUDA 不可用时对应测试会明确跳过。

GitHub Actions 使用相同检查入口，先安装 CPU 版 PyTorch，再安装其余依赖。Colab 保留已有的 CUDA 版 PyTorch，使用 Notebook 安装单元或 `requirements-demos.txt`。

CUDA 算子计时流程见[专用指南](experiments/CUDA.md)。`test_cuda_lab.py` 在 CPU 上核验独立 NumPy 参照、打包布局、缓存轴、L2 清理计时、状态汇总和失败记录。安装了 Triton 时，它还会在子进程中以 `TRITON_INTERPRET=1` 逐值核对融合内核，并用缩小形状把新增实验完整走一遍。有 CUDA 时额外运行九组 smoke 集成测试。可选 bitsandbytes 的实际 INT8 验证通过 `cuda_lab.py --case int8 decode_stack` 执行。新增 GPU 实验的结果保存在个人的 `runs/` 目录下，不把 CPU 检查写成 GPU 成绩。

## 更新发布结果

个人实验使用 `--write-results --output-dir runs/<name>`。只有维护仓库发布记录时，才使用以下默认输出命令：

```bash
python -B experiments/quant_lab.py --write-results --figures
python -B experiments/technique_lab.py --check-docs --write-results
python -B experiments/deep_dive_lab.py --write-results
python -B experiments/outlook_scenarios.py --write-results
python -B experiments/chapter_demo_lab.py --device cpu --write-results
python -B experiments/llm_int8_lab.py --write-results
python -B experiments/check_project.py --notebook-cpu
```

数值报告集中在 `experiments/results/`；LLM.int8 的结果与配套图表在 `figures/llm_int8_deep_dive/`。源码或路径改变后，以真实执行更新结果及哈希，不能手改哈希模拟重跑。历史 GPU 数据保留原协议和验证范围，不因当前 CPU 检查通过而升级为新版本验证。

## 提交问题与改动

- 公式或文献问题：提供章节、小节、所读论文版本和具体差异；资料规则见[来源索引](references/原始资料索引.md#search-method)。
- 运行问题：提供最小命令、Python/PyTorch 版本、设备、报错与预期行为。发布日志前移除访问令牌和个人路径。
- 代码改动：说明具体问题、改变后的行为及验证结果；优先修改对应章节与既有实验，不另建重复的补充或勘误页面。
- 新技术：先明确优化对象、表示成本、适用条件和省略部分，再给可核验的算例。方法数量不是完整论文复现数量。

模型权重、数据集缓存、个人输出和环境排错快照不放入仓库。真实模型报告需保留成功与失败状态、原始计时和执行来源；新协议的结果另建运行记录。

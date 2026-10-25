# 投机解码：原理、实践与研究

中文投机解码学习与调研资料，配套可离线运行的概率、Transformer 和训练原理实验。

**[阅读完整文档（Markdown）](docs/投机解码深度调研与实践.md)**

全部内容由一份主文维护，共 22 章、32 张原创图和 47 段可运行示例。理论推导、图表、反例及代码解析随对应知识点展开：

| 阅读阶段 | 重点 |
|---|---|
| [基础与正确性：第 1–4 章](docs/投机解码深度调研与实践.md#s1) | 自回归瓶颈、最大耦合、KV 状态、真实提案概率与误差界 |
| [候选与验证：第 5–9 章](docs/投机解码深度调研与实践.md#s5) | n-gram、EAGLE、MTP、并行／Markov 草稿、树与块验证 |
| [训练与成本：第 10–12 章](docs/投机解码深度调研与实践.md#s10) | KL/TV、D-PACE、跨轮目标、长上下文、因果调度与预算 |
| [接入与实验：第 13–16 章](docs/投机解码深度调研与实践.md#s13) | 框架条件、论文性能、完整复现、流式与配对统计 |
| [研究与证据：第 17–22 章](docs/投机解码深度调研与实践.md#s17) | 研究假设、来源索引、前沿发展与预测 |

原调研截止 2026-10-20，文档更新至 **2026-10-25**。框架支持按各来源所阅版本说明，具体访问日期见来源索引和正文；教学实验不代表生产 GPU 性能。

## 快速开始

建议在 Python 3.10 或更新版本的虚拟环境中运行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m examples.check_tutorial_snippets
```

Windows 可用 `.venv\Scripts\activate` 激活环境。基础实验不需要 GPU、checkpoint 或 API key。测试覆盖概率、缓存、训练梯度与统计口径；代码块检查执行主文中的全部标记示例。

```bash
python -m examples.speculative_sampling
python -m examples.tiny_transformer
python -m examples.ngram_speculation
python -m examples.principle_lab
python -m examples.exactness_lab
python -m examples.training_objectives_lab
python -m examples.systems_lab
python -m examples.cost_aware_lab
```

真实模型演示 [hf_assisted.py](examples/hf_assisted.py) 是可选入口，需要另行准备 Transformers 和兼容模型，不属于默认离线实验。

## 目录

| 路径 | 内容 |
|---|---|
| [docs/](docs/) | Markdown 主文档、SVG 图、来源索引和预测台账 |
| [examples/](examples/) | 原理实验、图像生成、HTML 构建与验收代码 |
| [tests/](tests/) | 概率、KV、lookup、训练梯度和统计合同测试 |
| [data/](data/) | 原创离线实验数值及可复现核算结果 |

## 更新实验与重建文档

```bash
python -m examples.run_experiments
python -m examples.principle_lab
python -m examples.check_block_derivations
python -m examples.check_probability_claims
python -m examples.exactness_lab
python -m examples.training_objectives_lab
python -m examples.systems_lab
python -m examples.cost_aware_lab
python -m examples.make_figures
python -m examples.validate_artifacts
```

构图只读取保存的实验结果；默认生成 SVG。需要 PNG 时运行 `python -m examples.make_figures --png`（技术地图仅提供 SVG）。临时执行报告写入 `build/`，不随源码提交。

## 可选 HTML 导出

```bash
python -m pip install -r requirements-html.txt
python -m examples.build_html
python -m examples.validate_artifacts --require-html
```

输出 `docs/投机解码深度调研与实践.html`，内嵌全部公式、SVG 和理论计算器，可离线阅读或打印。HTML 已加入忽略规则，Markdown 是唯一正文维护入口；更新正文或图片后需重新导出。

详细说明见文内[完整复现指南](docs/投机解码深度调研与实践.md#s15)、[来源与资料边界](docs/投机解码深度调研与实践.md#s19)。理论曲线与随机权重实验不代表真实 GPU benchmark。

## 许可证

本次先整理发布文件，许可证待项目所有者确定。确定前请勿将“公开可读”视为已经获得自由转载、修改或再分发的授权。第三方论文和项目通过链接引用，其权利与许可仍由原权利人决定。

# 可复现的教学实验数据

本目录只包含项目生成的小词表或随机权重实验结果，不含用户数据、模型 checkpoint 或第三方数据集。

| 文件 | 生成命令 |
|---|---|
| [experiment_results.json](experiment_results.json) | `python -m examples.run_experiments` |
| [ngram_results.json](ngram_results.json) | `python -m examples.ngram_speculation` |
| [principle_results.json](principle_results.json) | `python -m examples.principle_lab` |
| [block_derivation_checks.json](block_derivation_checks.json) | `python -m examples.check_block_derivations` |
| [claim_checks.json](claim_checks.json) | `python -m examples.check_probability_claims` |
| [exactness_results.json](exactness_results.json) | `python -m examples.exactness_lab` |
| [training_objectives_results.json](training_objectives_results.json) | `python -m examples.training_objectives_lab` |
| [systems_results.json](systems_results.json) | `python -m examples.systems_lab` |
| [cost_aware_results.json](cost_aware_results.json) | `python -m examples.cost_aware_lab` |

神经模型使用随机初始化，验证条件 logits、KV 和输出一致性，不代表文本能力或性能基准。调度成本和速度曲线使用明确的合成假设。按需重跑；临时日志和机器相关报告放入被忽略的 `build/` 目录。

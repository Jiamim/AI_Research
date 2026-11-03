"""Generate the source-mapping chapter (35) and code appendix (36) from the checked manifest."""
from pathlib import Path
import json
import platform
import torch

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "book/chapters"
MAP = {
    "2024": {1:"00–02",2:"01、05、08",3:"03–05",4:"06",5:"08–10",6:"09–10、21",7:"11",8:"11",9:"13",10:"05、13",11:"14",12:"14",13:"14",14:"04、14、16、18",15:"16",16:"17",17:"15"},
    "2025": {1:"00–02",2:"05、08",3:"03–05",4:"06",5:"08–10",6:"09–10",7:"11",8:"11",9:"13",10:"04、12",11:"05、13",12:"15",13:"04、14、16",14:"14",15:"16–17",16:"18",17:"17–18、20"},
    "2026": {1:"00–02",2:"05、08–09",3:"03–05",4:"06–07",5:"08–10",6:"09–10",7:"11",8:"11",9:"13",10:"04、12",11:"05、13",12:"15",13:"14",14:"14、16",15:"16–17",16:"18",17:"19"},
}


def main():
    m=json.loads((ROOT/"sources/manifest.json").read_text(encoding="utf-8"))
    docs={d["id"]:d for d in m["documents"]}
    validation_path=ROOT/'book/validation.json'
    validation=json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else None
    out=["# 35｜逐年覆盖表、资料索引与核验记录\n",
         "本书以 2026-10-26 可访问的官方材料为依据。三年课程表、51 份核心讲义、15 份主作业和一份沿用两年的安全补充作业均已获取；manifest 共 71 条记录，其中安全补充保留两个页面来源。讲义按 Python 文本及 PDF 抽取文本阅读核对，未声称逐个观看视频，也没有把所有课件图片作为已读文本处理。\n",
         "## 35.1 年份变化与合并决策\n",
         "| 主题 | 2024 | 2025 | 2026 | 本书处理 |\n|---|---|---|---|---|\n",
         "| 系统内核 | RMSNorm 自定义内核 | FlashAttention-2 | FlashAttention-2、checkpoint/FSDP 扩展 | 保留 RMSNorm 推导，展开在线注意力 |\n",
         "| 数据 | 四次独立数据课 | 两次数据课并吸收其他主题 | 数据来源、过滤、混合、合成 | 保留 DSIR、Bloom、后缀数组 |\n",
         "| 推理服务 | 分散于架构/系统 | 独立第十讲 | 保留并更新 | 独立第 12 章 |\n",
         "| 后训练 | SFT、偏好、RLHF/DPO | 数学 SFT、EI、GRPO | 归一化、RFT、MaxRL、GSPO | 分开解释偏好与可验证奖励 |\n",
         "| 其他架构 | MoE | MoE | 注意力替代与 MoE | 单列第 07 章 |\n",
         "| 第十七讲 | 评估 | RL 代码讲解 | 多模态 | 三个主题分别保留 |\n",
         "\n表中的年度差异是根据课程表、讲义与作业内容作的比较，不代表某主题此前完全没有被提及。\n",
         "## 35.2 逐讲覆盖\n"]
    for year in ["2024","2025","2026"]:
        out += [f"### {year}\n", "| 讲次 | 官网主题 | 本书章节 | 已核验来源 |\n|---|---|---|---|\n"]
        for row in m["schedule"]:
            if row["year"]!=year:continue
            n=row["lecture"]
            d=docs.get(f"L{year}-{n:02d}")
            if d:
                source=f"[讲义]({d['pinned_url']})"
                ch=MAP[year][n]
            else:
                source=f"[课程表]({docs['S'+year]['url']})；未列独立讲义"
                ch="资料边界，未推断嘉宾口述"
            desc=row['description'].replace('|','/')
            out.append(f"| {n} | {desc} | {ch} | {source} |\n")
    out += ["\n2024 年评估讲义注明嘉宾课没有录像；2025/2026 官网未为嘉宾行提供独立讲义链接。本书不能保证涵盖这些课独有的口述、互动或最新观点，读者不应把这部分当成已经完整核验。\n",
            "## 35.3 五组作业的知识覆盖\n",
            "| 作业 | 学习重点 | 本书章节 | 本书实际可运行内容 |\n|---|---|---|---|\n",
            "| A1 | tokenizer、模型、优化、训练与生成 | 01–05、08、20、28 | 按作业约定的 byte BPE（预切分、特殊 token、增量合并、流式编码）、带消融开关的解码器、AdamW/日程/裁剪、TinyStories CPU 实验 |\n",
            "| A2 | profiler、内核、通信、分片 | 08–11、28 | 在线 softmax 与 softmax 反向恒等式、Triton 行 softmax kernel（CPU 解释器可运行）、DDP/collective/优化器状态分片双进程检查、低精度累加（作业评分的 FlashAttention 实现不提供） |\n",
            "| A3 | IsoFLOP 与预算外推 | 13、28 | 已知真值曲面上的预算受限实验设计、最低点与抛物线两种估计、多种子误差分析 |\n",
            "| A4 | 抽取、过滤、去重、消融 | 14、28 | Gopher 式规则、联系方式替换、行级精确去重、MinHash+LSH 去重的参考实现（接口与作业不同） |\n",
            "| A5 | SFT、偏好、RLVR 及变体 | 16–18、28 | SFT/蒸馏/DPO/SimPO/GRPO/GSPO 目标、在小语言模型上的完整 GRPO 循环与变体比较、离散策略的 baseline 方差枚举（例子不同于作业） |\n",
            "\n这些实现用于建立知识与验证机制；官方作业的全量测试适配、竞赛结果、GPU 内核、互联网规模清洗和大模型训练运行不在已完成实验声明内。\n",
            "| 年份/作业 | PDF 页数 | 固定历史版本 |\n|---|---:|---|\n"]
    for d in m['documents']:
        if d['kind']=='assignment':out.append(f"| {d['id']} | {d['pages']} | [官方 PDF]({d['pinned_url']}) |\n")
    out += ["\n2025 年旧文件在当前 main 上失效，已从官方 git 历史恢复。选取的是 2026-01-01 之前最后一次提交，并在 `sources/manifest.json` 记录固定 commit、下载 URL、SHA-256 与恢复说明。2026 安全补充仍指向 2025 资料，不视作一份独立的新内容。\n",
            "## 35.4 关键原始论文\n",
            "论文标题与摘要已核验；正文主要解释依据课程讲义和独立推导，表中链接供进一步追溯，不表示每篇论文均逐页精读。\n",
            "| 论文 | 原始来源 |\n|---|---|\n"]
    for p in json.loads((ROOT/'sources/papers.json').read_text(encoding="utf-8")):
        if p.get('title'):out.append(f"| {p['title'].removeprefix('Title: ')} | [arXiv {p['id']}]({p['url']}) |\n")
    out += ["\n## 35.5 实际验证记录\n",
            f"本次检查环境：Python {validation['python'] if validation else platform.python_version()}，PyTorch {validation['torch'] if validation else torch.__version__}，CPU，未使用 CUDA。\n",
            (f"- 最近一次机器验证记录：{validation['tests_run']} 项测试，失败 {validation['failures']}，错误 {validation['errors']}；整体成功：{validation['success']}。见 `book/validation.json`。覆盖作业约定（BPE 示例、调度、初始化、核算数字）、数值边界（精度、padding、目标归一化、KL 梯度）、恢复、注意力、分布式与统计工具。\n" if validation else "- 未找到机器验证记录；运行 `python scripts/run_checks.py` 后再生成此附录。\n"),
            ("- 两个 CPU/Gloo 进程：等长与不等长本地 batch 的平均梯度与单进程全局目标一致；reduce-scatter+all-gather 等于 all-reduce；优化器状态分片的 AdamW 与不分片一致。\n"
             if validation and validation.get('distributed', {}).get('success') else
             "- 当前验证记录中的双进程检查未通过或未运行；原因见 `book/validation.json`，不能把单元测试通过当作分布式检查通过。\n"),
            "- TinyStories CPU 实验（第 20.4 节）：200 MiB 训练文本、10K BPE、8.3M 参数模型、每个完整运行 6.1M 训练 token；6 个学习率、7 个消融配置，另有 3 个基线重复种子。结果见 `book/tinystories-results.json`。\n",
            "- 合成算术小模型：120 步时 100 道题答对 10%，训练到 1000 步答对 100%；在其上运行的 GRPO 循环见第 18.16 节与 `book/grpo-results.json`。\n",
            "- 缩放实验为明确标注的合成曲面，用于比较实验设计，不是真实模型的缩放律。\n",
            "- 前沿章节（第 29–34 章）：14 个开源模型的参数公式与官方 safetensors 元数据逐个精确相等，YaRN 频率与 transformers 参考实现逐元素对照，gpt-oss-120b 的 checkpoint 大小由元数据复算为 60.8 GiB；合成数据蒸馏实验在 CPU 上用 5 个种子运行。结果见 `book/frontier-results.json`，由 `scripts/run_frontier_experiments.py` 重算。\n",
            "- CUDA/Triton 性能、多机扩展、真实大模型 RLVR 和多模态训练未在当前 CPU 环境中运行；Triton kernel 只在解释器模式下验证了正确性。\n",
            "\n验证命令与运行范围见仓库根目录 `README.md`。\n"]
    out.extend(["\n## 35.6 进阶章节的依据与边界\n",
                "第 22–27 章是为独立研究与应用增加的扩展，不归为某年课程的内容：研究设计、数据适配、RAG、推理时计算、压缩、机制可解释性。原始论文的标题摘要和重点相关全文片段已核对，访问与哈希记录见 `sources/advanced-sources.json`。教学公式和代码独立编写，不把简化更新视为完整 DoReMi/GPTQ 等方法复现。\n",
                "| 进阶资料 | 原始论文 |\n|---|---|\n"])
    advanced_path=ROOT/'sources/advanced-sources.json'
    if advanced_path.exists():
        for source in json.loads(advanced_path.read_text(encoding="utf-8")):
            if source.get('title'):
                out.append(f"| {source['title'].removeprefix('Title: ')} | [arXiv {source['id']}]({source['url']}) |\n")
    out.extend(["\n## 35.7 前沿章节的依据与边界\n",
                "第 29–34 章整理 2024–2025 年社区公开的模型、训练方法和数据配方，同样不归为某年课程的内容。模型配置按固定版本保存在 `sources/model-configs/`，下表的总参数取自 Hugging Face safetensors 元数据，与 `examples/architectures.py` 的公式逐个精确相等。各技术报告中引用的数字均对照原文核对过；它们是各团队在各自设定下的结果，本书没有复现这些大规模训练。\n",
                "| 模型 | 固定版本 | safetensors 总参数 |\n|---|---|---:|\n"])
    index_path=ROOT/'sources/model-configs/index.json'
    if index_path.exists():
        index=json.loads(index_path.read_text(encoding="utf-8"))
        for repo, info in index['models'].items():
            out.append(f"| [{repo}]({info['source']}) | `{info['revision'][:12]}` | {info['safetensors_total']:,} |\n")
        out.append(f"\n配置获取日期：{index['retrieved']}。\n")
    out.append("\n| 前沿资料 | 原始论文 | 章节 | 核对范围 |\n|---|---|---|---|\n")
    frontier_path=ROOT/'sources/frontier-sources.json'
    if frontier_path.exists():
        for source in json.loads(frontier_path.read_text(encoding="utf-8")):
            scope='摘要与正文相关段落' if source['status']=='fulltext_checked' else '摘要'
            out.append(f"| {source['title']} | [arXiv {source['id']}]({source['url']}) | {source['chapters']} | {scope} |\n")
    update_path = ROOT/'sources/review-update-sources.json'
    if update_path.exists():
        out.extend(["\n## 35.8 数值工具与模型进展的原始来源\n",
                    "新增作业诊断与验收、第 22–34 章综合自测、DPO 共享参数反例、RL 训练/推理一致性、NVFP4、BLT/H-Net、Muon 更新尺度、QAT/2:4、SAE/steering 与学习率重新升温。章节保持 37 章；相关来源和状态如下，教学数值见 `book/review-update-results.json`。工业训练结果仅引用论文，不声明复现。\n",
                    "| 原始来源 | 核对状态 |\n|---|---|\n"])
        for source in json.loads(update_path.read_text(encoding="utf-8")):
            out.append(f"| [{source['id']}]({source['url']}) | {source['status']} |\n")
    chapter_sources = ROOT/'sources/chapter-review-sources.json'
    if chapter_sources.exists():
        out.extend(["\n## 35.9 理论边界与实现契约的依据\n",
                    "第 00–34 章正文中的理论边界与实现契约由已获取资料和小规模数值实验支撑。`book/chapter-review-results.json` 保存 NoPE、PPO 等受控反例，`chapter_invariants` 字段记录分块 softmax 合并、Delta rule、CLIP 跨批负例、非均匀 MoE 路由、优化器零梯度语义、预条件梯度与 YaRN 频率计数，可用 `python scripts/run_chapter_review.py` 重算。独立关系检查见 `tests/test_chapter_invariants.py`，手算题见第 21.10 节；这些结果不代表完整模型训练或 GPU 性能验证。核验范围未包括无公开讲义的口述，也未重跑工业规模训练。\n",
                    "`interfaces` 字段补充 UTF-8 流式解码、独立文档 packing、线性注意力低精度、按簇配对 bootstrap、终止/截断 GAE、校准与密度比反例，以及直接由函数计算的 RoPE 频率曲线；相应测试见 `tests/test_learning_interfaces.py`。这些是接口与数学关系的教学核验，原有训练实验仍保留各自的日期与运行条件。\n",
                    "\n| 原始来源 | 核验范围 |\n|---|---|\n"])
        for source in json.loads(chapter_sources.read_text(encoding="utf-8"))['sources']:
            out.append(f"| [{source['name']}]({source['url']}) | {source['scope']} |\n")
    research_path=ROOT/'sources/research-sources.json'
    if research_path.exists():
        research=json.loads(research_path.read_text(encoding="utf-8"))
        out.extend(["\n## 35.10 深入机制的原始资料\n",
                    "以下资料用于补充词表损失、MLA/扫描、LoRA、后训练、检索、压缩与低精度。核验日期 2026-10-31；读取范围按表列出，不等于每篇逐页精读或复现原论文。数学教学实现见 `examples/mechanisms.py`，数值结果见 `book/mechanism-results.json`。\n",
                    "\n| 原始资料 | 本书章节 | 核对内容 |\n|---|---|---|\n"])
        for row in research['sources']:
            out.append(f"| [{row['title']}]({row['url']}) | {row['chapters']} | {row['scope']} |\n")
    (CHAPTERS/'35-sources.md').write_text(''.join(out), encoding="utf-8")
    out=["# 36｜完整教学代码\n",
         "以下源码从仓库实际可运行文件自动嵌入，便于在单文件 HTML、PDF 中离线阅读。执行仍建议使用仓库模块命令；带相对 import 的代码需要保留 `examples` 包结构。\n",
         "\n| 模块 | 内容 | 主要章节 |\n|---|---|---|\n",
         "| tokenizer | byte BPE（预切分、特殊 token、增量合并、流式编码/UTF-8 解码）、Unigram 切分 | 02 |\n",
         "| model | RMSNorm、RoPE、GQA、SwiGLU、KV cache、消融开关、top-p 解码 | 03、04 |\n",
         "| optim | AdamW、余弦与 WSD 日程、全局范数裁剪 | 05 |\n",
         "| train | token 文件训练、固定验证窗口、发散检测、精确恢复 | 05、20 |\n",
         "| accounting | 参数、FLOPs、显存、MFU、roofline、decode 延迟 | 08、12 |\n",
         "| attention | 在线 softmax、softmax 反向恒等式、Triton 行 softmax kernel | 10 |\n",
         "| systems | 自定义 RMSNorm 反向、张量并行反向、低精度累加 | 09、11 |\n",
         "| distributed | 双进程 DDP、collective 等价、优化器状态分片 | 11 |\n",
         "| inference | speculative decoding、分页 KV 与前缀共享 | 12 |\n",
         "| data | Gopher 式规则、联系方式替换、行级与 MinHash 去重、mmap 读取 | 05、14 |\n",
         "| scaling | 预算受限的 IsoFLOP 设计与多种子误差 | 13 |\n",
         "| objectives | SFT 损失、蒸馏、packing 注意力与目标边界 mask | 04、16 |\n",
         "| posttrain | DPO、SimPO、GRPO 族、GSPO、GAE 终止/截断、熵、小模型 GRPO 循环 | 17、18 |\n",
         "| evaluation | pass@k、Wilson、逐题/逐簇配对 bootstrap、Elo、算术诊断 | 15、20 |\n",
         "| extensions | MoE 路由、线性注意力、CLIP 损失 | 06、07、19 |\n",
         "| research / retrieval / reasoning / compression / interpretability | 第 22–27 章的小实验 | 22–27 |\n",
         "| architectures | 14 个开源模型的参数公式（GQA、MLA、MoE、MTP、Gated DeltaNet、Lightning Attention）、激活参数、KV 每 token | 29 |\n",
         "| long_context | RoPE 波长、PI、NTK、YaRN、可学习 sink 注意力、流式缓存 | 30 |\n",
         "| rl_objectives | DAPO 解耦裁剪与 token 级聚合、CISPO、动态采样、超长惩罚 | 31 |\n",
         "| agents | 工具调用回合、只在模型段计损失的掩码、语法树计算器、可验证奖励 | 32 |\n",
         "| synthetic_data | 教师采样、拒绝采样过滤、软标签蒸馏的对照实验 | 33 |\n",
         "| low_precision | MXFP4（MX Algorithm 1）、FP8 1×128 与 128×128 分块量化 | 34 |\n",
         "| numerical_tools | DPO/KL、预测区间、QAT/2:4、SAE/Muon、softmax/Delta rule、packing，以及校准/采样/接口反例 | 01–07、10、14–19、23、26–28、30–31、34 |\n",
         "| mechanisms | 词表分块/并行交叉熵、仿射扫描、MLA 吸收、ESS、验证器基率、潜势塑形与随机舍入 | 07–12、18、25、32、34 |\n",
         "\n实验脚本（不嵌入全文，见仓库 `scripts/`）：`run_core_experiments.py`、`run_tinystories.py`、`run_grpo_demo.py`、`run_advanced_experiments.py`、`run_frontier_experiments.py`、`run_review_update_experiments.py`、`run_chapter_review.py`、`run_mechanism_experiments.py`、`run_checks.py`。\n"]
    for name in ['tokenizer','model','optim','train','accounting','attention','systems','distributed','inference','data','scaling','objectives','posttrain','evaluation','extensions','research','retrieval','reasoning','compression','interpretability','architectures','long_context','rl_objectives','agents','synthetic_data','low_precision','numerical_tools','mechanisms']:
        path=ROOT/'examples'/f'{name}.py'
        out += [f"## {name}.py\n",f"[对应文件](../../examples/{name}.py)\n", "```python\n",path.read_text(encoding="utf-8"),"\n```\n"]
    (CHAPTERS/'36-code.md').write_text(''.join(out), encoding="utf-8")


if __name__=='__main__':main()

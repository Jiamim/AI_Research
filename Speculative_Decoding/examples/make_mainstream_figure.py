"""Generate an original SVG technical map; no measured performance is implied."""
from html import escape
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "docs/figures/09_mainstream_map.svg"


def main():
    parts = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="1010" viewBox="0 0 1120 1010" role="img" aria-labelledby="title desc">
<title id="title">投机解码技术地图</title>
<desc id="desc">八类候选生成路线共享目标验证与资源调度；方法间的连接不代表性能排名。</desc>
<defs><style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#18354d}.title{font-size:29px;font-weight:700}.subtitle{font-size:16px;fill:#5a7187}.heading{font-size:20px;font-weight:700}.body{font-size:17px}.small{font-size:15px;fill:#526b81}</style></defs>
<rect width="1120" height="1010" rx="18" fill="#f3f7fc"/>
<text x="40" y="53" class="title">投机解码：候选来源 × 验证 × 调度</text>
<text x="40" y="84" class="subtitle">路线可以组合；本图表示机制关系，不表示速度排名或稳定版支持范围。</text>''']
    cards = [
        ("01  独立小模型", "Draft LM · DistillSpec · 动态 lookahead", "读取 token 前缀，串行提出候选", "成本：额外权重、KV 与草稿调用"),
        ("02  历史与检索", "Prompt lookup · n-gram · suffix · REST", "从已出现过的文本中寻找续写", "成本：索引、检索、缓存及验证"),
        ("03  多头与递归头", "Medusa · Hydra · ReDrafter", "复用目标状态，增加轻量预测模块", "重点：候选依赖、树预算、训练头"),
        ("04  目标特征草稿", "EAGLE → EAGLE-2 → EAGLE-3", "特征信息 + 轻量自回归 + 链或树", "重点：训练／推理状态对齐"),
        ("05  权重复用与 MTP", "LayerSkip · 原生 MTP · FastMTP · UNO", "早退、原生预测模块或草稿适配器", "重点：训练资产与状态复用接口"),
        ("06  并行块草稿", "PARD · P-EAGLE · DFlash", "一次主干前向预测多个位置", "重点：合法上下文与块内一致性"),
        ("07  候选依赖修正", "Domino · DSpark · DFlash 2 · LiLiCorr", "轻量顺序头或候选格路径选择", "重点：实际提案概率与头部成本"),
        ("08  目标并行前瞻", "Lookahead Decoding", "目标计算候选 n-gram，再验证", "重点：额外 FLOPs 与资源收益"),
    ]
    for i, (heading, names, mechanism, concern) in enumerate(cards):
        x, y = 40 + (i % 2) * 535, 114 + (i // 2) * 151
        parts.append(f'<rect x="{x}" y="{y}" width="505" height="134" rx="10" fill="white" stroke="#cfddea"/>')
        parts.append(f'<rect x="{x}" y="{y+14}" width="5" height="105" rx="2" fill="{"#2457a7" if i<6 else "#167d78"}"/>')
        for dy, cls, value in [(31,"heading",heading),(60,"body",names),(88,"small",mechanism),(113,"small",concern)]:
            parts.append(f'<text x="{x+20}" y="{y+dy}" class="{cls}">{escape(value)}</text>')
    parts.append('''<path d="M560 718v22m-8-8 8 8 8-8" fill="none" stroke="#5a7187" stroke-width="3"/>
<rect x="40" y="752" width="1040" height="96" rx="10" fill="#e5f3ef" stroke="#a1cdc1"/>
<text x="62" y="785" class="heading">共享验证层：目标 causal／tree forward → 接受、修正、提交、回滚</text>
<text x="62" y="820" class="body">贪心一致 / 随机精确采样 / 专门块验证；概率必须对应真实提案过程</text>
<rect x="40" y="868" width="1040" height="96" rx="10" fill="#e8eefb" stroke="#b7c9e8"/>
<text x="62" y="901" class="heading">资源控制层：草稿深度、树节点、batch、KV、CUDA graph 与动态验证预算</text>
<text x="62" y="936" class="body">目标：实际延迟与有效吞吐；接受更长并不自动更快，动态选择也须保持正确性</text>
<text x="40" y="991" class="small">原创机制示意 · 图形维护 2026-10-20 · 具体支持条件见正文与来源快照</text>
</svg>''')
    DEST.parent.mkdir(exist_ok=True)
    svg = "\n".join(parts)
    # Presentation attributes also render in offline SVG/PDF viewers without CSS.
    styles = {
        "title": 'font-size="29" font-weight="700" fill="#18354d"',
        "subtitle": 'font-size="16" fill="#5a7187"',
        "heading": 'font-size="20" font-weight="700" fill="#18354d"',
        "body": 'font-size="17" fill="#18354d"',
        "small": 'font-size="15" fill="#526b81"',
    }
    for name, attributes in styles.items():
        svg = svg.replace(f'class="{name}"', f'class="{name}" {attributes} font-family="sans-serif"')
    DEST.write_text(svg, encoding="utf-8")
    print(f"Built {DEST.name}")


if __name__ == "__main__":
    main()

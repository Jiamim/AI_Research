"""Rebuild original SVG diagrams and plots; no upstream slides are reproduced."""
from pathlib import Path
import html
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Deterministic SVG output: fixed id salt and no timestamp, so a rebuild only changes files whose
# content changed. Each figure also gets its own id prefix (see _prefix_ids), because every SVG is
# inlined into one HTML page and ids such as "DejaVuSans-30" or "arrow" would otherwise collide.
matplotlib.rcParams["svg.hashsalt"] = "cs336-guide"
_savefig = matplotlib.figure.Figure.savefig


def _deterministic_savefig(self, fname, *args, **kwargs):
    kwargs.setdefault("metadata", {"Date": None})
    _savefig(self, fname, *args, **kwargs)
    if str(fname).endswith(".svg"):
        _prefix_ids(Path(fname))


matplotlib.figure.Figure.savefig = _deterministic_savefig


def _prefix_ids(path):
    """Namespace every id (and its references) with the file name so inlined SVGs never collide."""
    import re
    prefix = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()) + "-"
    text = path.read_text(encoding="utf-8")
    ids = set(re.findall(r'\bid="([^"]+)"', text))
    if not ids:
        return
    pattern = "|".join(re.escape(i) for i in sorted(ids, key=len, reverse=True))
    text = re.sub(r'\bid="(' + pattern + r')"', lambda m: f'id="{prefix}{m.group(1)}"', text)
    text = re.sub(r'(url\(#|href="#)(' + pattern + r')(?=[)"])', lambda m: m.group(1) + prefix + m.group(2), text)
    path.write_text(text, encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "book/assets"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = ["#e6f1fb", "#e3f4ec", "#fff1d6", "#f3e8fb"]


def diagram(name, title, rows, subtitle="", connect=True, connect_rows=True):
    width, box_width, box_height = 1080, 232, 83
    height = 115 + len(rows)*133
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
             '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker></defs>',
             '<rect width="100%" height="100%" rx="16" fill="#f8fafc"/>',
             '<g font-family="Noto Sans CJK SC,Noto Sans SC,Microsoft YaHei,sans-serif" fill="#18334a">',
             f'<text x="34" y="40" font-size="23" font-weight="700">{html.escape(title)}</text>',
             f'<text x="34" y="70" font-size="15" fill="#52687c">{html.escape(subtitle)}</text>']
    for r,row in enumerate(rows):
        y=98+r*133
        for col,lines in enumerate(row):
            x=30+col*270
            parts.append(f'<rect x="{x}" y="{y}" width="{box_width}" height="{box_height}" rx="10" fill="{COLORS[(r+col)%4]}" stroke="#bacbda"/>')
            for j,line in enumerate(lines):
                parts.append(f'<text x="{x+14}" y="{y+28+j*23}" font-size="{16 if j==0 else 14}" font-weight="{600 if j==0 else 400}">{html.escape(line)}</text>')
            if connect and col < len(row)-1:
                parts.append(f'<path d="M{x+box_width+4},{y+42} H{x+266}" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>')
        if connect_rows and r<len(rows)-1:
            last_x=30+(len(row)-1)*270+box_width/2
            parts.append(f'<path d="M{last_x},{y+box_height+5} V{y+109} H{30+box_width/2} V{y+128}" fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>')
    parts.append('</g></svg>')
    (OUT/f"{name}.svg").write_text("\n".join(parts), encoding="utf-8")
    _prefix_ids(OUT/f"{name}.svg")


def _svg_text(x, y, text, size=15, anchor="start", fill="#18334a"):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}">{html.escape(text)}</text>'


def _svg_box(x, y, width, lines, color=0):
    body = [f'<rect x="{x}" y="{y}" width="{width}" height="66" rx="10" fill="{COLORS[color]}" stroke="#bacbda"/>']
    body.extend(_svg_text(x+width/2, y+27+22*i, line, 16 if i==0 else 14, "middle") for i,line in enumerate(lines))
    return ''.join(body)


def _write_schematic(name, title, subtitle, height, body):
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 {height}" role="img" aria-label="{html.escape(title)}">',
           f'<title>{html.escape(title)}</title><desc>{html.escape(subtitle)}</desc>',
           '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="#64748b"/></marker></defs>',
           '<rect width="100%" height="100%" rx="16" fill="#f8fafc"/>',
           '<g font-family="Noto Sans CJK SC,Noto Sans SC,Microsoft YaHei,sans-serif" fill="#18334a">',
           _svg_text(34,40,title,23),_svg_text(34,70,subtitle,15),*body,'</g></svg>']
    path = OUT/f'{name}.svg'
    path.write_text('\n'.join(svg),encoding='utf-8')
    _prefix_ids(path)


def draw_transformer():
    body=[]
    for y, source, target, operation, detail in [
        (150,'X','U','因果注意力 + 输出投影','Q/K 施加 RoPE；逐头归一化'),
        (332,'U','Y','SwiGLU','SiLU(gate) × up → down'),
    ]:
        body += [_svg_box(30,y,100,[source,'[B,T,d]']),_svg_box(190,y,150,['RMSNorm','仅作用于支路'],1),
                 _svg_box(390,y,290,[operation,detail],2),_svg_box(890,y,155,[target,'[B,T,d]'],3)]
        for start,end in [(130,190),(340,390),(680,745),(795,890)]:
            body.append(f'<path d="M{start},{y+33} H{end-4}" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>')
        body += [f'<circle cx="770" cy="{y+33}" r="25" fill="white" stroke="#64748b" stroke-width="2"/>',
                 _svg_text(770,y+41,'+',26,'middle'),
                 f'<path d="M80,{y} V{y-36} H770 V{y+6}" fill="none" stroke="#087e8b" stroke-width="3" marker-end="url(#arrow)"/>',
                 _svg_text(440,y-49,f'{source} 的恒等旁路',15,'middle',fill='#087e8b')]
    body += ['<path d="M967,216 V254 H18 V365 H27" fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>',
             _svg_text(540,437,'先 U = X + Attention(Norm(X))，再 Y = U + MLP(Norm(U))',16,'middle')]
    _write_schematic('transformer','pre-norm Transformer：明确画出两次残差相加',
                     '绿色旁路传递原状态；归一化在子层支路上，MLP 读取注意力更新后的 U',465,body)


def draw_unigram():
    body=[]
    for x,offset,prefix in [(180,0,'空前缀'),(535,1,'a'),(890,2,'ab')]:
        body += [f'<circle cx="{x}" cy="220" r="28" fill="{COLORS[offset]}" stroke="#bacbda"/>',
                 _svg_text(x,227,str(offset),21,'middle'),_svg_text(x,270,prefix,15,'middle')]
    body += ['<path d="M208,220 H503 M563,220 H858" stroke="#64748b" stroke-width="2" fill="none" marker-end="url(#arrow)"/>',
             '<path d="M198,198 C340,95 730,95 872,198" stroke="#087e8b" stroke-width="3" fill="none" marker-end="url(#arrow)"/>',
             _svg_text(355,202,'a：0.4',16,'middle'),_svg_text(712,202,'b：0.4',16,'middle'),
             _svg_text(535,120,'ab：0.2',16,'middle',fill='#087e8b'),
             _svg_text(540,316,'两条路径：[ab] 权重 0.2；[a,b] 权重 0.4 × 0.4 = 0.16',16,'middle'),
             _svg_text(540,344,'Viterbi 取 max = 0.2；forward 求和 = 0.36',16,'middle')]
    _write_schematic('unigram-paths','Unigram：在字符边界图上选择路径或累加路径',
                     '结点是已读字符数，边是词片；这是固定字符串的切分权重，尚未定义停止分布',373,body)


def draw_packing():
    names=['A0','A1','A2','B0','B1','B2','PAD']
    groups=[0,0,0,1,1,1,-1]
    body=[_svg_text(34,108,'A0/B0 是 BOS；A2/B2 是 EOS；PAD 不属于任何文档',14),
          _svg_text(282,137,'注意力：行 query → 列 key',15,'middle'),
          _svg_text(620,137,'移位后的输入 → 目标',15),_svg_text(963,137,'loss mask',15,'middle')]
    for row,label in enumerate(names):
        body += [_svg_text(110,191+row*40,label,14,'end'),_svg_text(148+row*40,160,label,13,'middle')]
        for col in range(len(names)):
            allowed=col<=row and groups[row]>=0 and groups[col]==groups[row]
            x,y=128+col*40,172+row*40
            color='#d6e9fa' if allowed else '#edf0f3'
            body += [f'<rect x="{x}" y="{y}" width="38" height="38" fill="{color}"/>',
                     _svg_text(x+19,y+25,'1' if allowed else '0',14,'middle',fill='#18334a' if allowed else '#8392a0')]
    for row,(left,right) in enumerate(zip(names,names[1:])):
        valid=groups[row]>=0 and groups[row]==groups[row+1]
        y=192+row*41
        if not valid:body.append(f'<rect x="526" y="{y-24}" width="506" height="35" rx="5" fill="#fff1d6"/>')
        body += [_svg_text(646,y,f'{left} → {right}',17),_svg_text(963,y,str(int(valid)),17,'middle')]
    body += [_svg_text(540,488,'阻断跨文档注意力，不会自动删除 A2 → B0 标签；EOS 的监督仍要保留。',16,'middle'),
             _svg_text(540,512,'全空 PAD query 另作数值处理；角色监督还需与目标位置的掩码相交。',14,'middle')]
    _write_schematic('packing-boundaries','Packing：可见关系与目标边界是两套独立约束',
                     '例：BOS_A、a、EOS_A、BOS_B、b、EOS_B、PAD；按独立文档似然训练',538,body)


def main():
    draw_transformer()
    draw_unigram()
    draw_packing()
    diagram('vocabulary-loss', '大词表交叉熵：前向归约与反向重算', [
        [('隐藏状态与权重', 'H [M,d]；W [V,d]'), ('词表块 [M,b]', '每块计算 logits 与 LSE'), ('全局行统计', '合并 LSE，减目标 logit'), ('平均 loss', '保存 H/W/标签/行统计')],
        [('反向收到 dLoss', '使用相同输入与权重'), ('重算块概率', 'exp(logits − LSE)'), ('减目标 one-hot', '除有效 token 总数'), ('累加 dH / 写 dW', '不保存完整 [M,V] logits')],
    ], '数学等价不代表 GPU 性能等价；普通 autograd 循环可能仍保留每块中间量', connect_rows=False)
    diagram('mla-absorption', 'MLA 内容分支：两种等价的计算路径', [
        [('展开 K / V', '由每个历史 latent 投影'), ('逐头计算分数', 'q 与展开的 K 点积'), ('聚合 V 再投影', '各头输出相加')],
        [('吸收投影', 'query 转到 latent 空间'), ('逐头聚合 latent', '保留各自 softmax 分布'), ('合并 V / O 投影', '得到相同的内容输出')],
    ], '恒等式假定线性投影；RoPE 位置分支与非线性归一化需要单独处理', connect_rows=False)
    mechanisms = ROOT/'book/mechanism-results.json'
    if mechanisms.exists():
        result = json.loads(mechanisms.read_text(encoding="utf-8"))
        probability = np.geomspace(.001, .99, 240)
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for false_positive, color in [(.01, '#087e8b'), (.05, '#35618d'), (.1, '#c57b22')]:
            precision = .9 * probability / (.9 * probability + false_positive * (1 - probability))
            ax.plot(probability, precision, color=color, label=f'TPR=0.9, FPR={false_positive}')
        ax.scatter(result['verifier']['base_success'], result['verifier']['accepted_precision'], color='#35618d', zorder=3)
        ax.set(xscale='log', xlabel='Generator success probability', ylabel='Accuracy among accepted candidates',
               title='Rare correct candidates make false accepts costly', ylim=(0, 1.02))
        ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
        fig.savefig(OUT/'verifier-base-rate.svg'); plt.close(fig)
    review_path = ROOT/'book/chapter-review-results.json'
    if review_path.exists():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        diagram('nope-depth', 'NoPE 的顺序不变性：需要说明网络深度', [
            [('前缀交换', '[3,5,9] → [5,3,9]'), ('单层最后 query', '读取相同 K/V 集合'), ('最后 logits 不变', '差值 0：单层对照')],
            [('首层较早位置', '已读取不同前缀'), ('第二层 K/V', '隐藏表示集合改变'), ('双层能够区分顺序', f"最大差值 {review['nope_last_logit_difference']['2']:.3f}")],
        ], '随机初始化小模型的实测反例；因果 mask、训练后利用和长度泛化需分别验证', connect_rows=False)
        context = np.geomspace(1024, 131072, 100)
        full = context/8192
        hybrid = full/4 + .25
        fig, ax = plt.subplots(figsize=(8.4, 4.7))
        ax.plot(context/1024, full, label='Full attention', color='#52687c')
        ax.plot(context/1024, hybrid, label='Hybrid: 1/4 KV + 0.25 GiB state', color='#087e8b')
        ax.plot(context/1024, full/4, '--', label='Only the growing KV term', color='#c57b22')
        ax.set(xlabel='Context length (Ki tokens)', ylabel='History state (GiB)',
               title='Fixed recurrent state matters at short contexts')
        ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
        fig.savefig(OUT/'hybrid-state.svg'); plt.close(fig)
        scaling = review.get('interfaces',{}).get('rope_scaling')
        if scaling:
            fig,ax = plt.subplots(figsize=(8.4,4.6))
            for key,label,color in [('pi','PI: every frequency / 32','#c57b22'),
                                    ('ntk','NTK-aware: enlarge the base','#35618d'),
                                    ('yarn','YaRN: keep fast, interpolate slow','#087e8b')]:
                ax.plot(range(len(scaling[key])),scaling[key],label=label,color=color,lw=2)
            ax.set_yscale('log',base=2)
            ax.set_yticks([1,2,4,8,16,32],labels=['1','2','4','8','16','32'])
            ax.set(xlabel='Rotary pair index (0 = fastest)',ylabel='Frequency divisor (original / scaled)',
                   title='4K to 128K: dim=128, base=10000; amplitude scaling separate')
            ax.grid(alpha=.2);ax.legend(fontsize=9);fig.tight_layout()
            fig.savefig(OUT/'rope-scaling.svg');plt.close(fig)
    diagram("byte-hierarchy", "字节级层级模型：把重计算放到动态单元", [
        [("UTF-8 字节输入", "基本字节/结构标记"),("局部编码与边界", "BLT 熵 patch / H-Net chunk"),("压缩层级主干", "较短可变长度序列"),("局部解码", "恢复逐字节预测")],
    ], "省去固定外部分词器，仍需因果边界、编码与可验证的计算预算")
    diagram("rl-consistency", "RL 一致性诊断：先冻结权重，再允许更新", [
        [("实际 rollout 引擎 μ", "版本、dtype、路由、mask"),("旧训练引擎 π old", "冻结权重 replay 同一前缀"),("当前策略 π θ", "更新/数据复用造成 lag"),("核对目标", "log-ratio 尾部与支持")],
    ], "μ 与 π old 的差异是引擎问题；π θ 与 π old 的差异是更新问题")
    diagram("research-loop", "研究闭环：失败分类、干预与证据", [
        [("定位失败", "知识 / 检索 / 推理 / 格式"),("提出可证伪假设", "预测什么条件下改善"),("固定预算与对照", "数据、参数、token、时间"),("决定是否扩大", "收益、不确定性、成本")],
    ], "先验证正确性，再运行可区分不同解释的实验")
    diagram("retrieval-pipeline", "RAG：每个阶段都需要独立评估", [
        [("问题与权限", "锁定可见文档版本"),("候选检索", "稀疏 / 稠密 / 混合"),("重排序与证据整理", "去重、位置、长度、引文"),("回答与核验", "正确性、支撑率、拒答")],
        [("闭卷对照", "测参数记忆基线"),("人工充分证据", "隔离检索器失败"),("错误 / 缺失证据", "测冲突与不确定性"),("按版本回归", "索引、模型、提示共同固定")],
    ], "检索召回率不等于答案正确率；引文存在不等于引文支持答案",connect_rows=False)
    diagram("activation-patching", "激活干预：三次运行，共用同一固定模型", [
        [("Clean 输入", "固定 token 与位置"),("正常前向", "缓存指定 clean 激活"),("Clean 指标", "m(clean)")],
        [("Corrupt 输入", "只改选定因素"),("正常前向", "不安装替换 hook"),("Corrupt 指标", "m(corrupt)")],
        [("Corrupt 输入", "重新执行前向"),("局部 patch", "从 clean 缓存取替换值"),("Patched 指标", "m(patched)；计算恢复比")],
    ], "第三次仅改变指定激活；无关位置与多输入对照另测，结束后移除 hook",connect_rows=False)
    diagram("learning-map", "从数据到行为：全书知识依赖", [
        [("概率与 token", "目标、表示、边界"),("Transformer", "位置、注意力、MLP"),("稳定训练", "优化器、数值与验证"),("架构选择", "MoE、状态与记忆")],
        [("资源核算", "FLOPs、显存、读写"),("高效内核", "融合、在线 softmax"),("分布式与推理", "通信、KV、服务"),("缩放与数据", "预算、分布与质量")],
        [("可靠评估", "协议、区间、污染"),("SFT 与偏好", "条件似然、KL、DPO"),("推理强化学习", "采样、验证、估计量"),("多模态与项目", "感知接口、实验闭环")],
    ], "按知识依赖重组 2024 / 2025 / 2026 的公开核心课程")
    diagram("online-softmax", "在线 softmax：每个 query 行的状态更新", [
        [("旧状态", "m：最大值，ℓ：分母", "u：未归一化输出"),("读入 K/V 新块", "计算局部 QKᵀ 与 mask"),("统一数值尺度", "m′ = max(m, max scores)", "α = exp(m − m′)"),("更新与输出", "ℓ′ = αℓ + 新块分母", "u′ = αu + 新块分子")],
    ], "所有块处理完毕：O = u / ℓ；不把完整 T × T 概率矩阵写入 HBM")
    diagram("parallelism", "不同并行策略切分不同对象", [
        [("数据并行 DP", "样本分片，复制参数"),("FSDP / ZeRO", "分片状态，按需聚集"),("张量并行 TP", "切分层内矩阵"),("流水线 PP", "切分层，调度 microbatch")],
        [("序列并行 SP", "分片独立逐位置算子"),("上下文并行 CP", "长序列与 KV 跨卡"),("专家并行 EP", "路由到专家，再合并"),("共同约束", "拓扑、带宽、激活峰值")],
    ], "它们改变的通信路径不同；性能取决于组网与调度", connect=False, connect_rows=False)
    diagram("data-pipeline", "数据流水线与证据记录", [
        [("原始来源", "WARC / 代码 / 文档"),("抽取与规范化", "保留原始文本与来源"),("语言与质量过滤", "审计误删、漏删"),("精确与近似去重", "哈希 / MinHash / 子串")],
        [("按簇划分数据", "防止重复跨集合"),("混合与 packing", "记录比例、边界与掩码"),("训练消融", "固定验证集与计算量"),("数据卡与溯源", "版本、统计、保留理由")],
    ], "每一步都在改变训练分布；必须能追查哪些文档为何被保留")
    diagram("multimodal", "多模态的两类接口：理解输入与生成输出", [
        [("理解路径：图像 / 视频", "分辨率、帧率、裁剪"),("视觉编码器", "patch → 语义特征"),("Projector / Resampler", "投影宽度、压缩 token"),("语言模型", "视觉 + 文本 → 回答")],
        [("生成路径：图文前缀", "按模型接口编码"),("多模态主干", "预测离散码或连续条件"),("专用图像解码器", "VQ / diffusion / flow"),("图像输出", "生成质量单独评估")],
    ], "两行是不同接口路线；问答微调不会自动赋予模型一个图像解码器", connect_rows=False)
    plot_charts()


# Validated categorical order (blue, orange, aqua, yellow, magenta, green) and a one-hue blue ramp.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def style(ax, title, xlabel, ylabel, grid_axis="both"):
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_axisbelow(True)
    ax.grid(color=GRID, lw=1, which="major", axis=grid_axis)
    ax.tick_params(colors=MUTED)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, facecolor="#fcfcfb")
    plt.close(fig)


def label_end(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", fontsize=9, color=INK)


def legend_right(ax):
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9, labelcolor=INK)


def plot_tinystories(data):
    """Plot the baseline and hyperparameters in this result record, including skipped scans."""
    runs = {row['name']:row for row in data['runs']}
    base_name = data['base_run']
    if base_name not in runs:
        raise ValueError('TinyStories base_run is missing from the recorded runs')
    base = runs[base_name]
    steps = data['train_settings']['steps']
    total = base['parameters']['total']
    setting = f"{total / 1e6:.1f}M params, {steps:,} steps, {data.get('device', 'cpu').upper()}"

    def curve(ax, row, label, color, **kwargs):
        history = [point for point in row['history'] if point['step'] > 0]
        values = [point.get('val_loss') for point in history]
        finite = [type(value) in (int, float) and math.isfinite(value) for value in values]
        losses = [value if valid else float('nan') for value,valid in zip(values,finite)]
        if not any(finite):
            suffix = 'no finite val loss'
        elif row.get('diverged') or not finite[-1]:
            suffix = f"failed; last finite {next(v for v,ok in reversed(list(zip(values,finite))) if ok):.3f}"
        else:
            suffix = f'{values[-1]:.3f}'
        ax.plot([point['step'] for point in history], losses, color=color, lw=2,
                label=f'{label}: {suffix}', **kwargs)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    sweep = sorted([row for name,row in runs.items() if name.startswith('lr_')], key=lambda row:row['lr'])
    if sweep:
        for index,row in enumerate(sweep):
            curve(ax,row,f"lr {row['lr']:g}",RAMP[index % len(RAMP)])
        title = f'TinyStories learning-rate sweep ({setting})'
    else:
        curve(ax,base,f"selected lr {base['lr']:g}",RAMP[-1])
        title = f'TinyStories selected learning rate; scan skipped ({setting})'
    legend_right(ax)
    style(ax,title,'Optimizer updates','Validation loss (nats / token)')
    save(fig,'tinystories-lr.svg')

    fig, ax = plt.subplots(figsize=(9, 4.8))
    names = [(base_name,'pre-norm RoPE SwiGLU (base)'), ('post_norm','post-norm'),
             ('tied_embeddings','tied embeddings'), ('silu_ffn','SiLU FFN, d_ff = 4d'),
             ('nope','no position (NoPE)'), ('no_norm_lr_div3','no RMSNorm, lr / 3')]
    for color,(name,label) in zip(SERIES,names):
        if name in runs:
            curve(ax,runs[name],label,color)
    if 'no_norm' in runs:
        row = runs['no_norm']
        curve(ax,row,f"no RMSNorm, lr {row['lr']:g}",MUTED,ls=(0,(1,2)))
    legend_right(ax)
    style(ax,f'TinyStories ablations, same {steps:,}-step budget','Optimizer updates','Validation loss (nats / token)')
    save(fig,'tinystories-ablations.svg')


def plot_charts():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb",
                         "font.family": ["DejaVu Sans"]})
    # Roofline with the course's H100 numbers: 989 TFLOP/s BF16 dense, 3.35 TB/s HBM.
    fig, ax = plt.subplots(figsize=(9, 4.6))
    intensity = np.geomspace(0.1, 3000, 400)
    peak, bandwidth = 989, 3.35
    ax.loglog(intensity, np.minimum(peak, bandwidth * intensity), color=SERIES[0], lw=2)
    ridge = peak / bandwidth
    ax.axvline(ridge, color=MUTED, lw=1)
    ax.annotate(f"ridge ≈ {ridge:.0f} FLOPs/byte", (ridge, 30), xytext=(6, 0), textcoords="offset points", fontsize=9, color=INK)
    for x, name in [(1, "matvec model (B=1)"), (64, "batch-64 matmul model"), (2000, "large-matmul example")]:
        y = min(peak, bandwidth * x)
        ax.plot(x, y, "o", color=SERIES[1], ms=6, mec="#fcfcfb", mew=2)
        ax.annotate(name, (x, y), xytext=(5, -13), textcoords="offset points", fontsize=8.5, color=INK)
    style(ax, "H100 matmul roofline: theoretical bounds, not measured points", "Arithmetic intensity (FLOPs / byte)", "Ideal upper bound (TFLOP/s)")
    ax.xaxis.labelpad = 10  # Embedded book fonts are taller than Matplotlib's math tick-label metrics.
    save(fig, "roofline.svg")
    # Scaling: synthetic IsoFLOP curves on the known surface.
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ns = np.geomspace(1, 1e5, 300)
    for i, c in enumerate([1e4, 1e5, 1e6]):
        loss = 1 + 2 * ns ** -.35 + (c / (6 * ns)) ** -.3
        ax.semilogx(ns, loss, color=RAMP[2 * i + 1], lw=2, label=f"C = {c:.0e}")
        j = np.argmin(loss)
        ax.plot(ns[j], loss[j], "o", color=RAMP[2 * i + 1], ms=7, mec="#fcfcfb", mew=2)
    ax.set_ylim(1, 4)
    legend_right(ax)
    style(ax, "Synthetic IsoFLOP curves, C = 6ND (dots: minimum)", "Model size N (normalized units)", "Synthetic loss")
    save(fig, "scaling.svg")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    probabilities = np.linspace(0, 1, 401)
    for i, group in enumerate([2, 4, 8, 16]):
        y = 1 - probabilities ** group - (1 - probabilities) ** group
        ax.plot(probabilities, y, color=RAMP[i + 1], lw=2, label=f"G = {group}")
    legend_right(ax)
    style(ax, "Probability that a group has both rewards (IID Bernoulli, analytical)", "Success probability for one prompt", "P(mixed group)")
    save(fig, "group-signal.svg")
    results = json.loads((ROOT / "book/experiment-results.json").read_text(encoding="utf-8"))
    hist = results["arithmetic"]["120"]["history"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot([h["step"] for h in hist], [h["val_loss"] for h in hist], "o-", color=SERIES[0], lw=2, ms=6, mec="#fcfcfb", mew=2)
    style(ax, "Measured CPU run: 125,248-parameter decoder on synthetic arithmetic text", "Optimizer updates", "Validation NLL (nats / byte)")
    save(fig, "training.svg")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    roles = ["operand", "format", "answer"]
    for ax, steps in zip(axes, ["120", "1000"]):
        metrics = results["arithmetic"][steps]["diagnostics"]
        values = [metrics["position_metrics"][r]["nll"] for r in roles]
        bars = ax.bar(roles, values, color=SERIES[0], width=0.22)
        ax.axhline(np.log(10), color=MUTED, lw=1)
        for bar, v in zip(bars, values):
            ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK)
        style(ax, f"{steps} steps: exact answers {metrics['all_100_prompt_accuracy']:.0%}", "", "NLL (nats / target byte)" if steps == "120" else "", "y")
    axes[0].annotate("ln 10 = uniform over 10 digits", (1, np.log(10)), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8.5, color=MUTED)
    save(fig, "loss-by-role.svg")
    tiny = ROOT / "book/tinystories-results.json"
    if tiny.exists():
        plot_tinystories(json.loads(tiny.read_text(encoding="utf-8")))
    grpo = ROOT / "book/grpo-results.json"
    if grpo.exists():
        data = json.loads(grpo.read_text(encoding="utf-8"))
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for color, method, text in [(SERIES[0], "grpo", "GRPO (÷ std)"), (SERIES[1], "dr_grpo", "Dr. GRPO (centered)"), (SERIES[2], "maxrl", "MaxRL (÷ mean)")]:
            logs = [r["log"] for r in data["runs"] if r["method"] == method]
            steps = [x["step"] for x in logs[0]]
            acc = np.array([[x["train_accuracy"] for x in log] for log in logs])
            ax.fill_between(steps, acc.min(0), acc.max(0), color=color, alpha=0.12, lw=0)
            ax.plot(steps, acc.mean(0), color=color, lw=2, label=text)
        ax.set_ylim(0, 1.02)
        legend_right(ax)
        style(ax, "Tiny-LM GRPO: accuracy on the 80 RL prompts (mean, band = min–max of 3 seeds)", "RL steps", "Exact-answer accuracy")
        save(fig, "grpo-variants.svg")
    advanced = ROOT / "book/advanced-experiments.json"
    if advanced.exists():
        records = json.loads(advanced.read_text(encoding="utf-8"))["verifier"]["records"]
        fig, ax = plt.subplots(figsize=(9, 4.6))
        xs = [r["k"] for r in records]
        for color, key, text in [(SERIES[0], "oracle_coverage", "at least one correct candidate"), (SERIES[1], "selected_accuracy", "selected by imperfect verifier")]:
            ys = [r[key] for r in records]
            ax.semilogx(xs, ys, "o-", color=color, lw=2, ms=6, mec="#fcfcfb", mew=2, base=2, label=text)
        ax.set_ylim(0, 1.05)
        legend_right(ax)
        style(ax, "Synthetic verifier: wider coverage can worsen selection", "Number of candidates k", "Probability / accuracy")
        save(fig, "verifier-selection.svg")
    frontier = ROOT / "book/frontier-results.json"
    if frontier.exists():
        rows = {r["repo"]: r for r in json.loads(frontier.read_text(encoding="utf-8"))["architectures"]["rows"]}
        # Colour and marker both encode the attention/KV design; every point is also labelled.
        groups = [("Full attention (MHA / GQA)", SERIES[0], "o", ["allenai/OLMo-2-0425-1B", "HuggingFaceTB/SmolLM3-3B",
                   "allenai/OLMo-2-1124-7B", "Qwen/Qwen3-8B", "mistralai/Mistral-Small-24B-Base-2501", "Qwen/Qwen3-30B-A3B",
                   "Qwen/Qwen3-235B-A22B", "zai-org/GLM-4.5"]),
                  ("Banded window + full (1:1)", SERIES[1], "s", ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]),
                  ("Linear-attention hybrid (3:1, 7:1)", SERIES[2], "^", ["Qwen/Qwen3-Next-80B-A3B-Instruct",
                                                                          "MiniMaxAI/MiniMax-M1-80k"]),
                  ("MLA latent cache", SERIES[3], "D", ["deepseek-ai/DeepSeek-V3", "moonshotai/Kimi-K2-Instruct"])]
        labels = {"allenai/OLMo-2-0425-1B": ("OLMo 2 1B", (7, 0)), "HuggingFaceTB/SmolLM3-3B": ("SmolLM3 3B", (7, 0)),
                  "allenai/OLMo-2-1124-7B": ("OLMo 2 7B", (7, 0)), "Qwen/Qwen3-8B": ("Qwen3-8B", (7, 0)),
                  "mistralai/Mistral-Small-24B-Base-2501": ("Mistral Small 24B", (7, 4)),
                  "Qwen/Qwen3-30B-A3B": ("Qwen3-30B-A3B", (7, -4)), "Qwen/Qwen3-235B-A22B": ("Qwen3-235B-A22B", (7, 0)),
                  "zai-org/GLM-4.5": ("GLM-4.5", (7, 0)), "openai/gpt-oss-20b": ("gpt-oss-20b", (7, 0)),
                  "openai/gpt-oss-120b": ("gpt-oss-120b", (7, 0)), "Qwen/Qwen3-Next-80B-A3B-Instruct": ("Qwen3-Next-80B-A3B", (8, 0)),
                  "MiniMaxAI/MiniMax-M1-80k": ("MiniMax-M1", (8, 0)),
                  "deepseek-ai/DeepSeek-V3": ("DeepSeek-V3", (-8, 0)), "moonshotai/Kimi-K2-Instruct": ("Kimi K2", (8, 0))}
        fig, ax = plt.subplots(figsize=(9, 5.2))
        for text, color, marker, repos in groups:
            xs = [rows[r]["total_b"] for r in repos]
            ys = [rows[r]["kv_kib_per_token"] for r in repos]
            ax.scatter(xs, ys, s=90 if marker == "^" else 64, color=color, marker=marker, edgecolors="#fcfcfb", linewidths=2, label=text, zorder=3)
            for repo, x, y in zip(repos, xs, ys):
                name, offset = labels[repo]
                ax.annotate(name, (x, y), xytext=offset, textcoords="offset points", fontsize=8.5, color=INK,
                            va="center", ha="left" if offset[0] > 0 else "right")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.8, 2500)
        ax.set_ylim(15, 800)
        ax.set_yticks([16, 32, 64, 128, 256, 512])
        ax.set_yticklabels(["16", "32", "64", "128", "256", "512"])
        ax.set_xticks([1, 3, 10, 30, 100, 300, 1000])
        ax.set_xticklabels(["1", "3", "10", "30", "100", "300", "1000"])
        ax.minorticks_off()
        legend_right(ax)
        style(ax, "KV at 32K context (BF16); fixed recurrent state excluded",
              "Total parameters (billions, log scale)", "KV only (KiB per token, log scale)")
        save(fig, "open-models.svg")
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4.3))
    theta = np.linspace(0, .1, 201)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    chosen = sigmoid(-2 * theta) * sigmoid(theta)
    rejected = sigmoid(-2 * theta) * sigmoid(-theta)
    left.plot(theta, chosen, label="chosen probability", color="#087e8b", lw=2)
    left.plot(theta, rejected, label="rejected probability", color="#c28636", lw=2)
    left.axvline(.05, color="#607181", ls="--", lw=1)
    left.set(xlabel="Shared parameter θ", ylabel="Sequence probability", title="Both probabilities can decrease")
    left.legend(fontsize=8); left.grid(alpha=.2)
    right.plot(theta, chosen / rejected, color="#8c5072", lw=2)
    right.axvline(.05, color="#607181", ls="--", lw=1)
    right.set(xlabel="Shared parameter θ", ylabel="chosen / rejected", title="Relative preference still improves")
    right.grid(alpha=.2)
    fig.suptitle("Analytical normalized toy model; one DPO step reaches θ = 0.05", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "dpo-coupling.svg"); plt.close(fig)

if __name__ == "__main__":
    main()

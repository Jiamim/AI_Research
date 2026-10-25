#!/usr/bin/env python3
"""Integrate diagrams, executable examples and sources into the main chapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.technique_catalog import CARDS, CHAPTERS
from experiments.result_utils import write_text

READINGS = json.loads((ROOT/'references'/'reading_links.json').read_text(encoding='utf-8'))

CHAPTER_SUMMARIES = {
    '01': '先区分整数编码与重建值，再学习尺度、零点和分组；从误差与裁剪进入整数点积、输出重定标和字节打包。',
    '02': '沿“输出误差 → 约束补偿 → 二维手算 → 顺序更新”阅读，再比较 Cholesky、damping、非对称校准与损失感知网格，分清执行分块和量化分组。',
    '03': '先看离群通道分流，再用缩放恒等式比较 SmoothQuant 与 AWQ；接着学习逐层蒸馏、通道重排和输出偏置修正。',
    '04': '把裁剪、平移、尺度和舍入决策看作可优化变量；对照有数据、无数据校准，并补齐 AdaRound 与 BRECQ 的基础联系。',
    '05': '先证明正交变换保持什么，再检查它可以放在网络的哪些位置。随后比较固定旋转、学习旋转和一般可逆变换。',
    '06': '从标量码本的加权均值出发，扩展到向量码本、加法编码和稀疏例外，再分析位宽分配与组合压缩的完整成本。',
    '07': '先解决 round 的反向传播和主权重累积，再区分各类低精度训练；最后检查 QAT 状态切换、QLoRA 显存与合并后的函数。',
    '08': '先算 KV 缓存大小和误差传播，再看缓存如何编码、更新与恢复；对照低精度 Attention、物理分页和分步缓存正确性。',
    '09': '先读懂浮点代表值与缩放，再从容器走到块布局和执行路线；最后闭合保存、重载、Q/DQ 图与性能验证流程。',
    '12': '比较结构化旋转、向量与有状态编码，再理解组级和表示内部的位宽分配；重点区分 GPTVQ/VPTQ 与不同谱代理。',
    '13': '从尺度回归与 FP4 格式进入鲁棒目标、剪枝、低秩补偿和超低比特标量编码，按公开版本核对局部最优的条件。',
    '14': '先看梯度估计与训练数据，再比较专家共享、补偿和位宽分配；结合路由代理反例与循环 Transformer 的适用范围。',
    '15': '从 K/V 几何编码与 Attention 量化进入窗口/位宽预算、MLA、循环状态与条件查找表，最后对照实际低比特服务路径。',
}


def reading_link(resource_id):
    resource = READINGS['resources'][resource_id]
    link = f'[{resource["title"]}]({resource["url"]})'
    if resource.get('fulltext_url'):
        link += f'（[所读版本正文]({resource["fulltext_url"]})）'
    if resource.get('access_note'):
        link += f'（{resource["access_note"]}）'
    return link


def reading_lines(card):
    topic = READINGS['techniques'][card['id']]
    blogs = [key for key in topic['readings'] if READINGS['resources'][key]['kind'] == 'blog']
    other = [key for key in topic['readings'] if key not in blogs]
    lines = ['### 论文与延伸阅读', '']
    if topic['papers']:
        lines.append('- **论文**：'+'；'.join(reading_link(key) for key in topic['papers'])+'。')
    if blogs:
        lines.append('- **博客与技术文章**：'+'；'.join(reading_link(key) for key in blogs)+'。')
    if other:
        lines.append('- **作者项目、官方文档与规范**：'+'；'.join(reading_link(key) for key in other)+'。')
    if topic['note']:
        lines.extend(['', topic['note']])
    lines.append('')
    return lines


def validate_readings():
    if set(READINGS['techniques']) != {card['id'] for card in CARDS}:
        raise ValueError('Reading-link coverage must match all tutorial techniques.')
    for key, topic in READINGS['techniques'].items():
        if not topic['readings'] and not (topic['papers'] and topic['note']):
            raise ValueError(f'Provide verified reading links or explain the paper-only evidence for {key}')
        if not topic['papers'] and not topic['note']:
            raise ValueError(f'Explain the primary source when no standalone paper applies: {key}')
        for resource_id in topic['papers'] + topic['readings']:
            resource = READINGS['resources'][resource_id]
            if not resource['url'].startswith('https://'):
                raise ValueError(f'Expected an HTTPS source URL: {resource_id}')
            if resource_id in topic['papers'] and resource['kind'] != 'paper':
                raise ValueError(f'Do not label a blog, code repository, or format specification as a paper: {resource_id}')
    supplementary = READINGS.get('supplementary_topics', [])
    if len({topic['id'] for topic in supplementary}) != len(supplementary):
        raise ValueError('Supplementary topic IDs must be unique.')
    for topic in supplementary:
        if not topic['guide'].startswith('docs/') or not topic['resources']:
            raise ValueError(f'Missing supplementary topic destination or sources: {topic["id"]}')
        for key in topic['resources']:
            if key not in READINGS['resources']:
                raise ValueError(f'Unknown supplementary source: {key}')


def wrapped(text, limit=25):
    lines, line, width = [], "", 0
    # Modifier-letter superscripts are missing from several common CJK fonts.
    text = text.replace('⁻ᵀ', '^(-T)').replace('ᵀ', '^T')
    for char in text:
        if char == '\n':
            lines.append(line)
            line, width = "", 0
            continue
        size = 2 if unicodedata.east_asian_width(char) in "WF" else 1
        if width + size > limit:
            lines.append(line)
            line, width = "", 0
        line += char
        width += size
    if line:
        lines.append(line)
    return lines


def diagram(card):
    fork = card["id"] in {"llm_int8", "lora", "qlora", "kivi"}
    comparison = card["id"] == "dynamic_scaling"
    loops = {"gptq": 1, "lloyd_max": 1, "autoround": 1, "hqq": 0,
             "spinquant": 0, "qat_ste": 1, "pact": 0, "loftq": 0}
    height = 450 if fork else 340
    title = escape(card["title"])
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}" role="img" aria-labelledby="title desc">',
           f'<title id="title">{title}</title>',
           f'<desc id="desc">{escape(card["principle"])}；' + '；'.join(escape(s) for s in card['steps']) + '</desc>',
           '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#31748f"/></marker></defs>',
           f'<rect width="1100" height="{height}" rx="18" fill="#f5f8fc"/>',
           '<g font-family="Noto Sans CJK SC, Microsoft YaHei, sans-serif">',
           f'<text x="30" y="44" font-size="24" font-weight="700" fill="#15324f">{title}</text>']
    positions = [(24, 176), (420, 85), (420, 258), (828, 176)] if fork else [(24+270*i, 100) for i in range(4)]
    width, box_height = 248, 124

    def arrow(path, dashed=False):
        svg.append(f'<path d="{path}" fill="none" stroke="#31748f" stroke-width="2.5" marker-end="url(#arrow)"' + (' stroke-dasharray="6 5"' if dashed else '') + '/>')

    if fork:
        arrow('M272 238 L335 238 L335 147 L420 147')
        arrow('M272 238 L335 238 L335 320 L420 320')
        arrow('M668 147 L748 147 L748 238 L828 238')
        arrow('M668 320 L748 320 L748 238 L828 238')
    elif not comparison:
        for i in range(3):
            x = positions[i][0]+width
            arrow(f'M{x} 162 L{positions[i+1][0]} 162')
    for i, (step, (x, y)) in enumerate(zip(card["steps"], positions)):
        svg.append(f'<rect x="{x}" y="{y}" width="{width}" height="{box_height}" rx="12" fill="white" stroke="#c7d6e5" stroke-width="1.5"/>')
        svg.append(f'<text x="{x+18}" y="{y+27}" font-size="14" font-weight="700" fill="#31748f">{i+1:02d}</text>')
        lines = wrapped(step)
        for j, line in enumerate(lines):
            svg.append(f'<text x="{x+width/2}" y="{y+54+j*24}" font-size="17" text-anchor="middle" fill="#173650">{escape(line)}</text>')
    if card["id"] in loops:
        end_x = positions[loops[card['id']]][0]+width/2
        arrow(f'M952 224 L952 276 L{end_x} 276 L{end_x} 224', dashed=True)
        svg.append('<text x="590" y="267" font-size="13" fill="#31748f">继续下一轮</text>')
    caption = "三种尺度时序的对照；本步统计可供未来使用" if comparison else "教学数据流；实际实现范围与省略内容见配套代码说明"
    svg.append(f'<text x="30" y="{height-19}" font-size="13" fill="#546c82">{caption}</text></g></svg>')
    return '\n'.join(svg)+'\n'


def chapter_filename(chapter):
    return f"{chapter}_{CHAPTERS[chapter]}.md"


def inline_example(card, heading=True):
    """Only the practice material: the preceding chapter prose owns the theory."""
    lines = [f'<!-- inline-technique: {card["id"]} -->', '']
    if heading:
        lines.extend(['### 图解与代码', ''])
    lines.extend([f'![{card["title"]}](../figures/techniques/{card["id"]}.svg)', '',
                  f'**运行：** `python experiments/technique_lab.py --technique {card["id"]}`', '',
                  f'<!-- executable: {card["id"]} -->', '```python', card['code'], '```', '',
                  '**代码讲解**', ''])
    lines.extend(f'{i}. {item}' for i, item in enumerate(card['explanation'], 1))
    lines.extend(['', f'> **示例边界：** {card["boundary"]}', '',
                  '<details>', '<summary>论文、博客与官方资料</summary>', '',
                  *reading_lines(card)[2:], '</details>', '', '<!-- /inline-technique -->'])
    return '\n'.join(lines)


def validate_markers(source):
    """Reject broken generated regions before they can consume authored prose."""
    tags = 'chapter-overview|chapter-footer|inline-technique|publication|technique-links'
    pattern = rf'<!--\s*(/?)(?:({tags}))(?:\s*:\s*([^>]*?))?\s*-->'
    opened = None
    seen = set()
    for match in re.finditer(pattern, source):
        closing, tag, key = match.groups()
        key = key.strip() if key else None
        if closing:
            if opened is None or opened[0] != tag or key is not None:
                raise ValueError(f'Unmatched generated closing marker: {match[0]}')
            opened = None
        else:
            if opened is not None:
                raise ValueError(f'Nested or unclosed generated marker: {opened}')
            if tag in ('inline-technique', 'publication') and not key:
                raise ValueError(f'Missing technique ID: {match[0]}')
            opened = (tag, key)
            if opened in seen:
                raise ValueError(f'Duplicate generated region: {opened}')
            seen.add(opened)
    if opened is not None:
        raise ValueError(f'Unclosed generated marker: {opened}')


def strip_generated(source):
    """Recover editable chapter prose; preserve authored topic/practice slots."""
    validate_markers(source)
    from experiments.build_chapter_demos import strip as strip_chapter_demo
    source = strip_chapter_demo(source)
    for tag in ('chapter-overview', 'chapter-footer'):
        source = re.sub(rf'<!-- {tag} -->.*?<!-- /{tag} -->', '', source, flags=re.S)
    source = re.sub(r'<!-- inline-technique: [a-z0-9_]+ -->.*?<!-- /inline-technique -->', '', source, flags=re.S)
    source = re.sub(r'<!-- publication: [a-z0-9_]+ -->.*?<!-- /publication -->', '', source, flags=re.S)
    source = re.sub(r'<!-- technique-links: .*? -->.*?<!-- /technique-links -->', '', source, flags=re.S)
    anchors = '|'.join(re.escape(card['id']) for card in CARDS)
    source = re.sub(rf'<a id="(?:principle-[\d-]+|chapter-start|{anchors})"></a>\n?', '', source)
    source = re.sub(r'\n---\n\n\[返回学习目录\][^\n]*\n', '\n', source)
    return re.sub(r'\n{3,}', '\n\n', source).strip()


def chapter_overview(chapter, sections):
    lines = ['<!-- chapter-overview -->', '<a id="chapter-start"></a>', '',
             f'**阅读路线：** {CHAPTER_SUMMARIES[chapter]}', '',
             f'<details><summary>本章目录（{len(sections)} 节）</summary>', '']
    lines.extend(f'- [{number} {title}](#principle-{number.replace(".", "-")})' for number, title in sections)
    lines.extend(['', '</details>', '',
                  '每项技术的图示、代码和讲解紧跟原理。例子在项目根目录运行，只需 NumPy；'
                  '长篇参考资料可在各节末尾展开。[按技术查找](技术图解与代码索引.md) · [查看实验输出](../experiments/results/逐项技术实验结果.md)。', '',
                  '[分章测试与参考答案](#chapter-assessment) · [课程考核与综合练习](11_练习与答案.md#course-assessment-guide)', '',
                  '[本章实操：完整逐步注释](#chapter-demo)', '',
                  '<!-- /chapter-overview -->'])
    return '\n'.join(lines)


def chapter_footer(chapter):
    order = sorted(['00_基础与全景.md'] + [chapter_filename(key) for key in CHAPTERS]
                   + ['10_实验_评估与选型.md', '11_练习与答案.md', '16_未来趋势与研究路线.md'])
    index = order.index(chapter_filename(chapter))
    next_link = f'[下一章]({order[index+1]})' if index+1 < len(order) else '[技术速查](技术图解与代码索引.md)'
    return '\n'.join(['<!-- chapter-footer -->', '', '---', '',
                      f'[上一章]({order[index-1]}) · [本章目录](#chapter-start) · '
                      f'{next_link} · [学习首页](../README.md) · [论文与博客](../references/论文与博客导航.md)', '',
                      '<!-- /chapter-footer -->'])


def merged_chapter(chapter, cards, source):
    source = strip_generated(source)
    reference_start = re.search(r'^\[R\d+\]:', source, re.M)
    references = source[reference_start.start():].strip() if reference_start else ''
    if reference_start:
        source = source[:reference_start.start()].rstrip()
    headings = list(re.finditer(r'^## (\d+\.\d+) ([^\n]+)$', source, re.M))
    if not headings:
        raise ValueError(f'No numbered sections in chapter {chapter}')
    by_section = defaultdict(list)
    for card in cards:
        by_section[card['section']].append(card)
    if not set(by_section).issubset({match[1] for match in headings}):
        raise ValueError(f'Techniques target missing sections in chapter {chapter}')
    header = source[:headings[0].start()].strip()
    sections = [(match[1], match[2]) for match in headings]
    result = [header, chapter_overview(chapter, sections)]
    for i, match in enumerate(headings):
        number, title = match[1], match[2]
        end = headings[i+1].start() if i+1 < len(headings) else len(source)
        body = source[match.end():end].strip()
        section_cards = by_section[number]
        anchors = [f'<a id="principle-{number.replace(".", "-")}"></a>']
        for card in section_cards:
            key = card['id']
            if card.get('paper_id'):
                resource = READINGS['resources'][READINGS['techniques'][key]['papers'][0]]
                publication = (f'<!-- publication: {key} -->\n'
                               f'**文献时间：** 首次提交 {resource["first_submitted"]}；'
                               f'本次核对 {resource["reviewed_version"]}（版本日期 {resource["version_date"]}）。\n'
                               '<!-- /publication -->')
                body = publication+'\n\n'+body
            topic = f'<!-- topic: {key} -->'
            practice = f'<!-- practice: {key} -->'
            if topic in body or practice in body:
                if body.count(topic) != 1 or body.count(practice) != 1:
                    raise ValueError(f'Expected paired unique topic/practice slots for {key}')
                if body.index(topic) >= body.index(practice):
                    raise ValueError(f'Theory must precede practice for {key}')
                body = body.replace(topic, topic+f'\n<a id="{key}"></a>')
                body = body.replace(practice, practice+'\n\n'+inline_example(card, heading=False))
            elif len(section_cards) == 1:
                anchors.append(f'<a id="{key}"></a>')
                body += '\n\n'+inline_example(card)
            else:
                raise ValueError(f'Arrange a theory/practice pair for each method in shared section {number}: {key}')
        result.append('\n'.join(anchors)+f'\n\n## {number} {title}\n\n'+body)
    result.append(chapter_footer(chapter))
    if references:
        result.append(references)
    return '\n\n'.join(result)+'\n'


def outputs():
    ids = [card['id'] for card in CARDS]
    if len(set(ids)) != len(ids):
        raise ValueError('Technique IDs must be unique before generating any files.')
    validate_readings()
    files, coverage = {}, []
    grouped = defaultdict(list)
    ordered_cards = sorted(CARDS, key=lambda card: tuple(int(part) for part in card['section'].split('.')))
    for card in ordered_cards:
        grouped[card['chapter']].append(card)
        files[ROOT/'figures'/'techniques'/f'{card["id"]}.svg'] = diagram(card)
        coverage.append({
            'id': card['id'], 'title': card['title'],
            'principle': f'docs/{chapter_filename(card["chapter"])}#principle-{card["section"].replace(".", "-")}',
            'guide': f'docs/{chapter_filename(card["chapter"])}#{card["id"]}',
            'diagram': f'figures/techniques/{card["id"]}.svg',
            'code_sha256': hashlib.sha256(card['code'].encode()).hexdigest(),
            'run': f'python experiments/technique_lab.py --technique {card["id"]}',
            'source_references': card['refs'], 'scope': card['boundary'],
            'reading_resources': READINGS['techniques'][card['id']]['papers'] + READINGS['techniques'][card['id']]['readings'],
        })
    for chapter, cards in grouped.items():
        source_path = ROOT/'docs'/chapter_filename(chapter)
        merged = merged_chapter(chapter, cards, source_path.read_text(encoding='utf-8'))
        validate_markers(merged)
        displayed = re.findall(r'<!-- executable: ([a-z0-9_]+) -->', merged)
        if sorted(displayed) != sorted(card['id'] for card in cards):
            raise ValueError(f'Each technique must appear exactly once in chapter {chapter}')
        files[source_path] = merged

    index = ['# 技术速查：直接进入完整小节', '',
             f'本清单覆盖 **{len(CARDS)} 项技术、子方法与格式**。原理、推导、图示、代码、讲解与来源已经合并到主线章节，点击技术名称即可在同一处读完。'
             'OmniQuant 的 LWC/LET、NF4 与 Double Quantization 等仍分别设有入口。', '',
             '## 如何使用', '',
             '1. 连续学习时，从学习首页按章节阅读；每章提供目录与前后章导航。',
             '2. 按技术查找时，点击下表入口，依次阅读原理、图示和紧随其后的代码。',
             '3. 来源链接集中在各项的“论文、博客与官方资料”展开区；代码可用下列命令单独运行。', '',
             '```bash', 'python experiments/technique_lab.py --list',
             'python experiments/technique_lab.py --technique gptq',
             'python experiments/technique_lab.py --check-docs --write-results', '```', '',
             '运行结果见 [逐项技术实验记录](../experiments/results/逐项技术实验结果.md)，机器可读覆盖表见 [technique_coverage.json](../references/technique_coverage.json)。', '',
             '每项的论文、博客及官方说明也可集中查阅 [论文与博客导航](../references/论文与博客导航.md)。', '',
             '完成章节后可使用[分章测试与课程考核](11_练习与答案.md#course-assessment-guide)：主线、数学附录与 LLM.int8 专题共 19 套测试，答案附评分要点。', '',
             '未来判断与研究建议另见[第 16 章](16_未来趋势与研究路线.md)。其中的情景算例用于解释假设，不计入量化技术条目。', '',
             '需要完整操作流程时，进入[27 个逐章及专题实操 demo](../README.md#chapter-demos)：源码含中文逐步注释，各章节均提供运行、观察与修改建议。本索引页仅负责导航。', '',
             '## 覆盖清单', '', '| 技术：原理与实践 | 所在小节 | 运行 ID |', '|---|---|---|']
    for row, card in zip(coverage, ordered_cards):
        index.append(f'| [{card["title"]}]({row["guide"][5:]}) | {card["section"]} | `{card["id"]}` |')
    if READINGS.get('supplementary_topics'):
        index.extend(['', '## 补充技术与研究主题', '',
                      '以下主题直接并入原章节，以机制、公式和来源比较补齐技术覆盖。'
                      '它们未计入上述 93 项带独立图示的技术示例；数学反例由 `deep_dive_lab.py` 核验，新架构专题实操由 `chapter_demo_lab.py` 核验；原生探针另行显式执行。', '',
                      '| 技术或研究主题 | 讲解范围 |', '|---|---|'])
        for topic in READINGS['supplementary_topics']:
            index.append(f'| [{topic["title"]}]({topic["guide"][5:]}) | 原理、与相近路线的差别、验证与部署边界 |')
    index.extend(['', '## 复现与维护', '',
                  '这些代码是 CPU 教学实验，其中的完整算法、核心子问题和数值格式模拟均在对应“适用边界”中区分。'
                  'QDQ 数值数组不是物理低比特存储；只有明确展示字节打包的例子可以直接据其字节数讨论布局。', '',
                  '原理文字直接维护在主线章节；图示和可执行片段维护在 [technique_catalog.py](../experiments/technique_catalog.py)，'
                  '公共数值函数见 [technique_kernels.py](../experiments/technique_kernels.py) 和 [quant_lab.py](../experiments/quant_lab.py)。修改后执行：', '',
                  '```bash', 'python experiments/build_technique_docs.py',
                  'python experiments/technique_lab.py --check-docs --write-results', '```', '',
                  '论文、博客和官方文档链接统一维护在 [reading_links.json](../references/reading_links.json)，生成时检查全部技术的来源覆盖和资源类型。', '',
                  '生成器只刷新带标记的实践内容、目录和导航；共享小节中的 topic/practice 标记决定每种方法的插入位置。'
                  '编辑原理文字后重建，不会丢失手写推导，也不会恢复旧的分离图解页。', '',
                  '检查器会核对 Markdown 中实际显示的代码、图示、正文入口和覆盖表，避免“测试的是另一份代码”。'
                  '论文完整实现应继续查原始资料；教学例子不用于跨论文质量或性能排名。', '', '[返回学习目录](../README.md)', ''])
    files[ROOT/'docs'/'技术图解与代码索引.md'] = '\n'.join(index)
    files[ROOT/'references'/'technique_coverage.json'] = json.dumps(coverage, ensure_ascii=False, indent=2)+'\n'
    navigation = ['# 论文、博客与官方技术资料导航', '',
                  f'覆盖 **{len(CARDS)} 项技术、子方法与格式**。优先列原论文、作者博客、官方技术文章、作者项目和规范，并明确区分类型。', '',
                  f'最近增补日期：**{READINGS["checked_on"]}**。本次新论文另记录正文版本；早期资源保留其原核验时间，并非全部外链在同一天重查。'
                  '详情见 [外链核查记录](provenance/link_checks.json)；访问限制、跳转和已排除的失效地址也保留在记录中。', '',
                  '没有单独列出的作者博客时，提供作者 README 或官方文档。HQQ、GGUF 及部分 GGML 格式以技术材料或规范为主；基础算法可参考综述。'
                  '相关论文不自动等于某个教学例子的完整实现，具体对应关系见主线小节末尾的阅读说明。', '',
                  '[原始资料索引](原始资料索引.md) · [技术图解与代码](../docs/技术图解与代码索引.md) · '
                  '[未来预测的证据对照](../docs/16_未来趋势与研究路线.md#forecast-sources)', '']
    for chapter, cards in grouped.items():
        navigation.extend([f'## {chapter}｜{CHAPTERS[chapter]}', '',
                           '| 技术与代码入口 | 论文 | 博客 / 作者项目 / 官方说明 |', '|---|---|---|'])
        for card in cards:
            topic = READINGS['techniques'][card['id']]
            paper_links = '<br>'.join(reading_link(key) for key in topic['papers']) or '以技术材料或官方规范为主，见逐项说明'
            reading_links = '<br>'.join(f'{ {"blog":"博客", "project":"作者项目", "docs":"文档", "spec":"规范", "code":"源码"}[READINGS["resources"][key]["kind"]]}：{reading_link(key)}' for key in topic['readings'])
            if not reading_links:
                reading_links = '以所读论文正文与伪代码为主；本次未确认独立作者项目，见节内说明'
            guide = f'../docs/{chapter_filename(chapter)}#{card["id"]}'
            navigation.append(f'| [{card["title"]}]({guide}) | {paper_links} | {reading_links} |')
        navigation.append('')
    navigation.extend(['## 补充论文索引', '', 'P 编号补充基础方法、LoRA、QJL 与格式论文，不改变主线 R01–R36 的既有编号。', '',
                       '| 编号 | 论文 |', '|---|---|'])
    for resource_id in READINGS['resources']:
        if resource_id.startswith('P'):
            navigation.append(f'| {resource_id} | {reading_link(resource_id)} |')
    navigation.extend(['', '## 本次增补论文', '',
                       'N 编号对应[检索范围与纳入规则](原始资料索引.md#search-method)；后续补充主题见下表和既有研究来源清单。', '',
                       '| 编号 | 论文 | 首次提交 | 所读版本 |', '|---|---|---|---|'])
    for resource_id, resource in READINGS['resources'].items():
        if resource_id.startswith('N'):
            navigation.append(f'| {resource_id} | {reading_link(resource_id)} | {resource["first_submitted"]} | {resource["reviewed_version"]} |')
    if READINGS.get('supplementary_topics'):
        navigation.extend(['', '## 补充主题的原始来源', '',
                           '补充主题的章节入口、论文版本与作者项目列在下方。读取相关方法段落不等于完成模型或 GPU 复现。', '',
                           '| 主题与章节 | 原始来源 | 首次提交 / 所读版本 |', '|---|---|---|'])
        for topic in READINGS['supplementary_topics']:
            primary = READINGS['resources'][topic['resources'][0]]
            sources = '<br>'.join(reading_link(key) for key in topic['resources'])
            navigation.append(f'| [{topic["title"]}](../{topic["guide"]}) | {sources} | '
                              f'{primary["first_submitted"]} / {primary["reviewed_version"]} |')
    navigation.extend(['', '## 版本与阅读提示', '',
                       '- GPTQv2 的作者仓库已经迁移为 GPTAQ；导航采用当前项目入口。',
                       '- HQQ 原博客地址本次返回 404，未作为推荐入口；保留作者项目、官方文档及核心源码。',
                       '- AutoGPTQ 集成博客、早期 TensorRT 文章提供历史背景，安装方式与接口应以目标版本文档为准。',
                       '- FP8/FP4 文档固定在 Transformer Engine 2.8；NVFP4 预训练论文属于扩展阅读，训练配方和推理格式示例需要区分。',
                       '- 出版社 DOI 可能需要机构权限，网络限制也可能影响官方博客；访问失败不自动代表文献不存在。', '',
                       '[返回学习目录](../README.md)', ''])
    files[ROOT/'references'/'论文与博客导航.md'] = '\n'.join(navigation)
    # 章节实操在原理/技术示例排版完成后插入，避免生成区域互相嵌套。
    from experiments.build_chapter_demos import add_outputs
    add_outputs(files)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check generated content without writing')
    args = parser.parse_args()
    files = outputs()
    mismatches = []
    for path, content in files.items():
        if args.check:
            if not path.exists() or path.read_text(encoding='utf-8') != content:
                mismatches.append(str(path.relative_to(ROOT)))
        else:
            write_text(path, content)
    if mismatches:
        raise SystemExit('Generated content differs: '+', '.join(mismatches))
    print(f'{"CHECK" if args.check else "BUILD"}: {len(CARDS)} techniques in {len(CHAPTERS)} merged chapters, {len(CARDS)} SVG diagrams.')


if __name__ == '__main__':
    main()

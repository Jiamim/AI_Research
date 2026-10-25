#!/usr/bin/env python3
"""Execute the exact NumPy snippets embedded in the merged main chapters."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.technique_catalog import CARDS, CHAPTERS
from experiments.result_utils import PUBLISHED_RESULTS, file_link, run_metadata, write_json, write_text

# E01–E03仍是原三档题；新增架构主题沿用E编号、单独计分。
GPU_EXTENSION_DIFFICULTIES = {'09': ['困难'], '13': ['困难', '特别困难'],
                              '14': ['困难', '特别困难', '困难'], '15': ['困难']}

def validate_assessment(text, key):
    """Validate the authored course contract without generating another question bank.

    This checks structure and grading metadata, not the mathematical truth of answers.
    """
    if text.count('<!-- chapter-assessment -->') != 1 or text.count('<!-- /chapter-assessment -->') != 1:
        raise ValueError(f'{key}: expected exactly one complete assessment region')
    start = text.index('<!-- chapter-assessment -->')
    end = text.index('<!-- /chapter-assessment -->')
    if end < start:
        raise ValueError(f'{key}: assessment markers are reversed')
    region = text[start:end]
    expected_core = 10 if key in ('12', '13', '14', '15') else 8
    expected_ids = [f'{key}-{i:02d}' for i in range(1, expected_core + 1)]
    extension_difficulties = ['简单', '困难', '特别困难', *GPU_EXTENSION_DIFFICULTIES.get(key, [])]
    expected_ids += [f'{key}-E{i:02d}' for i in range(1, len(extension_difficulties)+1)]
    questions = re.split(r'^### ', region, flags=re.M)[1:]
    if len(questions) != len(expected_ids):
        raise ValueError(f'{key}: missing or extra assessment questions')
    total = 0
    for expected, block in zip(expected_ids, questions):
        heading = block.splitlines()[0]
        match = re.fullmatch(r'([^｜]+)｜.+（(简单|困难|特别困难)｜.+）', heading)
        if not match or match[1] != expected:
            raise ValueError(f'{key}: question ID or difficulty is invalid: {heading}')
        if region.count(f'id="exercise-{expected.lower()}"') != 1:
            raise ValueError(f'{key}: question anchor must be unique: {expected}')
        if '-E' in expected:
            if match[2] != extension_difficulties[int(expected[-2:]) - 1] or '单独计分' not in heading:
                raise ValueError(f'{key}: extension difficulty or grading is invalid')
        else:
            points = re.search(r'，(\d+) 分）$', heading)
            if not points or int(points[1]) <= 0:
                raise ValueError(f'{key}: missing positive core score')
            total += int(points[1])
        answers = re.findall(r'<details><summary>参考答案与评分要点</summary>\s*(.*?)\s*</details>', block, re.S)
        if len(answers) != 1 or '<details' in answers[0] or '**评分要点：**' not in answers[0]:
            raise ValueError(f'{key}: missing, nested or incomplete answer: {expected}')
        answer = answers[0].split('**评分要点：**', 1)[0]
        if len(re.findall(r'^\*\*[^*]+[：。]\*\*', answer, re.M)) < 3:
            raise ValueError(f'{key}: fewer than three reference paths: {expected}')
    if total != 100:
        raise ValueError(f'{key}: core scores must sum to 100, got {total}')
    if key == '11':
        review = re.findall(r'^## 题 (\d+)：[^\n]+（(?:简单|困难|特别困难)）\n(.*?)(?=^## |\Z)', text, re.M | re.S)
        if [int(number) for number, _ in review] != list(range(1, 27)):
            raise ValueError('11: expected all 26 review questions with difficulty labels')
        for number, body in review:
            answer = re.findall(r'<details><summary>答案</summary>\s*(.*?)\s*</details>', body, re.S)
            if len(answer) != 1 or len(re.findall(r'^\*\*[^*]+[：。]\*\*', answer[0], re.M)) < 3:
                raise ValueError(f'11: review question {number} needs three reference paths')
    return len(questions) + (26 if key == '11' else 0)


def validate_docs():
    from experiments.build_technique_docs import outputs
    for path, expected in outputs().items():
        if not path.exists() or path.read_text(encoding='utf-8') != expected:
            raise AssertionError(f'Stale generated content: {path.relative_to(ROOT)}')
    checked = 0
    for card in CARDS:
        path = ROOT/'docs'/f'{card["chapter"]}_{CHAPTERS[card["chapter"]]}.md'
        pattern = rf'<!-- executable: {re.escape(card["id"])} -->\n```python\n(.*?)\n```'
        matches = re.findall(pattern, path.read_text(encoding='utf-8'), re.S)
        if matches != [card['code']]:
            raise AssertionError(f'Displayed code differs from executed code: {card["id"]}')
        tree = ET.parse(ROOT/'figures'/'techniques'/f'{card["id"]}.svg')
        if tree.getroot().tag != '{http://www.w3.org/2000/svg}svg':
            raise AssertionError(f'Invalid SVG: {card["id"]}')
        checked += 1
    displayed_ids = []
    for path in (ROOT/'docs').rglob('*.md'):
        displayed_ids.extend(re.findall(r'<!-- executable: ([a-z0-9_]+) -->', path.read_text(encoding='utf-8')))
    if sorted(displayed_ids) != sorted(card['id'] for card in CARDS):
        raise AssertionError('Every technique must appear exactly once in the merged documentation.')
    course_paths = sorted((ROOT/'docs').glob('[0-9][0-9]_*.md'))
    if [p.name[:2] for p in course_paths] != [f'{i:02d}' for i in range(17)]:
        raise AssertionError('Expected main course chapters 00 through 16')
    assessments = [(p, p.name[:2]) for p in course_paths] + [
        (ROOT/'docs'/'附录A_推导所需的数学.md', 'A'),
        (ROOT/'docs'/'专题_LLM_int8_原理与实测.md', 'INT8'),
    ]
    question_count = sum(validate_assessment(p.read_text(encoding='utf-8'), key)
                         for p, key in assessments)
    if question_count != 250:
        raise AssertionError('Course question count differs from the documented 250')
    # Resolve all actual Markdown file links and explicit anchor links.
    markdown = sorted(ROOT.glob('*.md'))
    for directory in ('docs', 'references', 'experiments'):
        markdown.extend((ROOT/directory).rglob('*.md'))
    # Runnable notebooks move with their local documentation links; historical
    # notebooks remain immutable snapshots and are verified by archive hashes.
    markdown.extend((ROOT/'experiments'/'notebooks').glob('*.ipynb'))
    for path in markdown:
        text = path.read_text(encoding='utf-8')
        if path.suffix == '.ipynb':
            text = '\n'.join(''.join(cell['source']) for cell in json.loads(text)['cells']
                             if cell['cell_type'] == 'markdown')
        anchors = re.findall(r'<a id="([^"]+)"', text)
        if len(anchors) != len(set(anchors)):
            raise AssertionError(f'Duplicate Markdown anchor in {path}')
        for destination in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
            if '://' in destination:
                continue
            relative, _, anchor = destination.partition('#')
            relative = unquote(relative)
            target = path.parent/relative if relative else path
            if not target.exists():
                raise AssertionError(f'Broken link in {path}: {destination}')
            if anchor and target.suffix == '.md':
                target_text = target.read_text(encoding='utf-8')
                if f'id="{anchor}"' not in target_text:
                    raise AssertionError(f'Missing explicit anchor in {path}: {destination}')
    return checked


def execute(card):
    if not __debug__:
        raise RuntimeError('Run without -O: tutorial assertions must remain enabled.')
    output = io.StringIO()
    tree = ast.parse(card['code'])
    assertions = sum(isinstance(node, ast.Assert) for node in ast.walk(tree))
    if not assertions:
        raise AssertionError(f'No numerical assertion in {card["id"]}')
    with contextlib.redirect_stdout(output):
        exec(compile(tree, f'<tutorial:{card["id"]}>', 'exec'), {'__name__': '__main__'})
    return dict(id=card['id'], title=card['title'], passed=True,
                assertion_locations=assertions,
                code_sha256=hashlib.sha256(card['code'].encode()).hexdigest(),
                output=output.getvalue().strip(), scope=card['boundary'])


def main():
    if not __debug__:
        raise SystemExit('Run without -O: tutorial assertions must remain enabled.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--technique', choices=['all']+[c['id'] for c in CARDS], default='all')
    parser.add_argument('--check-docs', action='store_true')
    parser.add_argument('--write-results', action='store_true')
    parser.add_argument('--output-dir', type=Path, help='Result directory (requires --write-results)')
    args = parser.parse_args()
    if args.list:
        for card in CARDS:
            print(f'{card["id"]:22s} {card["title"]}')
        return
    if args.write_results and args.technique != 'all':
        parser.error('--write-results requires all techniques to preserve the complete record')
    if args.output_dir is not None and not args.write_results:
        parser.error('--output-dir requires --write-results')
    # Publishing a report always verifies the displayed code first.
    checked = validate_docs() if args.check_docs or args.write_results else 0
    selected = [c for c in CARDS if args.technique == 'all' or c['id'] == args.technique]
    results = []
    for card in selected:
        try:
            result = execute(card)
        except Exception as exc:
            raise RuntimeError(f'Technique {card["id"]} failed') from exc
        results.append(result)
        print(f'PASS {card["id"]}')
        if len(selected) == 1:
            print(result['output'])
    print(f'PASS: {len(results)} executable techniques; {checked} displayed-code/diagram checks.')
    if args.write_results:
        output_dir = args.output_dir.resolve() if args.output_dir is not None else PUBLISHED_RESULTS
        payload = dict(**run_metadata({'technique': args.technique, 'check_docs': True},
                                     ['references/reading_links.json', *[f'docs/{key}_{title}.md' for key, title in CHAPTERS.items()]]),
                       numpy=np.__version__,
                       scope='Synthetic CPU demonstrations, not complete paper or hardware reproductions',
                       passed_techniques=len(results), checked_docs=checked, results=results)
        write_json(output_dir/'technique_results.json', payload)
        command = 'python experiments/technique_lab.py --check-docs --write-results'
        lines = ['# 逐项技术实验记录', '', f'运行命令：`{command}`。', '',
                 f'Python {platform.python_version()}；NumPy {np.__version__}；各示例独立固定随机种子。', '',
                 f'**{len(results)} 项技术示例全部通过**；本次核对了 {checked} 项 Markdown 实际代码与 SVG 图示。', '',
                 '本记录是 CPU 数学与数据流实验。QAT、量化算法及格式示例的简化范围见逐项说明；没有运行真实 LLM 或测量 GPU 加速。', '',
                 '运行时先检查合并后的主线章节，再执行与 Markdown 完全一致的片段。每项代码只保留一个正文位置。断言包括代数恒等式、概率误差、状态顺序、存储布局和格式边界；不代表完整论文已复现。', '',
                 file_link('返回技术索引', ROOT/'docs'/'技术图解与代码索引.md', output_dir) + ' · [机器可读结果](technique_results.json)', '',
                 '运行时间、参数、公共函数和执行器的源码指纹见机器可读结果。', '']
        for result in results:
            lines.extend([f'## {result["id"]}｜{result["title"]}', '', '状态：通过。', '',
                          '```text', result['output'], '```', '', f'范围：{result["scope"]}', ''])
        write_text(output_dir/'逐项技术实验结果.md', '\n'.join(lines))


if __name__ == '__main__':
    main()

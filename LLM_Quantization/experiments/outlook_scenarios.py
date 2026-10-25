#!/usr/bin/env python3
"""Run the exact hypothetical scenario snippets in chapter 16.

These checks validate arithmetic and stated mathematical conditions only.
They do not test future adoption, real-model quality, or hardware performance.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import platform
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.result_utils import PUBLISHED_RESULTS, file_link, run_metadata, write_json, write_text
CHAPTER = ROOT/'docs'/'16_未来趋势与研究路线.md'
TITLES = {
    'amortization': '离线成本摊销',
    'decision_margin': '决策间隔与误差方向',
    'kv_budget': '包含尺度与残留区的 KV 预算',
    'successful_task_cost': '单次尝试与成功任务成本',
}


def run_cases(case='all'):
    if not __debug__:
        raise RuntimeError('Run without -O so numerical assertions remain enabled.')
    if case not in ('all', *TITLES):
        raise ValueError(f'Unknown scenario: {case}')
    snippets = re.findall(r'<!-- outlook-example: ([a-z_]+) -->\n```python\n(.*?)\n```',
                          CHAPTER.read_text(encoding='utf-8'), re.S)
    if len(snippets) != len(TITLES) or {key for key, _ in snippets} != set(TITLES):
        raise ValueError('Each outlook scenario must occur exactly once in the chapter.')
    records = []
    for key, code in snippets:
        if case != 'all' and key != case:
            continue
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(code, f'<outlook:{key}>', 'exec'), {'__name__': '__main__'})
        records.append(dict(id=key, title=TITLES[key], passed=True,
                            code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                            output=output.getvalue().strip()))
    return records


def write_results(records, output_dir=None):
    output_dir = Path(output_dir).resolve() if output_dir is not None else PUBLISHED_RESULTS
    payload = dict(**run_metadata({'case': 'all'}, ['docs/16_未来趋势与研究路线.md']),
                   scope='Hypothetical arithmetic and mathematical demonstrations; not forecast or benchmark validation.',
                   numpy=np.__version__, passed_scenarios=len(records), results=records)
    write_json(output_dir/'outlook_results.json', payload)
    lines = ['# 未来趋势：情景算例记录', '',
             '运行命令：`python experiments/outlook_scenarios.py --write-results`。', '',
             f'Python {platform.python_version()}；NumPy {np.__version__}。共 {len(records)} 个算例通过。', '',
             '**验证范围：**检查文档中假设数据的算术、决策间隔条件和容量约束，不证明趋势预测正确，也不代表真实 LLM 或硬件 benchmark。', '',
             '代码直接从'+file_link('第 16 章', CHAPTER, output_dir)+'读取，避免文档与执行片段不一致。'
             '这些算例与已有 93 项量化技术示例分开计数。', '',
             file_link('交互式成本计算器', ROOT/'figures'/'outlook_cost_calculator.html', output_dir)+' · [机器可读记录](outlook_results.json)', '',
             '运行时间、参数与完整源码指纹见机器可读记录。', '']
    for record in records:
        lines.extend([f'## {record["title"]}', '', '状态：通过。', '', '```text', record['output'], '```', ''])
    write_text(output_dir/'未来趋势_情景算例.md', '\n'.join(lines))


def main():
    if not __debug__:
        raise SystemExit('Run without -O so numerical assertions remain enabled.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all', *TITLES], default='all')
    parser.add_argument('--write-results', action='store_true')
    parser.add_argument('--output-dir', type=Path, help='Result directory (requires --write-results)')
    args = parser.parse_args()
    if args.write_results and args.case != 'all':
        parser.error('--write-results requires all cases to preserve the complete record')
    if args.output_dir is not None and not args.write_results:
        parser.error('--output-dir requires --write-results')
    records = run_cases(args.case)
    for record in records:
        print(f'PASS {record["id"]}\n{record["output"]}')
    if args.write_results:
        write_results(records, args.output_dir)
    print(f'PASS: {len(records)} hypothetical scenarios; no forecast accuracy claim.')


if __name__ == '__main__':
    main()

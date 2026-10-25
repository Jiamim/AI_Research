#!/usr/bin/env python3
"""Run the supplemental derivations embedded in docs, without GPU or downloads."""

import argparse
import ast
import contextlib
import hashlib
import io
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.result_utils import PUBLISHED_RESULTS, run_metadata, write_json

CASES = {
    'transformed_hessian': '坐标变换与 Hessian/damping',
    'segmented_attention': '不同缓存分段的 softmax 合并',
    'fp4_scale_error': 'FP4 码字与两级尺度误差',
    'residual_cancellation': '精确残差不保证完整输出误差最小',
    'truncated_kl': '截断 KL 与条件分布 KL',
    'variance_equalization': '方差均衡需要同构代价',
    'moe_error_proxy': 'MoE 误差交叉项与局部代理',
    'polar_error_geometry': '极坐标范数与方向误差',
    'rope_scale_commutation': 'RoPE 与配对尺度可交换条件',
    'dual_axis_scaling': '双轴尺度的等价计算与固定码字拟合',
    'discrete_code_update': '离散码字更新的步长限制',
    'bitplane_lookup': '位平面与激活查表的点积等价性',
    'ternary_support_fit': '三值支持集合与尺度联合拟合',
    'recurrent_update_loss': '重复量化导致的小更新丢失',
    'integer_requantization': '整数偏置、输出重定标与残差尺度',
    'lowbit_packing': '二比特补码、尾部填充与字节往返',
    'bias_correction': '输出均值修正与中心化误差',
    'adapter_merge_base': 'QLoRA 合并时的基础权重差异',
    'calibration_weighting': 'padding 掩码与校准样本权重',
    'serving_goodput': '请求时序、吞吐与延迟达标率',
}


def snippets():
    found = {}
    for path in sorted((ROOT/'docs').glob('*.md')):
        for key, code in re.findall(r'<!-- depth-example: ([a-z0-9_]+) -->\n```python\n(.*?)\n```',
                                    path.read_text(encoding='utf-8'), re.S):
            if key in found:
                raise ValueError(f'Duplicate derivation: {key}')
            found[key] = (path, code)
    if set(found) != set(CASES):
        raise ValueError('Documented derivations differ from the declared cases')
    return found


def run_cases(case='all'):
    if not __debug__:
        raise RuntimeError('Run without -O to keep numerical assertions enabled')
    if case != 'all' and case not in CASES:
        raise ValueError(f'Unknown derivation: {case}')
    results = []
    found = snippets()
    for key in CASES:
        if case not in ('all', key):
            continue
        path, code = found[key]
        tree = ast.parse(code)
        if not any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
            raise ValueError(f'Derivation has no numerical check: {key}')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(tree, str(path), 'exec'), {'__name__': '__main__'})
        results.append(dict(id=key, title=CASES[key], document=str(path.relative_to(ROOT)),
                            code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                            passed=True, output=output.getvalue().strip()))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all', *CASES], default='all')
    parser.add_argument('--write-results', action='store_true', help='Save numerical results as JSON')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.output_dir is not None and not args.write_results:
        parser.error('--output-dir requires --write-results')
    if args.write_results and args.case != 'all':
        parser.error('--write-results requires all derivations')
    records = run_cases(args.case)
    for row in records:
        print(f'PASS {row["id"]}: {row["output"]}')
    if args.write_results:
        output = args.output_dir or PUBLISHED_RESULTS
        payload = {**run_metadata({'case': 'all'}, {row['document'] for row in records}),
                   'scope': 'Supplemental algebra and counterexamples, not new algorithms or model benchmarks',
                   'passed_derivations': len(records), 'results': records}
        write_json(output/'deep_dive_results.json', payload)


if __name__ == '__main__':
    main()

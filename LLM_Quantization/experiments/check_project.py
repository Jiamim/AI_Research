#!/usr/bin/env python3
"""Read-only project checks. Use --notebook-cpu for optional PyTorch checks."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_notebook():
    import nbformat
    path = ROOT/'experiments'/'notebooks'/'colab_qwen3_quantization.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    validation = notebook.metadata.get('quantization_validation', {})
    if validation.get('gpu_benchmark') == 'not_run_for_current_source':
        if any(cell.get('outputs') or cell.get('execution_count') is not None
               for cell in notebook.cells if cell.cell_type == 'code'):
            raise AssertionError('Unexecuted current notebook must not display historical outputs.')
    code = []
    protocol = None
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            source = '\n'.join(line for line in cell.source.splitlines()
                               if not line.lstrip().startswith(('%', '!')))
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == 'PROTOCOL_VERSION' for t in node.targets):
                    protocol = ast.literal_eval(node.value)
            code.append(source)
    if protocol != validation.get('protocol_version'):
        raise AssertionError('Notebook code and validation metadata disagree on protocol version.')
    evidence_path = ROOT/'experiments'/'archive'/'colab_t4_2026-10-15'/'evidence_from_notebook.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    archive = evidence_path.parent/evidence['source']
    if (path.parent/validation['historical_notebook']).resolve() != archive.resolve():
        raise AssertionError('Notebook historical reference differs from the evidence archive.')
    if hashlib.sha256(archive.read_bytes()).hexdigest() != evidence['source_sha256_at_analysis']:
        raise AssertionError('Historical notebook differs from the analyzed snapshot.')
    print(f'PASS notebook schema and syntax: {len(code)} code cells', flush=True)
    return code


def main():
    if not __debug__:
        raise SystemExit('Run without -O: numerical assertions must stay enabled.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook-cpu', action='store_true', help='Also run notebook numerical tests (requires torch)')
    args = parser.parse_args()
    from experiments.quant_lab import run
    from experiments.technique_lab import validate_docs, execute
    from experiments.technique_catalog import CARDS
    from experiments.outlook_scenarios import run_cases
    docs = validate_docs()
    metrics, checks, _ = run()
    for card in CARDS:
        execute(card)
    scenarios = run_cases()
    from experiments.llm_int8_lab import run_experiments
    int8_results = run_experiments()
    from experiments.deep_dive_lab import run_cases as run_derivations
    derivations = run_derivations()
    from experiments.chapter_demo_lab import run_all as run_chapter_demos
    chapter_demos = run_chapter_demos()
    print(f'PASS {docs} documented techniques, {len(metrics)} baseline experiments, '
          f'{len(checks)} numerical checks, {len(scenarios)} scenarios; '
          f'LLM.int8: {len(int8_results["ablation"])} ablations / '
          f'{len(int8_results["threshold_sweep"])} thresholds; '
          f'{len(derivations)} supplemental derivations; '
          f'{len(chapter_demos)} annotated chapter demos', flush=True)
    check_notebook()
    from experiments.build_colab_chapter_demos import validate_notebook as check_chapter_notebook
    check_chapter_notebook()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_regressions.py')
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_llm_int8_lab.py'))
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_run_artifacts.py'))
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_chapter_demos.py'))
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_demo_torch_kernels.py'))
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_cuda_lab.py'))
    if args.notebook_cpu:
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_notebook_cpu.py'))
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
    print('PASS project checks; published files were not updated.', flush=True)


if __name__ == '__main__':
    main()

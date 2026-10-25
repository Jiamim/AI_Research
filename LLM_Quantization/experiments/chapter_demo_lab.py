#!/usr/bin/env python3
"""运行逐章实操；默认只打印，可显式保存带源码哈希的结果。"""
import argparse
import ast
import contextlib
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import re
import sys
import traceback
import uuid

# 从任意工作目录调用此入口时，都让同项目的数值helper可以被导入。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.chapter_demo_catalog import DEMOS
from experiments.result_utils import PUBLISHED_RESULTS, run_metadata, write_json


def validate_docs():
    """实际展示的代码必须等于执行的源码，包括逐步中文注释。"""
    for demo in DEMOS:
        text = (ROOT/demo['document']).read_text(encoding='utf-8')
        pattern = rf'<!-- chapter-demo-code: {demo["id"]} -->\n```python\n(.*?)\n```'
        matches = re.findall(pattern, text, re.S)
        code = (ROOT/demo['source']).read_text(encoding='utf-8').strip()
        if matches != [code]:
            raise ValueError(f'章节展示与源码不一致: {demo["id"]}；练习修改可加 --practice；'
                             '发布前运行 python experiments/build_technique_docs.py')
    return len(DEMOS)


def execute(demo, device='auto'):
    """隔离全局变量并捕获标准输出；执行的是本地可信教学脚本。"""
    if not __debug__:
        raise RuntimeError('请不要使用 -O，实操需要保留数值断言。')
    path = ROOT/demo['source']
    code = path.read_text(encoding='utf-8')
    tree = ast.parse(code)
    if not any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
        raise ValueError(f'实操缺少可核验条件: {demo["id"]}')
    from experiments.demo_runtime import DemoRuntime, use_runtime
    import torch
    runtime = DemoRuntime(device)
    namespace = {'__name__': '__main__', '__file__': str(path)}
    output = io.StringIO()
    try:
        with use_runtime(runtime), contextlib.redirect_stdout(output):
            exec(compile(tree, str(path), 'exec'), namespace)
            runtime.synchronize()
    except BaseException as exc:
        # 保留失败前输出，也保留原异常类型，便于定位断言或CUDA错误。
        exc.demo_output = output.getvalue()
        raise
    # 审核实际数值结果的设备，避免仅创建一个CUDA标记却仍用CPU计算。
    tensors = []
    seen = set()
    def collect(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
    for value in namespace.values():
        collect(value)
    if not tensors or any(t.device != runtime.device for t in tensors):
        error = AssertionError('demo数值张量未全部位于指定设备')
        error.demo_output = output.getvalue()
        raise error
    return dict(id=demo['id'], title=demo['title'], source=demo['source'],
                document=demo['document'], passed=True,
                code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                output=output.getvalue().strip(), scope=demo['boundary'],
                runtime=runtime.metadata(), checked_tensor_count=len(tensors))


def run_all(device='cpu', *, check_docs=True):
    # 检查文档后再运行，防止“演示一份代码、测试另一份代码”。
    if check_docs:
        validate_docs()
    return [execute(demo, device=device) for demo in DEMOS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chapter', choices=['all', *[d['id'] for d in DEMOS]], default='all')
    parser.add_argument('--list', action='store_true', help='列出章节实操入口')
    parser.add_argument('--device', default='auto', help='auto优先GPU；cpu、cuda、cuda:N。请求cuda失败时报错')
    parser.add_argument('--practice', action='store_true', help='运行修改后的练习源码；跳过正文一致性检查，保留数值断言与设备核验')
    parser.add_argument('--write-results', action='store_true', help='显式保存JSON；默认不写文件')
    parser.add_argument('--output-dir', type=Path, help='个人结果目录，须与--write-results一起使用')
    args = parser.parse_args()
    if args.output_dir is not None and not args.write_results:
        parser.error('--output-dir requires --write-results')
    if args.write_results and args.chapter != 'all' and args.output_dir is None:
        parser.error('保存单章请指定 --output-dir，避免覆盖发布的整套运行记录')
    if args.practice and args.write_results and args.output_dir is None:
        parser.error('练习结果请指定 --output-dir，不能覆盖发布记录')
    if args.list:
        for demo in DEMOS:
            print(demo['id'], demo['title'])
        return
    selected = [d for d in DEMOS if args.chapter in ('all', d['id'])]
    sources = [d[key] for d in DEMOS for key in ('source', 'document')]
    payload = {**run_metadata({'chapter': args.chapter, 'requested_device': args.device,
                              'practice': args.practice}, sources),
               'schema_version': 3, 'run_id': uuid.uuid4().hex,
               'scope': 'PyTorch CPU/CUDA chapter exercises; synthetic data, not full LLM or native low-bit kernel benchmarks',
               'status': 'running', 'selected_demos': [d['id'] for d in selected],
               'passed_demos': 0, 'checked_docs': 0, 'results': [],
               'document_validation': 'skipped_practice' if args.practice else 'pending',
               'active_demo': None, 'error': None}
    directory = args.output_dir if args.output_dir is not None else PUBLISHED_RESULTS
    # 个人目录保留每轮记录；根目录同名文件始终表示最近一次尝试，包括失败。
    archive = directory/'runs'/payload['run_id'] if args.output_dir is not None else None

    def save():
        payload['updated_utc'] = datetime.now(timezone.utc).isoformat()
        if args.write_results:
            if archive is not None:
                write_json(archive/'chapter_demo_results.json', payload)
            write_json(directory/'chapter_demo_results.json', payload)

    save()  # 在文档校验与执行前覆盖最近状态，不能遗留旧的成功记录冒充本轮。
    try:
        if args.write_results and archive is not None:
            # 保存本轮源码与正文，Colab下载后可以重现修改及检查来源。
            for name in payload['source_files_sha256']:
                target = archive/'sources'/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT/name).read_bytes())
        if not args.practice:
            payload['checked_docs'] = validate_docs()
            payload['document_validation'] = 'passed'
        else:
            print('练习模式：正文一致性未检查；仍执行数值断言和设备核验。')
        for demo in selected:
            payload['active_demo'] = demo['id']
            save()
            result = execute(demo, device=args.device)
            payload['results'].append(result)
            payload['passed_demos'] = len(payload['results'])
            payload['status'] = 'partial'
            save()
            print(f'PASS {demo["id"]}｜{demo["title"]}｜{result["runtime"]["device"]}')
            if len(selected) == 1:
                print(result['output'])
    except BaseException as exc:
        payload['status'] = 'failed'
        if payload['document_validation'] == 'pending':
            payload['document_validation'] = 'failed'
        payload['error'] = dict(type=type(exc).__name__, message=str(exc),
                                demo=payload['active_demo'], output=getattr(exc, 'demo_output', ''),
                                traceback=traceback.format_exc())
        save()
        raise
    payload['active_demo'] = None
    payload['status'] = 'completed'
    save()
    print(f'PASS {payload["passed_demos"]} 个实操；正文校验：{payload["document_validation"]}。')
    if args.write_results:
        print('结果：', directory/'chapter_demo_results.json', '；run_id:', payload['run_id'])


if __name__ == '__main__':
    main()

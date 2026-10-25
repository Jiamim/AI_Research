"""章节生成与入口的行为检查；数学正确性由各实操独立断言核验。"""
import json
import ast
import base64
import contextlib
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from types import ModuleType
from unittest.mock import patch

from experiments import build_chapter_demos, chapter_demo_lab
from experiments.chapter_demo_catalog import DEMOS

ROOT = Path(__file__).resolve().parents[1]


class ChapterDemos(unittest.TestCase):
    def test_mr_gptq_detects_wrong_sign_and_inverse_update(self):
        demo = next(d for d in DEMOS if d['id'] == 'mr_gptq')
        original = Path.read_text
        mutations = [
            ('current[:, column:] += epsilon', 'current[:, column:] -= epsilon'),
            ('inverse[1:, 1:]-torch.outer', 'inverse[1:, 1:]+torch.outer'),
        ]
        for before, after in mutations:
            def modified(path, *args, **kwargs):
                text = original(path, *args, **kwargs)
                return text.replace(before, after) if path.name == 'mr_gptq.py' else text
            with self.subTest(mutation=after), patch.object(Path, 'read_text', modified):
                with self.assertRaises(AssertionError):
                    chapter_demo_lab.execute(demo, device='cpu')

    def test_native_metadata_uses_actual_tensor_types(self):
        import torch
        from experiments import native_nvfp4_probe as probe
        data = probe.tensor_metadata(input=torch.ones(2, 4, dtype=torch.bfloat16),
                                     loss=torch.ones((), dtype=torch.float32))
        self.assertEqual(data['input'], dict(dtype='torch.bfloat16', shape=[2, 4], device='cpu'))
        self.assertEqual(data['loss']['dtype'], 'torch.float32')
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, 'argv', ['native', '--output-dir', directory]):
                with patch.object(probe, 'run', side_effect=RuntimeError('unsupported hardware')):
                    with self.assertRaises(RuntimeError):
                        probe.main()
            failure = json.loads((Path(directory)/'native_nvfp4_probe.json').read_text())
            self.assertEqual(failure['status'], 'failed')
            self.assertEqual(failure['error']['message'], 'unsupported hardware')

    def test_failure_report_preserves_progress_and_previous_run(self):
        original = chapter_demo_lab.execute
        with tempfile.TemporaryDirectory() as directory:
            args = ['chapter_demo_lab.py', '--chapter', '01', '--device', 'cpu',
                    '--write-results', '--output-dir', directory]
            with patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()):
                chapter_demo_lab.main()
            path = Path(directory)/'chapter_demo_results.json'
            first = json.loads(path.read_text())
            read_text = Path.read_text
            def broken_source(path, *args, **kwargs):
                text = read_text(path, *args, **kwargs)
                if path.name == 'ch02.py':
                    text += "\nprint('failure output marker')\nraise RuntimeError('injected third chapter failure')\n"
                return text
            def fail(demo, device):
                if demo['id'] == '02':
                    with patch.object(Path, 'read_text', broken_source):
                        return original(demo, device)
                return original(demo, device)
            args[2] = 'all'
            with patch.object(sys, 'argv', args), patch.object(chapter_demo_lab, 'execute', fail):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                    chapter_demo_lab.main()
            failed = json.loads(path.read_text())
            self.assertEqual(failed['status'], 'failed')
            self.assertEqual(failed['passed_demos'], 2)
            self.assertEqual(failed['error']['demo'], '02')
            self.assertIn('injected third', failed['error']['traceback'])
            self.assertIn('failure output marker', failed['error']['output'])
            self.assertNotEqual(first['run_id'], failed['run_id'])
            old = Path(directory)/'runs'/first['run_id']/'chapter_demo_results.json'
            self.assertEqual(json.loads(old.read_text()), first)
            # 校验阶段失败也须刷新本轮状态，不能继续显示上一轮的成功或部分结果。
            with patch.object(sys, 'argv', args), patch.object(chapter_demo_lab, 'validate_docs', side_effect=ValueError('drift')):
                with self.assertRaises(ValueError):
                    chapter_demo_lab.main()
            invalid = json.loads(path.read_text())
            self.assertEqual(invalid['document_validation'], 'failed')
            self.assertEqual(invalid['passed_demos'], 0)

    def test_colab_extract_edit_run_and_download(self):
        # 从生成的真实ZIP启动，修改参数；验证不依赖仓库外的重建配置。
        from experiments.build_colab_chapter_demos import notebook_text
        notebook = json.loads(notebook_text())
        codes = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        assignment = next(n for n in ast.parse(codes[0]).body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'PACKAGE_B64' for t in n.targets))
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(assignment.value)))) as archive:
                archive.extractall(project)
            source = project/'experiments/demos/ch03.py'
            source.write_text(source.read_text().replace('test[:, 0] *= 12', 'test[:, 0] *= 15'))
            strict = subprocess.run([sys.executable, '-B', 'experiments/chapter_demo_lab.py',
                                     '--chapter', '03', '--device', 'cpu'], cwd=project, capture_output=True, text=True)
            self.assertNotEqual(strict.returncode, 0)
            self.assertIn('--practice', strict.stderr)
            code = next(c for c in codes if "CHAPTER = 'all'" in c)
            code = code.replace("CHAPTER = 'all'", "CHAPTER = '03'").replace("DEVICE = 'cuda'", "DEVICE = 'cpu'")
            code = code.replace('PRACTICE = False', 'PRACTICE = True')
            env = dict(project=project, Path=Path, tempfile=tempfile, sys=sys, subprocess=subprocess, json=json)
            previous = Path.cwd()
            try:
                os.chdir(project)
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(code, '<colab-run>', 'exec'), env)
            finally:
                os.chdir(previous)
            report = env['report']
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['document_validation'], 'skipped_practice')
            self.assertEqual(report['results'][0]['code_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
            snapshot = env['RESULT_DIR']/'runs'/report['run_id']/'sources'/source.relative_to(project)
            self.assertEqual(snapshot.read_bytes(), source.read_bytes())
            # 用替代下载接收端执行Notebook实际下载单元，检查原生结果和日志都进入ZIP。
            (env['RESULT_DIR']/'native_nvfp4_probe.json').write_text('{"status":"failed"}')
            google, colab = ModuleType('google'), ModuleType('google.colab')
            downloads = []
            files = ModuleType('files'); files.download = downloads.append
            colab.files = files; google.colab = colab
            download = next(c for c in codes if 'files.download' in c)
            with patch.dict(sys.modules, {'google': google, 'google.colab': colab}):
                exec(compile(download, '<colab-download>', 'exec'), env)
            with zipfile.ZipFile(downloads[0]) as archive:
                self.assertIn('native_nvfp4_probe.json', archive.namelist())
                self.assertIn('chapter_execution.log', archive.namelist())
                self.assertEqual(json.loads(archive.read('chapter_demo_results.json'))['run_id'], report['run_id'])

    def test_sibling_topic_regions_preserve_manual_slots(self):
        demos = [d for d in DEMOS if d['document'].startswith('docs/13_')]
        slots = '\n'.join(f'<!-- demo-slot: {d["id"]} -->' for d in demos if d.get('slot'))
        source = '手写开头\n'+slots+'\n<!-- chapter-assessment -->\n手写结尾'
        merged = source
        for demo in demos:
            merged = build_chapter_demos.merge(merged, demo)
        self.assertEqual(len(build_chapter_demos.regions(merged)), len(demos))
        for demo in demos:
            self.assertEqual(merged, build_chapter_demos.merge(merged, demo))
        stripped = build_chapter_demos.strip(merged)
        self.assertIn('手写开头', stripped)
        self.assertIn('手写结尾', stripped)
        self.assertIn(slots.splitlines()[0], stripped)

    def test_native_probe_refuses_unsupported_hardware(self):
        import torch
        from experiments.native_nvfp4_probe import require_native_device
        with patch.object(torch.version, 'hip', None), patch.object(torch.cuda, 'is_available', return_value=True):
            for capability in ((7, 5), (8, 0), (9, 0), (12, 0), (13, 0)):
                with patch.object(torch.cuda, 'get_device_capability', return_value=capability), self.assertRaises(RuntimeError):
                    require_native_device(torch.device('cuda:0'))
            for capability in ((10, 0), (10, 3)):
                with patch.object(torch.cuda, 'get_device_capability', return_value=capability):
                    self.assertEqual(require_native_device(torch.device('cuda:0')), capability)

    def test_drift_in_displayed_code_is_rejected(self):
        original = Path.read_text
        target = ROOT/DEMOS[0]['document']

        def modified(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            return text.replace('layers, hidden, intermediate = 32, 4096, 11008',
                                'layers, hidden, intermediate = 16, 4096, 11008') if path == target else text

        with patch.object(Path, 'read_text', modified), self.assertRaises(ValueError):
            chapter_demo_lab.validate_docs()

    def test_bad_markers_do_not_erase_chapter_prose(self):
        demo = DEMOS[0]
        valid = build_chapter_demos.render(demo)
        for source in (valid.replace('<!-- /chapter-demo -->', ''),
                       valid+valid, '<!-- /chapter-demo -->'+valid):
            with self.subTest(source=source[:40]), self.assertRaises(ValueError):
                build_chapter_demos.merge(source, demo)
            with self.assertRaises(ValueError):
                build_chapter_demos.strip(source)

    def test_cli_default_is_read_only_and_export_is_explicit(self):
        script = ROOT/'experiments'/'chapter_demo_lab.py'
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            command = [sys.executable, '-B', str(script), '--chapter', '01', '--device', 'cpu']
            result = subprocess.run(command, cwd=out, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(out.iterdir()), [])
            invalid = subprocess.run(command+['--output-dir', str(out)], capture_output=True, text=True)
            self.assertEqual(invalid.returncode, 2)
            partial = subprocess.run(command+['--write-results'], capture_output=True, text=True)
            self.assertEqual(partial.returncode, 2)
            saved = subprocess.run(command+['--write-results', '--output-dir', str(out)],
                                   capture_output=True, text=True)
            self.assertEqual(saved.returncode, 0, saved.stderr)
            payload = json.loads((out/'chapter_demo_results.json').read_text())
            self.assertEqual(payload['passed_demos'], 1)
            self.assertEqual(payload['results'][0]['id'], '01')
            self.assertEqual(payload['results'][0]['runtime']['device'], 'cpu')
            self.assertGreater(payload['results'][0]['checked_tensor_count'], 0)
            self.assertIn('experiments/demos/ch01.py', payload['source_files_sha256'])


if __name__ == '__main__':
    unittest.main()

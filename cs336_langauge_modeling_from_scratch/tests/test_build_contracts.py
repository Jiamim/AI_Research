"""Build provenance and validation failure handling, using disposable fixtures."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts import build_book, run_checks


class BuildInputTests(unittest.TestCase):
    def test_dependency_hashes_cover_renderer_images_and_css_fonts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contents = {
                'scripts/build_book.py': b'build implementation',
                'scripts/artifact_io.py': b'publication implementation',
                'book/chapters/00-test.md': b'# Test\n![figure](../assets/test.svg)\n',
                'book/assets/test.svg': b'<svg/>',
                'book/assets/vendor/katex.min.css': b'@font-face { src: url(fonts/math.woff2); }',
                'book/assets/vendor/katex.min.js': b'katex',
                'book/assets/vendor/guide-chinese.woff2': b'chinese font',
                'book/assets/vendor/guide-emoji.woff2': b'emoji font',
                'book/assets/vendor/fonts/math.woff2': b'math font',
            }
            for name, data in contents.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            with mock.patch.object(build_book, 'ROOT', root), mock.patch.object(build_book, 'BOOK', root / 'book'):
                chapters = [root / 'book/chapters/00-test.md']
                baseline = build_book.build_inputs(chapters)
                self.assertEqual(set(baseline), set(contents))
                for name, data in contents.items():
                    with self.subTest(file=name):
                        (root / name).write_bytes(data + b'changed')
                        after = build_book.build_inputs(chapters)
                        self.assertNotEqual(after[name], baseline[name])
                        (root / name).write_bytes(data)
                # A referenced image disappearing must also fail, not vanish from the manifest.
                (root / 'book/assets/test.svg').unlink()
                with self.assertRaises(FileNotFoundError):
                    build_book.build_inputs(chapters)


class ValidationFailureTests(unittest.TestCase):
    def test_timeout_is_written_as_failure_instead_of_leaving_old_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'book').mkdir()
            report = root / 'book/validation.json'
            report.write_text('{"success":true}', encoding='utf-8')
            timed_out = subprocess.TimeoutExpired(['python'], 120, output=b'partial output', stderr=b'partial error')
            with mock.patch.object(run_checks, 'ROOT', root), \
                 mock.patch.object(run_checks.unittest.defaultTestLoader, 'discover', return_value=unittest.TestSuite([unittest.FunctionTestCase(lambda:None)])), \
                 mock.patch.object(run_checks.subprocess, 'run', side_effect=timed_out), \
                 mock.patch.object(run_checks.importlib.util, 'find_spec', return_value=None), \
                 self.assertRaises(SystemExit) as error:
                run_checks.main()
            self.assertEqual(error.exception.code, 1)
            result = json.loads(report.read_text(encoding='utf-8'))
            self.assertFalse(result['success'])
            self.assertTrue(result['distributed']['timed_out'])
            self.assertEqual(result['distributed']['stdout'], 'partial output')
            self.assertIn('partial error', result['distributed']['stderr'])
            self.assertEqual(result['triton_interpreter']['status'], 'skipped')

    def test_startup_error_is_a_failed_check(self):
        with mock.patch.object(run_checks.subprocess, 'run', side_effect=FileNotFoundError('missing executable')):
            result = run_checks.subprocess_check(['missing'], cwd=Path.cwd(), env={}, timeout=1)
        self.assertFalse(result['success'])
        self.assertIn('FileNotFoundError', result['stderr'])

    def test_artifact_exclusion_uses_modules_not_suite_string_contents(self):
        semantic = type('Semantic', (unittest.TestCase,), {'test_artifacts_word': lambda self: None})
        artifact = type('Artifact', (unittest.TestCase,), {'__module__': 'tests.test_artifacts', 'test_book': lambda self: None})
        keep = semantic('test_artifacts_word')
        nested = unittest.TestSuite([unittest.TestSuite([keep, unittest.TestSuite([artifact('test_book')])])])
        self.assertEqual(list(run_checks.semantic_tests(nested)), [keep])

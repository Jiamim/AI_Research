"""The committed book artifacts must come from the committed sources.

These tests read the generated files instead of rebuilding them, so they are cheap and catch the
common slip of editing a chapter or example and forgetting to rerun the build or the PDF export.
Fix a failure by rerunning, in order: make_appendices.py, build_book.py, render_pdf.py.
"""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / 'book'


def load_build_book():
    spec = importlib.util.spec_from_file_location('build_book', ROOT / 'scripts/build_book.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def combined_markdown():
    """The single top-level .md in book/. Located by suffix: on a GBK filesystem the Chinese name of a
    UTF-8 checkout is stored as different bytes, so the literal name would not be found."""
    found = sorted(BOOK.glob('*.md'))
    if len(found) != 1:
        raise AssertionError(f'expected one combined Markdown file in book/, found {[p.name for p in found]}')
    return found[0]


def read_json(name):
    return json.loads((BOOK / name).read_text(encoding='utf-8'))


class ArtifactConsistencyTests(unittest.TestCase):
    def test_html_was_built_from_current_chapters(self):
        stats = read_json('build-stats.json')
        chapters = sorted((BOOK / 'chapters').glob('*.md'))
        self.assertEqual(stats['chapters'], len(chapters))
        self.assertEqual(stats.get('chapters_sha256'), load_build_book().chapters_digest(chapters),
                         'chapters changed after the last build: rerun scripts/build_book.py')
        self.assertEqual(stats['html_sha256'], sha256((BOOK / 'guide.html').read_bytes()).hexdigest(),
                         'guide.html differs from the recorded build')
        current = load_build_book().build_inputs(chapters)
        recorded = stats.get('inputs_sha256', {})
        changed = sorted(name for name in current.keys() | recorded.keys() if current.get(name) != recorded.get(name))
        self.assertFalse(changed, f'build inputs changed: {changed}; rerun scripts/build_book.py')

    def test_combined_markdown_matches_recorded_build(self):
        self.assertEqual(read_json('build-stats.json').get('markdown_sha256'),
                         sha256(combined_markdown().read_bytes()).hexdigest(),
                         'combined Markdown differs from the build: rerun scripts/build_book.py')

    def test_pdf_was_exported_from_current_html(self):
        render = read_json('render-check.json')
        audit = read_json('pdf-audit.json')
        # The PDF the exporter names in its report: on a GBK filesystem a UTF-8 checkout stores the
        # Chinese file name as different bytes, so locate it by suffix rather than by literal name.
        pdfs = sorted(BOOK.glob('*.pdf'))
        self.assertEqual(len(pdfs), 1, [p.name for p in pdfs])
        pdf = sha256(pdfs[0].read_bytes()).hexdigest()
        self.assertEqual(render.get('htmlSha256'), read_json('build-stats.json')['html_sha256'],
                         'the PDF was exported from an older guide.html: rerun scripts/render_pdf.py')
        self.assertEqual(render['pdfSha256'], pdf)
        self.assertEqual(audit['sha256'], pdf)
        self.assertTrue(audit['passed'])
        self.assertTrue(render['glyphAudit']['passed'])
        for key in ('mathErrors', 'missingAnchors', 'externalAssets', 'pageErrors',
                    'printOverflows', 'pdfOutsidePageText'):
            self.assertEqual(render[key], [], f'export reported {key}')
        self.assertEqual(render['pdfBookmarks'], render['chapters'])

    def test_code_appendix_embeds_current_examples(self):
        appendix = (BOOK / 'chapters/36-code.md').read_text(encoding='utf-8')
        blocks = re.findall(r'^## ([a-z_]+\.py)\n\[对应文件\]\([^\n]+\)\n```python\n(.*?)\n```', appendix, re.M | re.S)
        embedded = dict(blocks)
        self.assertEqual(len(embedded), len(blocks), 'duplicate code appendix sections')
        paths = {path.name: path for path in (ROOT / 'examples').glob('*.py') if path.name != '__init__.py'}
        self.assertEqual(set(embedded), set(paths), 'code appendix module list is stale')
        for name, path in sorted(paths.items()):
            with self.subTest(module=path.name):
                # Avoid dumping whole source modules when a check fails.
                self.assertTrue(path.read_text(encoding='utf-8').rstrip('\n') == embedded[name].rstrip('\n'),
                                f'{path.name} changed after the appendix was generated: rerun scripts/make_appendices.py')


if __name__ == '__main__':
    unittest.main()

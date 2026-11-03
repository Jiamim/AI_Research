"""File-level export regressions using a real PDF and a simulated browser transport."""
import contextlib
from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import pymupdf

from scripts import render_pdf
from scripts.audit_pdf import parse_pages


class PdfRenderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        book = self.root / 'book'
        (book / 'chapters').mkdir(parents=True)
        (book / 'assets').mkdir()
        (book / 'chapters/00-reading-map.md').write_text('# 00｜Test\n', encoding="utf-8")
        (book / 'guide.html').write_text('<html>fixture</html>', encoding="utf-8")
        self.originals = {
            'CS336中文学习指南.pdf': b'previous PDF',
            'assets/preview.png': b'previous preview',
            'pdf-audit.json': b'{"previous": "glyph audit"}',
            'render-check.json': b'{"previous": "render check"}',
        }
        for name, contents in self.originals.items():
            (book / name).write_bytes(contents)
        self.fixture = self.root / 'fixture.pdf'
        with pymupdf.open() as pdf:
            font = pymupdf.Font('cjk').buffer
            for heading in ('Cover', 'Contents', '00｜Test'):
                page = pdf.new_page()
                page.insert_font(fontname='Fixture', fontbuffer=font)
                page.insert_text((50, 60), heading, fontname='Fixture', fontsize=16)
                page.insert_textbox(pymupdf.Rect(50, 90, 540, 310),
                                    'Text for the PDF audit. ' * 28, fontname='Fixture', fontsize=12)
            pdf.save(self.fixture, deflate=True)
        patcher = mock.patch.object(render_pdf, 'ROOT', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def browser_page(self, *, interrupted=False, mobile_overflow=False):
        page = mock.Mock()

        def evaluate(expression):
            if expression == 'document.fonts.ready':
                return None
            if 'chapters:' in expression:
                return dict(chapters=1, math=0, mathErrors=[], missingAnchors=[], externalAssets=[],
                            width=1440, viewport=1440)
            if 'main table' in expression:
                return []
            if expression == '({width:document.documentElement.scrollWidth,viewport:innerWidth})':
                return dict(width=500 if mobile_overflow else 390, viewport=390)
            raise AssertionError('Unexpected browser evaluation: ' + expression)

        def write_pdf(**options):
            if interrupted:
                Path(options['path']).write_bytes(b'incomplete PDF')
                raise RuntimeError('simulated browser failure')
            shutil.copy2(self.fixture, options['path'])

        page.evaluate.side_effect = evaluate
        page.screenshot.side_effect = lambda **options: Path(options['path']).write_bytes(b'new preview')
        page.pdf.side_effect = write_pdf
        page.locator.return_value.count.return_value = 1
        return page

    def assert_originals_preserved(self):
        for name, contents in self.originals.items():
            self.assertEqual((self.root / 'book' / name).read_bytes(), contents, name)
        self.assertEqual(list((self.root / 'book').glob('.render-*')), [])
        self.assertFalse((self.root / 'proof').exists())

    def test_browser_failure_does_not_replace_existing_outputs(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'browser failure'):
            render_pdf.render_book(self.browser_page(interrupted=True), self.root / 'proof')
        self.assert_originals_preserved()

    def test_out_of_range_proof_selection_preserves_existing_outputs(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'within 1..3'):
            render_pdf.render_book(self.browser_page(), self.root / 'proof', pages=parse_pages('4'))
        self.assert_originals_preserved()

    def test_late_validation_failure_preserves_outputs_and_proof_images(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'mobile horizontal overflow'):
            render_pdf.render_book(self.browser_page(mobile_overflow=True), self.root / 'proof',
                                   pages=parse_pages('3'), contact_sheets=True)
        self.assert_originals_preserved()

    def test_changing_html_during_export_does_not_publish_mismatched_pdf(self):
        page=self.browser_page()
        original=page.pdf.side_effect
        def changed(**options):
            original(**options)
            (self.root/'book/guide.html').write_text('<html>changed while rendering</html>',encoding='utf-8')
        page.pdf.side_effect=changed
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'changed during PDF export'):
            render_pdf.render_book(page,self.root/'proof')
        self.assert_originals_preserved()

    def test_success_publishes_matching_pdf_reports_and_selected_proof(self):
        page = self.browser_page()
        with contextlib.redirect_stdout(io.StringIO()):
            result = render_pdf.render_book(page, self.root / 'proof',
                                            pages=parse_pages('3'), contact_sheets=True)
        page.screenshot.assert_not_called()
        book = self.root / 'book'
        pdf_path = book / 'CS336中文学习指南.pdf'
        digest = sha256(pdf_path.read_bytes()).hexdigest()
        audit = json.loads((book / 'pdf-audit.json').read_text(encoding="utf-8"))
        stored = json.loads((book / 'render-check.json').read_text(encoding="utf-8"))
        self.assertEqual(audit['sha256'], digest)
        self.assertEqual(audit['file'], str(pdf_path))
        self.assertEqual(stored, result)
        self.assertEqual(result['pdfSha256'], digest)
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['exported_pages'], [3])
        self.assertEqual(audit['contact_sheets'], [str(self.root / 'proof/contact-01.png')])
        self.assertTrue(Path(audit['contact_sheets'][0]).is_file())
        self.assertEqual({p.name for p in (self.root / 'proof').glob('page-*.png')}, {'page-003.png'})
        self.assertEqual((book / 'assets/preview.png').read_bytes(), b'previous preview')
        self.assertEqual(list(book.glob('.render-*')), [])
        with pymupdf.open(stream=pdf_path.read_bytes(), filetype='pdf') as pdf:
            self.assertEqual(pdf.get_toc(), [[1, '00｜Test', 3]])


if __name__ == '__main__':
    unittest.main()

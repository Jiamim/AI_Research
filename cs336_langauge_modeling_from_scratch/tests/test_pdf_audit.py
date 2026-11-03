"""PDF proof exports must not narrow the scope of the automatic audit."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import weakref

import pymupdf

from scripts import audit_pdf
from scripts.audit_pdf import audit, parse_pages, select_pages


class PdfAuditTests(unittest.TestCase):
    def make_pdf(self, path, page_count=3):
        font = pymupdf.Font('helv').buffer
        with pymupdf.open() as pdf:
            for number in range(1, page_count + 1):
                page = pdf.new_page(width=595, height=842)
                page.insert_font(fontname='AuditFixture', fontbuffer=font)
                if number != 2:
                    page.insert_textbox(
                        pymupdf.Rect(50, 60, 540, 300),
                        'Text for the automatic PDF audit. ' * 20,
                        fontname='AuditFixture', fontsize=12,
                    )
                # A page containing only a footer must still be reported as blank.
                page.insert_text((275, 820), str(number), fontname='AuditFixture', fontsize=8)
            pdf.save(path)

    def test_default_audit_checks_all_pages_without_exporting_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_pdf(root / 'fixture.pdf')
            proof = root / 'proof'
            result = audit(root / 'fixture.pdf', root / 'reports/audit.json', proof)
            self.assertEqual(result['rasterized_pages'], 3)
            self.assertEqual(result['blank_pages'], [2])
            self.assertEqual(result['unembedded_fonts'], [])
            self.assertEqual(result['missing_glyphs'], [])
            self.assertFalse(result['passed'])
            self.assertEqual(result['exported_pages'], [])
            self.assertEqual(result['contact_sheets'], [])
            self.assertFalse(proof.exists())
            self.assertTrue((root / 'reports/audit.json').is_file())

    def test_selected_exports_still_detect_unselected_blank_page(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_pdf(root / 'fixture.pdf')
            proof = root / 'proof'
            proof.mkdir()
            previous = proof / 'page-999.png'
            previous.write_bytes(b'previous proof')
            result = audit(root / 'fixture.pdf', root / 'audit.json', proof,
                           pages={1, 3}, contact_sheets=True)
            self.assertEqual(result['rasterized_pages'], 3)
            self.assertEqual(result['blank_pages'], [2])
            self.assertFalse(result['passed'])
            self.assertEqual(result['exported_pages'], [1, 3])
            self.assertEqual(len(result['contact_sheets']), 1)
            self.assertTrue(Path(result['contact_sheets'][0]).is_file())
            self.assertEqual({p.name for p in proof.glob('page-*.png')},
                             {'page-001.png', 'page-003.png', 'page-999.png'})
            self.assertEqual(previous.read_bytes(), b'previous proof')

    def test_invalid_page_selection_fails_before_writing_outputs(self):
        self.assertEqual(select_pages(parse_pages('1, 3-5, 3'), 5, False), {1, 3, 4, 5})
        for value in ('', '0', '3-1', '1-2-3', 'one', '1,'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_pages(value)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_pdf(root / 'fixture.pdf')
            for pages in ({0}, {4}, {True}, {1.0}, set(), parse_pages('1-1000000000000')):
                with self.subTest(pages=pages), self.assertRaises(ValueError):
                    audit(root / 'fixture.pdf', root / 'audit.json', root / 'proof', pages=pages)
            self.assertFalse((root / 'proof').exists())
            self.assertFalse((root / 'audit.json').exists())

    def test_contact_sheets_release_each_batch_of_thumbnails(self):
        from PIL import Image

        previous = []
        batches = []
        original_save = audit_pdf.save_contact_sheet

        def observe(batch, proof_dir, number):
            self.assertTrue(all(ref() is None for ref in previous))
            previous[:] = [weakref.ref(thumbnail) for _, thumbnail in batch]
            batches.append([page for page, _ in batch])
            return original_save(batch, proof_dir, number)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_pdf(root / 'fixture.pdf', page_count=25)
            with mock.patch.object(audit_pdf, 'save_contact_sheet', side_effect=observe):
                result = audit(root / 'fixture.pdf', root / 'audit.json', root / 'proof', contact_sheets=True)
            self.assertEqual(batches, [list(range(1, 13)), list(range(13, 25)), [25]])
            self.assertEqual(result['rasterized_pages'], 25)
            self.assertEqual(result['exported_pages'], list(range(1, 26)))
            self.assertEqual(len(result['contact_sheets']), 3)
            self.assertEqual(result['blank_pages'], [2])
            self.assertTrue(all(ref() is None for ref in previous))
            with Image.open(result['contact_sheets'][-1]) as sheet:
                self.assertEqual(sheet.size, (1420, 1515))
                self.assertEqual(sheet.getpixel((370, 30)), (219, 227, 234))


if __name__ == '__main__':
    unittest.main()

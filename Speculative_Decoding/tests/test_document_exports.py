"""Regression checks for stale prose and silently unconverted math in exports."""
import base64
import hashlib
from pathlib import Path
import tempfile
import unittest

from examples.validate_artifacts import check_export


class ExportTests(unittest.TestCase):
    def test_prose_change_invalidates_html_even_when_counts_match(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'lesson.md'
            output = source.with_suffix('.html')
            source.write_text('# Lesson\n\nOriginal explanation.\n')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            output.write_text(f'<meta name="source-sha256" content="{digest}"><h1>Lesson</h1><p>Original explanation.</p>')
            self.assertEqual(check_export(source, output), [])
            source.write_text('# Lesson\n\nCorrected explanation.\n')
            self.assertTrue(any('fingerprint' in issue for issue in check_export(source, output)))

    def test_unknown_tex_command_is_detected_without_merror(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'lesson.md'
            output = source.with_suffix('.html')
            source.write_text('# Lesson\n')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            output.write_text(f'<meta name="source-sha256" content="{digest}">'
                              r'<math><mi>\arg</mi><mo>max</mo></math>')
            self.assertTrue(any('MathML command' in issue for issue in check_export(source, output)))

    def test_changed_figure_invalidates_embedded_image(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'lesson.md'
            output = source.with_suffix('.html')
            figure = Path(folder)/'figure.svg'
            figure.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
            source.write_text('![Figure](figure.svg)\n')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            embedded = base64.b64encode(figure.read_bytes()).decode()
            output.write_text(f'<meta name="source-sha256" content="{digest}">'
                              f'<img src="data:image/svg+xml;base64,{embedded}">')
            self.assertEqual(check_export(source, output), [])
            figure.write_text('<svg xmlns="http://www.w3.org/2000/svg"><title>Updated</title></svg>')
            self.assertTrue(any('stale embedded image' in issue for issue in check_export(source, output)))


if __name__ == '__main__':
    unittest.main()

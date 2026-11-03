"""Text-only acceptance for the offline book: artifacts, links, TeX and embedded CJK fonts."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.artifact_io import write_json


def exists_any_encoding(target):
    """Path.exists, tolerant of a UTF-8 checkout read under a non-UTF-8 locale (e.g. zh_CN.GBK on Linux):
    there the Chinese file names on disk are UTF-8 bytes that the locale decodes differently."""
    if target.exists():
        return True
    parent = target.parent
    if not parent.is_dir():
        return False
    wanted = target.name.encode('utf-8')
    return any(os.fsencode(entry) == wanted for entry in os.listdir(parent))


def check(include_pdf=True):
    import unittest
    from tests.test_artifacts import ArtifactConsistencyTests
    methods = unittest.defaultTestLoader.getTestCaseNames(ArtifactConsistencyTests)
    suite = unittest.TestSuite(ArtifactConsistencyTests(name) for name in methods
                              if include_pdf or name != 'test_pdf_was_exported_from_current_html')
    tested = unittest.TestResult()
    suite.run(tested)
    issues = [f'{test.id()}: {error.splitlines()[-1]}' for test, error in tested.failures + tested.errors]
    parser = MarkdownIt('commonmark').enable('table')
    chapters = sorted((ROOT/'book/chapters').glob('*.md'))
    for path in [ROOT/'README.md', ROOT/'sources/README.md', *chapters]:
        for token in parser.parse(path.read_text(encoding='utf-8')):
            for item in token.children or []:
                attribute = 'href' if item.type == 'link_open' else 'src' if item.type == 'image' else None
                if attribute:
                    address = item.attrGet(attribute)
                    url = urlsplit(address)
                    if not url.scheme and not url.netloc and url.path:
                        target = (path.parent/unquote(url.path)).resolve()
                        if not exists_any_encoding(target):issues.append(f'{path.relative_to(ROOT)}: missing {address}')
    html_path = ROOT/'book/guide.html'
    soup = BeautifulSoup(html_path.read_text(encoding='utf-8'), 'html.parser')
    ids = [node['id'] for node in soup.select('[id]')]
    if len(ids) != len(set(ids)):issues.append('duplicate HTML anchors')
    known = set(ids)
    issues.extend('missing HTML anchor '+node['href'] for node in soup.select('a[href^="#"]') if node['href'][1:] not in known)
    formulas = [dict(tex=node['data-tex'], display=node.get('data-display')=='true') for node in soup.select('.math-source')]
    stats=json.loads((ROOT/'book/build-stats.json').read_text(encoding='utf-8'))
    if len(formulas)!=stats['math_expressions'] or len(soup.select('section.chapter'))!=stats['chapters']:
        issues.append('rendered chapter/formula counts differ from build record')
    formula_status = 'not_run'
    if shutil.which('node'):
        javascript = """
const fs = require('fs'); const path = require('path');
const katex = require(path.join(process.cwd(), 'book/assets/vendor/katex.min.js'));
const formulas = JSON.parse(fs.readFileSync(0, 'utf8')); const errors = [];
for (const [i, item] of formulas.entries()) {
  try { katex.renderToString(item.tex, {displayMode:item.display, throwOnError:true, strict:'ignore', trust:false}); }
  catch (error) { errors.push({index:i, error:error.message}); }
}
process.stdout.write(JSON.stringify(errors));
"""
        result = subprocess.run(['node','-e',javascript], input=json.dumps(formulas), capture_output=True,
                                text=True, encoding='utf-8', cwd=ROOT, timeout=60)
        if result.returncode:
            issues.append('KaTeX validation failed: '+result.stderr[-500:])
            formula_status = 'failed'
        else:
            errors = json.loads(result.stdout)
            issues.extend('TeX '+str(error) for error in errors)
            formula_status = 'failed' if errors else 'passed'
    else:
        issues.append('Node.js is required for text-only KaTeX parsing; formulas were not checked')
    font_status = 'not_run'
    try:
        from fontTools.ttLib import TTFont
        font = TTFont(ROOT/'book/assets/vendor/guide-chinese.woff2')
        covered = set(font.getBestCmap())
        font.close()
        missing = sorted({c for c in soup.get_text() if '\u3400' <= c <= '\u9fff' and ord(c) not in covered})
        if missing:issues.append('missing embedded CJK glyphs: '+''.join(missing))
        font_status = 'failed' if missing else 'passed'
    except ImportError:
        issues.append('fontTools is required to validate the embedded CJK font')
    return dict(success=not issues, chapters=len(chapters), math_expressions=len(formulas),
                formulas=formula_status, chinese_font=font_status, artifact_tests=tested.testsRun,
                pdf_checked=include_pdf, html_sha256=sha256(html_path.read_bytes()).hexdigest(), issues=issues)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--without-pdf', action='store_true', help='check current HTML/Markdown; report PDF as not checked')
    parser.add_argument('--report', default='book/text-check.json')
    args = parser.parse_args()
    try:
        report = check(include_pdf=not args.without_pdf)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report = dict(success=False, issues=[type(error).__name__+': '+str(error)], pdf_checked=not args.without_pdf)
    write_json(ROOT/args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['success']:raise SystemExit(1)


if __name__ == '__main__':main()

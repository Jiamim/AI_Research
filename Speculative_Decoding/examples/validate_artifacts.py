"""Check maintained documents, embedded figures, experiment traces and public references.

Run: python -m examples.validate_artifacts [--require-html] [--output /path/to/report.json]
HTML checks run only when the optional export exists, unless --require-html is set.
Checks local artifacts only. This does not contact reference URLs or render a browser.
"""
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree
import argparse
import base64
import json
import hashlib
import re
import shutil
import subprocess

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from .forecast_contract import load_forecasts, render_table, TABLE_START, TABLE_END

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'docs/投机解码深度调研与实践.md'


def document_soup(path):
    text = path.read_text(encoding='utf-8')
    if path.suffix == '.md':
        text = MarkdownIt('commonmark', {'html': True}).enable('table').render(text)
    return BeautifulSoup(text, 'html.parser')


def check_links(path, soup):
    errors = []
    counts = Counter(tag['id'] for tag in soup.find_all(id=True))
    errors += [f'{path.name}: duplicate id {key}' for key, count in counts.items() if count > 1]
    for tag in soup.select('a[href],img[src]'):
        uri = tag.get('href', tag.get('src', ''))
        parsed = urlparse(uri)
        if parsed.scheme or parsed.netloc:
            continue
        target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
        if not target.exists():
            errors.append(f'{path.name}: missing local target {uri}')
        elif parsed.fragment and target.suffix in {'.md', '.html'}:
            target_soup = soup if target == path else document_soup(target)
            if not target_soup.find(id=unquote(parsed.fragment)):
                errors.append(f'{path.name}: missing fragment {uri}')
    return errors


def check_calculator(html):
    if not shutil.which('node'):
        return 'not run: Node.js unavailable'
    scripts = html.find_all('script')
    if len(scripts) != 1:
        raise ValueError('Expected exactly one local calculator script')
    harness = r'''
const vm=require('vm'),assert=require('assert'),fs=require('fs');
const defaults={alpha:'.8',gamma:'4',cost:'.1',slope:'.035',overhead:'0',prefill:'.1'},nodes={};
for(const [id,value] of Object.entries(defaults)){
 nodes[id]={value,addEventListener:(event,fn)=>assert.equal(event,'input')};
 nodes[id+'Value']={textContent:''};
}
for(const id of ['expected','speed','e2e','best','speedChart'])nodes[id]={textContent:'',innerHTML:''};
const c={document:{getElementById:id=>{assert(nodes[id],id);return nodes[id];}}};vm.createContext(c);
vm.runInContext(fs.readFileSync(0,'utf8'),c);
assert.equal(nodes.expected.textContent,'3.362');
assert.equal(nodes.speed.textContent,(3.3616/1.54).toFixed(2)+'×');
assert(nodes.speedChart.innerHTML.includes('polyline'));
nodes.gamma.value='0';c.refresh();assert.equal(nodes.speed.textContent,'1.00×');assert.equal(nodes.e2e.textContent,'1.00×');
nodes.gamma.value='4';nodes.alpha.value='1';c.refresh();assert.equal(nodes.expected.textContent,'5.000');
nodes.alpha.value='0';c.refresh();assert.equal(nodes.expected.textContent,'1.000');assert.equal(nodes.best.textContent,'0（关闭）');
'''
    subprocess.run(['node', '-e', harness], input=scripts[0].get_text(), text=True,
                   capture_output=True, check=True, timeout=15)
    return 'passed: default, gamma=0, alpha=0/1 and SVG updates'


def check_export(source, html_path):
    """Validate every exported deep dive, not only the original main document."""
    from .check_tutorial_snippets import PATTERN
    issues = []
    markdown, rendered = document_soup(source), document_soup(html_path)
    fingerprint = rendered.find('meta', attrs={'name': 'source-sha256'})
    if not fingerprint or fingerprint.get('content') != hashlib.sha256(source.read_bytes()).hexdigest():
        issues.append(f'{source.name}: HTML source fingerprint is missing or stale')
    original, embedded = markdown.find_all('img'), rendered.find_all('img')
    if len(original) != len(embedded):
        issues.append(f'{source.name}: source/HTML image counts differ')
    for src, dest in zip(original, embedded):
        uri = dest.get('src', '')
        if not uri.startswith('data:'):
            issues.append(f'{source.name}: image was not embedded')
            continue
        try:
            data = base64.b64decode(uri.split(',', 1)[1], validate=True)
        except (ValueError, IndexError):
            issues.append(f'{source.name}: malformed embedded image')
            continue
        if data != (source.parent / unquote(src['src'])).read_bytes():
            issues.append(f'{source.name}: stale embedded image {src["src"]}')
    if [c.get_text() for c in markdown.select('pre > code')] != [c.get_text() for c in rendered.select('pre > code')]:
        issues.append(f'{source.name}: source/HTML code blocks differ')
    if rendered.find_all('script', src=True) or rendered.find_all('link', rel='stylesheet'):
        issues.append(f'{source.name}: external rendering dependency')
    if rendered.find_all('merror'):
        issues.append(f'{source.name}: invalid MathML')
    for token in rendered.select('math mi, math mo, math mtext'):
        if re.search(r'\\[A-Za-z]+', token.get_text()):
            issues.append(f'{source.name}: unconverted MathML command {token.get_text()}')
    if any('$' in text and not text.find_parent(['script', 'style', 'pre', 'code', 'math'])
           for text in rendered.find_all(string=True)):
        issues.append(f'{source.name}: unconverted math delimiter')
    snippets = PATTERN.findall(source.read_text(encoding='utf-8'))
    if len(rendered.select('.runnable-label')) != len(snippets):
        issues.append(f'{source.name}: executable label count differs')
    return issues


def validate(require_html=False):
    html_path = MAIN.with_suffix('.html')
    if require_html and not html_path.is_file():
        raise ValueError('HTML export missing; run python -m examples.build_html first')
    doc_sources = sorted((ROOT / 'docs').glob('*.md'))
    maintained = [ROOT / 'README.md', ROOT / 'data/README.md', *doc_sources]
    html = document_soup(html_path) if html_path.is_file() else None
    exports = [source.with_suffix('.html') for source in doc_sources if source.with_suffix('.html').is_file()]
    maintained.extend(exports)
    issues = []
    for path in maintained:
        issues.extend(check_links(path, document_soup(path)))
    for path in exports:
        issues.extend(check_export(path.with_suffix('.md'), path))
    markdown = document_soup(MAIN)
    source_text = MAIN.read_text(encoding='utf-8')
    source_images = markdown.find_all('img')
    all_svg = sorted((ROOT / 'docs/figures').glob('*.svg'))
    for path in all_svg:
        ElementTree.parse(path)
    from .check_tutorial_snippets import PATTERN
    snippets = PATTERN.findall(source_text)
    all_snippets = sum(len(PATTERN.findall(p.read_text(encoding='utf-8'))) for p in doc_sources)
    source_sections = markdown.find_all('h3')
    chapter_count = len(re.findall(r'^## \d+\.', source_text, flags=re.M))
    embedded = []
    if html is not None:
        embedded = html.find_all('img')
        if len(source_images) != len(embedded):
            issues.append('Source/HTML image counts differ')
        for src, dest in zip(source_images, embedded):
            path = MAIN.parent / src['src']
            data = base64.b64decode(dest['src'].split(',', 1)[1])
            if data != path.read_bytes():
                issues.append(f'Stale embedded image: {path.name}')
            ElementTree.fromstring(data)
        if html.find_all('script', src=True) or html.find_all('link', rel='stylesheet'):
            issues.append('External script/stylesheet dependency')
        if not html.find_all('math') or html.find_all('merror'):
            issues.append('Missing/invalid MathML')
        if any('$' in text and not text.find_parent(['script', 'style', 'pre', 'code', 'math'])
               for text in html.find_all(string=True)):
            issues.append('Unconverted math delimiter')
        if len(html.select('.runnable-label')) != len(snippets):
            issues.append('Executable code labels do not match source snippets')
        if [code.get_text() for code in markdown.select('pre > code')] != [
                code.get_text() for code in html.select('pre > code')]:
            issues.append('Source/HTML code blocks differ')
        if len(source_sections) != len(html.find_all('h3')):
            issues.append('Knowledge section count changed during HTML rendering')
        if len(html.select('.chapter-guide')) != chapter_count:
            issues.append('Missing chapter knowledge navigation')
    references = json.loads((ROOT / 'docs/references.json').read_text())['sources']
    source_records = {record['id']: record for record in references}
    if len(source_records) != len(references):
        issues.append('Duplicate public reference IDs')
    for reference in references:
        parsed = urlparse(reference['url'])
        if parsed.scheme not in {'https', 'http'} or not parsed.netloc:
            issues.append(f"Invalid reference URL: {reference['id']}")
        digest = reference.get('snapshot_sha256')
        if digest and not re.fullmatch(r'[0-9a-f]{64}', digest):
            issues.append(f"Malformed source fingerprint: {reference['id']}")
    ledger = load_forecasts()
    forecasts = ledger['forecasts']
    if source_text.count(TABLE_START) != 1 or source_text.count(TABLE_END) != 1:
        issues.append('Forecast table markers missing or duplicated')
    else:
        rendered = source_text[source_text.index(TABLE_START):source_text.index(TABLE_END)+len(TABLE_END)]
        if rendered != render_table(ledger):
            issues.append('Forecast table differs from reviewed ledger')
    for forecast in forecasts:
        for source in forecast['basis']:
            if source not in source_records:
                issues.append(f"Forecast {forecast['id']} has no public reference for {source}")
    results = json.loads((ROOT / 'data/experiment_results.json').read_text())
    for name in ['tiny_transformer', 'lookup_transformer']:
        sample = results[name]
        accepted = committed = proposed = 0
        for row in sample['trace']:
            committed += len(row['emitted'])
            accepted += row['accepted']
            proposed += len(row['proposed'])
            if row['cache_length'] != row['prefix_length'] + len(row['emitted']) - 1:
                issues.append(f'{name}: cache does not match committed prefix')
            if row['accepted'] > len(row['emitted']):
                issues.append(f'{name}: accepted count exceeds committed count')
        for key, total in dict(accepted=accepted, committed=committed, proposed=proposed).items():
            if sample['stats'][key] != total:
                issues.append(f'{name}: invalid {key} summary')
    if issues:
        raise ValueError('\n'.join(issues))
    return dict(status='passed', maintained_documents=len(maintained),
                html_validation='passed' if html is not None else 'not run: optional HTML not generated',
                embedded_svg_count=len(embedded) if html is not None else None,
                math_nodes=len(html.find_all('math')) if html is not None else None,
                figure_xml_count=len(all_svg),
                knowledge_sections=len(source_sections), tables=len(markdown.find_all('table')),
                chapters=chapter_count, forecasts_checked=len(forecasts),
                executable_snippets=len(snippets),
                executable_snippets_all_documents=all_snippets,
                markdown_documents=len(doc_sources), html_exports_checked=len(exports),
                calculator=check_calculator(html) if html is not None else 'not run: optional HTML not generated',
                public_references_checked=len(references),
                reference_body_hashes="not checked: third-party bodies are not distributed",
                trace_validation='passed: neural and lookup',
                browser_render='not performed; figure images visually inspected separately', issues=[])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-html', action='store_true',
                        help='Fail if the optional HTML export has not been generated')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = validate(require_html=args.require_html)
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text)

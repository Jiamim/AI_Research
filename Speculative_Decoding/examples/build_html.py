"""Build a self-contained offline HTML reader, native MathML and calculator.

Run from project root: python -m examples.build_html
"""
from pathlib import Path
import base64
import html
import hashlib
import re
from markdown_it import MarkdownIt
from latex2mathml.converter import convert
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "投机解码深度调研与实践.md"


def inline_math(state, silent):
    start = state.pos
    if state.src[start] != "$" or state.src.startswith("$$", start):
        return False
    end = state.src.find("$", start + 1)
    if end < 0 or "\n" in state.src[start + 1:end]:
        return False
    if not silent:
        token = state.push("math_inline", "math", 0)
        token.content = state.src[start + 1:end]
    state.pos = end + 1
    return True


def block_math(state, start_line, end_line, silent):
    first = state.src[state.bMarks[start_line] + state.tShift[start_line]:state.eMarks[start_line]].strip()
    if first != "$$":
        return False
    if silent:
        return True
    next_line = start_line + 1
    while next_line < end_line:
        line = state.src[state.bMarks[next_line]:state.eMarks[next_line]].strip()
        if line == "$$":
            break
        next_line += 1
    if next_line == end_line:
        raise ValueError(f"Unclosed block math near line {start_line}")
    token = state.push("math_block", "math", 0)
    token.block = True
    token.map = [start_line, next_line + 1]
    token.content = state.getLines(start_line + 1, next_line, 0, False).strip()
    state.line = next_line + 1
    return True


def render_inline(tokens, idx, options, env):
    return convert(tokens[idx].content, display="inline")


def render_block(tokens, idx, options, env):
    return '<div class="equation">' + convert(tokens[idx].content, display="block") + '</div>\n'


CSS = """
:root{--ink:#172b43;--muted:#5b6d80;--blue:#2457a7;--line:#dce5ef;--paper:#fff}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}
body{margin:0;background:#edf2f8;color:var(--ink);font:16px/1.85 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.topbar{background:#17375e;color:#fff;padding:12px 5vw;font-size:14px;letter-spacing:.04em}
.topbar a{color:#deebff}main{max-width:1100px;margin:28px auto;padding:40px 64px;background:var(--paper);border-radius:12px;box-shadow:0 8px 35px #14345a10}
h1{font-size:34px;line-height:1.4;margin-top:0;color:#163b69}h2{font-size:25px;margin-top:60px;padding-top:18px;border-top:2px solid var(--line);line-height:1.5}
h3{font-size:20px;margin-top:32px;line-height:1.5}p{margin:15px 0}a{color:#2457a7;text-decoration:none}a:hover{text-decoration:underline}
img{max-width:100%;height:auto;display:block;margin:22px auto}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.65;margin:22px 0;display:block;overflow-x:auto}
th,td{padding:10px 12px;border:1px solid var(--line);text-align:left;vertical-align:top}th{background:#ecf3fc;font-weight:650}tr:nth-child(even){background:#f8fafd}
pre{background:#10273f;color:#e2ecf8;padding:20px 23px;border-radius:8px;overflow:auto;line-height:1.65;font-size:13px}
.chapter-guide{padding:12px 18px;background:#f0f5fb;border-radius:8px;margin:16px 0}.chapter-guide summary{cursor:pointer;font-weight:600}.chapter-guide ul{columns:2;column-gap:24px;padding-left:22px}.chapter-guide li{break-inside:avoid;font-size:14px}.runnable-label{color:#126650;font-size:13px;margin:18px 0 -12px;font-weight:600}.section-link{color:#7891aa;margin-left:8px;font-size:14px}
code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}p code,li code,td code{background:#edf2f7;padding:2px 5px;border-radius:3px;font-size:.88em}
blockquote{margin:24px 0;padding:12px 24px;background:#ecf6f5;border-left:4px solid #138a72;border-radius:0 8px 8px 0}
math{font-family:"STIX Two Math","Cambria Math",math;font-size:1.06em}.equation{overflow-x:auto;padding:16px 5px;margin:18px 0;background:#f8fafd;border-radius:7px}
.calculator{border:1px solid #b7cfe9;border-radius:10px;background:#f4f8fe;padding:24px;margin:32px 0}.calculator h2{margin:0;padding:0;border:0;font-size:22px}
.calc-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px 30px;margin:20px 0}.control label{display:flex;justify-content:space-between;font-size:14px}.control input{width:100%;accent-color:#2457a7}
.metrics{display:flex;gap:14px;flex-wrap:wrap}.metric{background:#fff;border:1px solid #dce5ef;border-radius:7px;padding:12px 18px;flex:1;min-width:160px}.metric strong{display:block;font-size:25px;color:#2457a7}.note{color:var(--muted);font-size:13px}
#speedChart{display:block;width:100%;max-height:320px;background:white;border-radius:6px;margin-top:22px}.skip{display:block;text-align:right;font-size:13px}
@media(max-width:700px){main{margin:0;padding:25px 19px;border-radius:0}h1{font-size:28px}h2{font-size:23px}.calc-grid{grid-template-columns:1fr}.chapter-guide ul{columns:1}table{font-size:13px}pre{padding:15px;font-size:12px}}
@media print{body{background:white;font-size:11pt}main{max-width:none;padding:0;margin:0;box-shadow:none}.topbar,.calculator,.skip{display:none}h2{break-before:auto}pre,table,figure,img,.equation{break-inside:avoid}pre{background:#f4f6f8;color:#172b43}a{color:#172b43}h2,h3{break-after:avoid}}
"""

CALCULATOR = """
<section class="calculator" aria-label="投机解码理论计算器">
<h2>动手调整：什么条件下值得投机？</h2>
<p class="note">常数条件接受率近似；验证成本 v(γ)=1+kγ；串行草稿成本 γc。这里只计算理论情景，不代表硬件实测。</p>
<div class="calc-grid">
<div class="control"><label for="alpha">条件接受率 α <output id="alphaValue"></output></label><input id="alpha" type="range" min="0" max="1" step="0.01" value="0.8"></div>
<div class="control"><label for="gamma">草稿长度 γ <output id="gammaValue"></output></label><input id="gamma" type="range" min="0" max="16" step="1" value="4"></div>
<div class="control"><label for="cost">草稿单步成本 c <output id="costValue"></output></label><input id="cost" type="range" min="0" max="0.5" step="0.01" value="0.1"></div>
<div class="control"><label for="slope">验证成本增长系数 k <output id="slopeValue"></output></label><input id="slope" type="range" min="0" max="0.3" step="0.005" value="0.035"></div>
<div class="control"><label for="overhead">额外开销 o <output id="overheadValue"></output></label><input id="overhead" type="range" min="0" max="0.5" step="0.01" value="0"></div>
<div class="control"><label for="prefill">基线中 prefill 时间占比 f <output id="prefillValue"></output></label><input id="prefill" type="range" min="0" max="0.9" step="0.01" value="0.1"></div>
</div>
<div class="metrics" aria-live="polite"><div class="metric">预期本轮提交 token<strong id="expected"></strong></div><div class="metric">理论 decode 加速<strong id="speed"></strong></div><div class="metric">理论端到端加速<strong id="e2e"></strong></div><div class="metric">0–16 内最优 γ<strong id="best"></strong></div></div>
<svg id="speedChart" viewBox="0 0 780 290" role="img" aria-label="理论加速比随草稿长度的变化"></svg>
<p class="note">γ=0 按关闭投机、速度为 1 处理；端到端估算假设 prefill 不变，不含额外草稿 prefill。EOS、序列尾部及 batch 变化不在该模型内。图中绿色点表示当前选择。</p>
</section>
"""

SCRIPT = """
const ids=['alpha','gamma','cost','slope','overhead','prefill'];
function expected(a,g){let total=1,power=1;for(let i=1;i<=g;i++){power*=a;total+=power;}return total;}
function modeledSpeed(a,g,c,k,o){return g===0?1:expected(a,g)/(1+g*(c+k)+o);}
function refresh(){
 const vals={};for(const id of ids){vals[id]=Number(document.getElementById(id).value);document.getElementById(id+'Value').textContent=id==='gamma'?vals[id]:vals[id].toFixed(id==='slope'?3:2);}
 const a=vals.alpha,g=vals.gamma,c=vals.cost,k=vals.slope,o=vals.overhead,f=vals.prefill;
 const values=Array.from({length:17},(_,i)=>modeledSpeed(a,i,c,k,o)), s=values[g];
 let best=0;for(let i=1;i<values.length;i++){if(values[i]>values[best]+1e-10)best=i;}
 document.getElementById('expected').textContent=expected(a,g).toFixed(3);
 document.getElementById('speed').textContent=s.toFixed(2)+'×';
 document.getElementById('e2e').textContent=(1/(f+(1-f)/s)).toFixed(2)+'×';
 document.getElementById('best').textContent=best===0?'0（关闭）':best;
 const ymax=Math.max(1.25,...values)*1.12, x=i=>55+i*42.5, y=v=>240-v/ymax*208;
 let svg='<text x="18" y="21" fill="#5b6d80" font-size="12">理论 decode 加速比</text>';
 for(let j=0;j<=4;j++){let v=ymax*j/4;svg+=`<line x1="55" y1="${y(v)}" x2="735" y2="${y(v)}" stroke="#e2e9f2"/><text x="45" y="${y(v)+4}" text-anchor="end" fill="#5b6d80" font-size="11">${v.toFixed(1)}</text>`;}
 svg+=`<line x1="55" y1="${y(1)}" x2="735" y2="${y(1)}" stroke="#c56b52" stroke-dasharray="5 5"/>`;
 svg+=`<polyline points="${values.map((v,i)=>x(i)+','+y(v)).join(' ')}" stroke="#2457a7" fill="none" stroke-width="3"/>`;
 for(let i=0;i<=16;i+=2)svg+=`<text x="${x(i)}" y="262" text-anchor="middle" fill="#5b6d80" font-size="12">${i}</text>`;
 svg+=`<circle cx="${x(g)}" cy="${y(s)}" r="6" fill="#138a72"/><text x="660" y="283" font-size="12" fill="#5b6d80">草稿长度 γ</text>`;
 document.getElementById('speedChart').innerHTML=svg;
}
for(const id of ids)document.getElementById(id).addEventListener('input',refresh);refresh();
"""


def build(source=SOURCE, linked_sources=()):
    source = source.resolve()
    destination = source.with_suffix(".html")
    markdown_source = source.read_text(encoding='utf-8')
    title = markdown_source.splitlines()[0].removeprefix("# ")
    date_match = re.search(r"(?:文档更新日期|资料核验日期|复核日期|核验日期)[：:]\s*(\d{4}-\d{2}-\d{2})", markdown_source)
    verified_date = date_match.group(1) if date_match else "资料日期见正文"
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    md.inline.ruler.before("escape", "math_inline", inline_math)
    md.block.ruler.before("fence", "math_block", block_math, {"alt": ["paragraph", "reference", "blockquote", "list"]})
    md.renderer.rules["math_inline"] = render_inline
    md.renderer.rules["math_block"] = render_block
    body = md.render(markdown_source)
    # Long-form lessons need links to individual knowledge points, not only
    # the top-level chapters. This changes presentation, never source snippets.
    body = re.sub(r'<!-- runnable: ([a-z0-9_]+) -->',
                  lambda m: '<p class="runnable-label">可运行示例 · ' + m.group(1) + '</p>', body)
    soup = BeautifulSoup(body, "html.parser")
    # --all creates a browsable offline collection while retaining source links
    # to code/data and the Markdown source in the top bar.
    from urllib.parse import urlparse, unquote, urlunparse
    linked_sources = {p.resolve() for p in linked_sources}
    for link in soup.select('a[href]'):
        parsed = urlparse(link['href'])
        if not parsed.scheme and not parsed.netloc and parsed.path.endswith('.md'):
            target = (source.parent / unquote(parsed.path)).resolve()
            if target in linked_sources:
                link['href'] = urlunparse(parsed._replace(path=parsed.path[:-3]+'.html'))
    for heading in soup.find_all("h3"):
        match = re.match(r"(\d+)\.(\d+)\s", heading.get_text())
        if match:
            section_id = f"s{match.group(1)}-{match.group(2)}"
            explicit = soup.find(id=section_id)
            if explicit is not None and explicit.name == 'a' and not explicit.get_text(strip=True):
                explicit.decompose()
            heading["id"] = section_id
    for chapter in soup.find_all("h2"):
        links = []
        sibling = chapter.find_next_sibling()
        while sibling is not None and sibling.name != "h2":
            if sibling.name == "h3" and sibling.get("id"):
                links.append((sibling["id"], sibling.get_text()))
            sibling = sibling.find_next_sibling()
        if links:
            guide = soup.new_tag("details", attrs={"class": "chapter-guide"})
            summary = soup.new_tag("summary")
            summary.string = f"本章知识点（{len(links)}）"
            guide.append(summary)
            listing = soup.new_tag("ul")
            for anchor, label in links:
                item, link = soup.new_tag("li"), soup.new_tag("a", href=f"#{anchor}")
                link.string = label
                item.append(link)
                listing.append(item)
            guide.append(listing)
            chapter.insert_after(guide)
    body = str(soup)

    def embed_image(match):
        path = source.parent / html.unescape(match.group(1))
        mime = "image/svg+xml" if path.suffix == ".svg" else "image/png"
        return 'src="data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode() + '"'

    body = re.sub(r'src="(figures/[^\"]+)"', embed_image, body)
    # The source explicitly chooses where the calculator belongs.
    calculator_marker = '<div id="speculative-calculator"></div>'
    has_calculator = calculator_marker in body
    body = body.replace(calculator_marker, CALCULATOR)
    calculator_script = f'<script>{SCRIPT}</script>' if has_calculator else ''
    document = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="投机解码深度学习文档：概率证明、真实 Transformer、KV 回滚、技术路线、性能模型与可运行代码。">
<meta name="source-sha256" content="{hashlib.sha256(source.read_bytes()).hexdigest()}">
<title>{html.escape(title)}</title><style>{CSS}</style></head>
<body><div class="topbar">SPECULATIVE DECODING · 中文调研文档 · {verified_date} · <a href="{html.escape(source.name)}">Markdown 源文档</a></div>
<main id="page-root">{body}<a class="skip" href="#page-root">返回顶部 ↑</a></main>{calculator_script}</body></html>'''
    destination.write_text(document, encoding='utf-8')
    print(f"Built {destination.name}: {len(document.encode()):,} bytes; embedded figures and MathML work offline.")
    return destination


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--source', type=Path, help='Build a specific Markdown document')
    group.add_argument('--all', action='store_true', help='Build and cross-link every docs/*.md document')
    args = parser.parse_args()
    sources = sorted(SOURCE.parent.glob('*.md')) if args.all else [args.source or SOURCE]
    for source in sources:
        build(source, linked_sources=sources)


if __name__ == "__main__":
    main()

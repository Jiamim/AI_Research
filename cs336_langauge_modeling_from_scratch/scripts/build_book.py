"""Build an offline single-file HTML book and a combined Markdown edition."""
from pathlib import Path
from hashlib import sha256
import base64
import html
import json
import re
import tempfile
from markdown_it import MarkdownIt
try:
    from scripts.artifact_io import publish_files
except ModuleNotFoundError:
    from artifact_io import publish_files

ROOT=Path(__file__).resolve().parents[1]
BOOK=ROOT/'book'


def chapters_digest(paths):
    """One hash over every chapter's name and bytes, so a stale guide.html can be detected."""
    digest=sha256()
    for path in sorted(paths):
        digest.update(path.name.encode('utf-8')+b'\0'+path.read_bytes()+b'\0')
    return digest.hexdigest()


def build_inputs(chapters):
    """Hash the renderer, chapters, embedded SVGs, KaTeX and every embedded font."""
    vendor = BOOK / 'assets/vendor'
    paths = {ROOT / 'scripts/build_book.py', ROOT / 'scripts/artifact_io.py', *chapters,
             vendor / 'katex.min.css', vendor / 'katex.min.js',
             vendor / 'guide-chinese.woff2', vendor / 'guide-emoji.woff2'}
    parser = MarkdownIt('commonmark', {'html': True}).enable('table')
    for chapter in chapters:
        for token in parser.parse(chapter.read_text(encoding='utf-8')):
            for child in token.children or []:
                if child.type == 'image':
                    path = (chapter.parent / child.attrGet('src')).resolve()
                    if path.suffix == '.svg':
                        paths.add(path)
    for url in re.findall(r'url\(([^)]+)\)', (vendor / 'katex.min.css').read_text(encoding='utf-8')):
        paths.add((vendor / url.strip('"\'')).resolve())
    return {path.relative_to(ROOT).as_posix(): sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def data_uri(path):
    mime={'woff2':'font/woff2','woff':'font/woff','ttf':'font/ttf'}.get(path.suffix[1:],'application/octet-stream')
    return f'data:{mime};base64,'+base64.b64encode(path.read_bytes()).decode()


def protect_math(text):
    # Protect math before Markdown can interpret underscores, backslashes and table pipes.
    pieces=re.split(r'(```[\s\S]*?```|`[^`\n]*`)',text)
    pattern=r'\$\$([\s\S]*?)\$\$|(?<!\\)\$([^$\n]+?)(?<!\\)\$'
    def repl(m):
        display=m.group(1) is not None
        formula=m.group(1) if display else m.group(2)
        tag='div' if display else 'span'
        return f'<{tag} class="math-source" data-display="{str(display).lower()}" data-tex="{html.escape(formula.strip(),quote=True).replace(chr(10), "&#10;")}">{html.escape(formula.strip())}</{tag}>'
    return ''.join(p if p.startswith('`') else re.sub(pattern,repl,p) for p in pieces)


def main():
    md=MarkdownIt('commonmark',{'html':True}).enable('table')
    chapters=sorted((BOOK/'chapters').glob('*.md'))
    inputs = build_inputs(chapters)
    titles={p.stem:p.read_text(encoding="utf-8").splitlines()[0].removeprefix('# ') for p in chapters}
    nav=''.join(f'<a href="#ch-{p.stem}" data-title="{html.escape(titles[p.stem])}">{html.escape(titles[p.stem])}</a>' for p in chapters)
    sections=[]; combined=[]; math_count=0
    for p in chapters:
        raw=p.read_text(encoding="utf-8")
        prepared=protect_math(raw)
        math_count += prepared.count('class="math-source"')
        rendered=md.render(prepared)
        def image_repl(match):
            src=match.group(1)
            file=(p.parent/src).resolve()
            if file.suffix=='.svg' and file.exists():
                svg=file.read_text(encoding="utf-8")
                svg=re.sub(r'<\?xml[^>]*\?>|<!DOCTYPE[^>]*\[[\s\S]*?\]>|<!DOCTYPE[^>]*>', '',svg)
                return f'<figure>{svg}</figure>'
            return match.group(0)
        rendered=re.sub(r'<img src="([^"]+)"[^>]*>',image_repl,rendered)
        rendered=re.sub(r'href="../../examples/([a-z_]+)\.py"',r'href="#code-\1"',rendered)
        rendered=re.sub(r'<h2>([a-z_]+)\.py</h2>',r'<h2 id="code-\1">\1.py</h2>',rendered)
        sections.append(f'<section class="chapter" id="ch-{p.stem}">{rendered}</section>')
        combined.append(raw.replace('](../assets/','](assets/').replace('](../../examples/','](../examples/'))
    vendor=BOOK/'assets/vendor'
    katex_css=(vendor/'katex.min.css').read_text(encoding="utf-8")
    katex_css=re.sub(r'url\(([^)]+)\)',lambda m:'url('+data_uri(vendor/m[1].strip('"\''))+')',katex_css)
    chinese=data_uri(vendor/'guide-chinese.woff2')
    emoji=data_uri(vendor/'guide-emoji.woff2')
    css='''
    :root { --ink:#193246; --muted:#607181; --accent:#087e8b; --paper:#fff; --bg:#edf2f6; }
    * { box-sizing:border-box; } html { scroll-behavior:smooth; }
    body { margin:0; color:var(--ink); background:var(--bg); font-family:GuideChinese,"Noto Sans CJK SC",GuideEmoji,sans-serif; font-size:16px; line-height:1.9; }
    aside { position:fixed; inset:0 auto 0 0; width:294px; background:#132b3a; color:white; padding:25px 18px; overflow:auto; }
    aside strong { font-size:22px; color:#e6f3fa; } aside p { color:#b8d2df; font-size:13px; line-height:1.6; }
    aside a { display:block; color:#d5e4ec; text-decoration:none; font-size:13px; line-height:1.6; padding:7px 9px; border-radius:5px; }
    aside a:hover { background:#285064; color:white; } aside a[hidden] {display:none;} aside input { font-family:inherit; width:100%; margin:14px 0; padding:10px; border:1px solid #506875; border-radius:5px; background:#203e4f; color:white; }
    main { margin-left:294px; max-width:1250px; padding:35px; }
    .cover,.chapter { background:white; padding:44px 52px; margin-bottom:26px; border:1px solid #dde6ed; border-radius:9px; }
    .cover { background:linear-gradient(135deg,#e9f7f4,#edf3fc); padding-top:70px; padding-bottom:70px; }
    .cover small { text-transform:uppercase; letter-spacing:3px; color:var(--accent); }
    .cover h1 { font-size:42px; line-height:1.35; border:0; margin:22px 0; }
    .cover p { max-width:760px; } .cover .meta { color:var(--muted); font-size:14px; }
    .cover a { display:inline-block; padding:8px 15px; border:1px solid var(--accent); border-radius:5px; margin:7px 8px 0 0; text-decoration:none; }
    h1 { font-size:29px; line-height:1.5; padding-bottom:16px; border-bottom:3px solid #b4dcd9; }
    h2 { font-size:22px; margin-top:37px; line-height:1.5; } h3 { font-size:18px; }
    a { color:#087487; text-underline-offset:3px; overflow-wrap:anywhere; }
    p { margin:17px 0; } strong { color:#163b51; }
    table { border-collapse:collapse; width:100%; font-size:14px; line-height:1.65; margin:23px 0; table-layout:auto; }
    th,td { border:1px solid #d3e0e8; padding:10px 12px; text-align:left; overflow-wrap:anywhere; }
    th { background:#eaf3f6; } tr:nth-child(even) td { background:#fafcfd; }
    pre { background:#142e3f; color:#e4edf3; padding:18px; border-radius:6px; overflow:auto; line-height:1.6; font-size:13px; tab-size:4; }
    code { font-family:ui-monospace,"DejaVu Sans Mono",Consolas,GuideChinese,GuideEmoji,monospace; }
    :not(pre)>code { background:#edf3f6; padding:2px 5px; border-radius:3px; color:#175870; font-size:.9em; overflow-wrap:anywhere; word-break:break-word; }
    figure { margin:26px 0; } figure svg { display:block; width:100%; height:auto; max-height:none; }
    figure svg text { font-family:GuideChinese,"DejaVu Sans",sans-serif !important; }
    .katex-display { overflow-x:auto; overflow-y:hidden; padding:8px 0; } .katex { font-size:1.08em; }
    .katex .cjk_fallback { font-family:GuideChinese,sans-serif; }
    .katex .hangul_fallback { font-family:GuideChinese,sans-serif; }
    .math-source { color:#21465a; } .toolbar { font-size:13px; color:var(--muted); }
    button { cursor:pointer; background:#087e8b; color:white; border:0; border-radius:4px; padding:10px 15px; font:inherit; }
    .print-toc { display:none; } section { scroll-margin-top:20px; }
    @media(max-width:1050px) { aside {width:242px;} main {margin-left:242px;padding:18px;} .chapter,.cover {padding:30px;} }
    @media(max-width:760px) { aside {position:relative;width:100%;max-height:320px;} main {margin:0;padding:10px;} .chapter,.cover {padding:23px 18px;} .cover h1 {font-size:32px;} body {font-size:15px;} table {font-size:12px;} }
    @page { size:A4; margin:18mm 16mm 19mm; }
    @media print {
      body {background:white;font-size:10pt;line-height:1.7;color:#152b3a;}
      aside,.toolbar,.cover a,.cover button {display:none!important;} main {margin:0;padding:0;max-width:none;}
      .cover,.chapter {border:0;border-radius:0;padding:0;margin:0;}
      .cover {min-height:245mm;background:white;padding-top:35mm;break-after:page;}
      .cover h1 {font-size:34pt;} .chapter {break-before:page;}
      h1 {font-size:22pt;} h2 {font-size:15pt;margin-top:24px;break-after:avoid;} h3 {font-size:12pt;break-after:avoid;}
      p,li {orphans:3;widows:3;} p {margin:12px 0;} pre {white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f7;color:#173043;font-size:7.7pt;padding:10px;line-height:1.5;orphans:4;widows:4;}
      table {font-size:8.5pt;line-height:1.6;} th,td {padding:5px 7px;} th[style*="right"],td[style*="right"] {white-space:nowrap;} thead {display:table-header-group;} tr {break-inside:avoid;}
      figure {break-inside:avoid;} .katex-display {font-size:.95em;overflow:visible;break-inside:avoid;}
      .print-toc {display:block;break-before:page;break-after:page;} .print-toc a {display:block;margin:3px 0;font-size:10pt;line-height:1.35;text-decoration:none;}
      a {text-decoration:none;} figure svg {max-height:155mm;} 
      #ch-07-alternative-attention p {margin:10px 0;}
      #ch-00-reading-map,#ch-01-objective,#ch-06-moe,#ch-13-scaling,#ch-35-sources {font-size:9.5pt;line-height:1.6;}
      #ch-00-reading-map p,#ch-01-objective p,#ch-06-moe p,#ch-13-scaling p,#ch-35-sources p {margin:8px 0;}
      #ch-00-reading-map h2,#ch-01-objective h2,#ch-06-moe h2,#ch-13-scaling h2,#ch-35-sources h2 {font-size:14pt;margin-top:18px;}
      #ch-00-reading-map {font-size:9.3pt;line-height:1.5;}
      #ch-00-reading-map h2 {margin-top:14px;}
      #ch-02-tokenization p,#ch-10-flashattention p,#ch-20-laboratory p {margin:9px 0;}
      #ch-02-tokenization h2,#ch-10-flashattention h2,#ch-20-laboratory h2 {margin-top:20px;}
      #ch-02-tokenization,#ch-27-interpretability {font-size:9.7pt;line-height:1.6;}
      #ch-02-tokenization p,#ch-27-interpretability p {margin:7px 0;}
      #ch-02-tokenization h2,#ch-27-interpretability h2 {margin-top:18px;}
      #ch-24-retrieval,#ch-25-test-time-compute {font-size:9.7pt;line-height:1.6;}
      #ch-24-retrieval p,#ch-25-test-time-compute p {margin:8px 0;}
      #ch-24-retrieval h2,#ch-25-test-time-compute h2 {margin-top:20px;}
      #ch-25-test-time-compute figure {margin:18px 0;}
      #ch-25-test-time-compute figure svg {max-height:115mm;}
    }
    '''
    cover=f'''<div class="cover"><small>CS336 · 2024 / 2025 / 2026</small><h1>从零构建大语言模型<br>中文深度学习指南</h1><p>为已经掌握深度学习的读者，把模型、系统、数据、评估与后训练连成一条可推导、可实现、可验证的学习路径。</p><p class="meta">{len(chapters)} 章 · 原创图解 · 完整教学代码 · 逐年来源映射<br>资料核验：2026-10-26 · 独立编写，非 Stanford 官方教材</p><p>阅读无需先观看课程视频。公开核心讲义与作业是内容依据；资料范围见第 35 章，五组作业的实战导读见第 28 章。</p><a href="#ch-00-reading-map">开始阅读</a><a href="#ch-20-laboratory">进入实验</a><a href="#ch-28-assignments">作业导读</a><button onclick="window.print()">打印 / 保存 PDF</button><p class="toolbar">本文件包含中文字体、公式与插图，可离线阅读。按 Ctrl/Cmd + F 全文检索。</p></div>'''
    script='''
    document.querySelectorAll('.math-source').forEach(node => {
      try { katex.render(node.dataset.tex,node,{displayMode:node.dataset.display==='true',throwOnError:true,strict:'ignore',trust:false}); }
      catch(error) { node.classList.add('math-error'); node.title=error.message; console.error(error.message); }
    });
    document.getElementById('toc-filter').addEventListener('input', event => {
      const q=event.target.value.trim().toLowerCase();
      document.querySelectorAll('aside nav a').forEach(a=>a.hidden=!a.dataset.title.toLowerCase().includes(q));
    });
    window.BOOK_READY=true;
    '''
    doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CS336 中文深度学习指南｜2024–2026</title><style>@font-face{{font-family:GuideChinese;src:url({chinese}) format('woff2');font-weight:100 900;font-style:normal;}}@font-face{{font-family:GuideEmoji;src:url({emoji}) format('woff2');font-weight:100 900;font-style:normal;}}{katex_css}{css}</style></head><body><aside><strong>CS336 学习指南</strong><p>从零构建大语言模型<br>2024–2026 公开核心课程</p><input id="toc-filter" aria-label="筛选章节" placeholder="筛选章节标题"><nav>{nav}</nav></aside><main>{cover}<div class="print-toc"><h1>目录</h1>{nav}</div>{''.join(sections)}</main><script>{(vendor/'katex.min.js').read_text(encoding="utf-8")}</script><script>{script}</script></body></html>'''
    html_bytes = doc.encode('utf-8')
    markdown_bytes = ('# 从零构建大语言模型：CS336 中文深度学习指南\n\n2024–2026 公开核心课程综合版，核验日期 2026-10-26。\n\n'+'\n\n---\n\n'.join(combined)).encode('utf-8')
    # Hash exactly the bytes written, including on platforms whose text mode translates newlines.
    stats=dict(chapters=len(chapters),source_characters=sum(len(p.read_text(encoding="utf-8")) for p in chapters),math_expressions=math_count,html_bytes=len(doc.encode()),
               html_sha256=sha256(html_bytes).hexdigest(),markdown_sha256=sha256(markdown_bytes).hexdigest(),
               chapters_sha256=chapters_digest(chapters),inputs_sha256=inputs)
    if build_inputs(chapters) != inputs:
        raise RuntimeError('build inputs changed during rendering; retry from a stable source snapshot')
    with tempfile.TemporaryDirectory(prefix='.build-', dir=BOOK) as directory:
        staging = Path(directory)
        outputs = {'guide.html':html_bytes, '学习指南.md':markdown_bytes,
                   'build-stats.json':json.dumps(stats,indent=2).encode('utf-8')}
        for name, content in outputs.items():
            (staging/name).write_bytes(content)
        publish_files([(staging/name, BOOK/name) for name in outputs])
    print(json.dumps({key:value for key,value in stats.items() if key!='inputs_sha256'}))


if __name__=='__main__':main()

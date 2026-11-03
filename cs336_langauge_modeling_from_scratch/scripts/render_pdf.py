"""Validate the offline HTML in Chromium, then export an A4 PDF.

Optional build dependency: playwright plus its Chromium browser.
"""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import os
import tempfile
if __package__:
    from .audit_pdf import add_proof_arguments, audit, display_path, open_pdf
    from .artifact_io import publish_files
else:
    from audit_pdf import add_proof_arguments, audit, display_path, open_pdf
    from artifact_io import publish_files

ROOT = Path(__file__).resolve().parents[1]


def render_book(page, proof_dir, *, pages=None, contact_sheets=False, preview=False):
    """Publish the PDF and reports after validation; screenshots require preview=True."""
    book = ROOT / 'book'
    html_digest = sha256((book/'guide.html').read_bytes()).hexdigest()
    chapter_snapshot = {path.name:path.read_bytes() for path in (book/'chapters').glob('*.md')}
    with tempfile.TemporaryDirectory(prefix='.render-', dir=book) as directory:
        output = Path(directory)
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto((ROOT/'book/guide.html').as_uri(),wait_until='load')
        page.wait_for_function('window.BOOK_READY === true')
        page.evaluate('document.fonts.ready')
        result=page.evaluate('''() => ({
            chapters:document.querySelectorAll('section.chapter').length,
            math:document.querySelectorAll('.math-source').length,
            mathErrors:[...document.querySelectorAll('.math-error')].map(n=>({tex:n.dataset.tex,error:n.title})),
            missingAnchors:[...document.querySelectorAll('a[href^="#"]')].filter(a=>!document.getElementById(a.getAttribute('href').slice(1))).map(a=>a.getAttribute('href')),
            externalAssets:[...document.querySelectorAll('[src]')].map(n=>n.getAttribute('src')).filter(x=>x.startsWith('http')),
            width:document.documentElement.scrollWidth,
            viewport:innerWidth
        })''')
        result['pageErrors']=errors
        result['htmlSha256']=html_digest
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        if result['mathErrors'] or result['missingAnchors'] or errors or result['externalAssets'] or result['width'] > result['viewport']:
            raise RuntimeError('HTML validation failed')
        if preview:
            page.screenshot(path=str(output/'preview.png'),full_page=False)
        page.emulate_media(media='print')
        # A4 content width after 16mm side margins, in CSS pixels.
        page.set_viewport_size({'width':673,'height':1123})
        page.evaluate('document.fonts.ready')
        result['printOverflows']=page.evaluate('''() => [...document.querySelectorAll('main table,main pre,main .katex-display')]
            .filter(n=>n.scrollWidth>n.clientWidth+2).map(n=>({tag:n.tagName,width:n.clientWidth,scroll:n.scrollWidth,text:n.innerText.slice(0,100)}))''')
        if result['printOverflows']:
            raise RuntimeError('print content exceeds its column: '+str(result['printOverflows']))
        page.pdf(path=str(output/'CS336中文学习指南.pdf'),format='A4',print_background=True,
                 prefer_css_page_size=True,display_header_footer=True,
                 header_template='<div></div>',
                 footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#64748b;">CS336 · 2024–2026 &nbsp; <span class="pageNumber"></span> / <span class="totalPages"></span></div>')
        pdf_path=output/'CS336中文学习指南.pdf'
        with open_pdf(pdf_path) as pdf:
            headings=[path.read_text(encoding="utf-8").splitlines()[0].removeprefix('# ') for path in sorted((ROOT/'book/chapters').glob('*.md'))]
            toc=[]
            first_chapter=next(i for i in range(2,len(pdf)) if pdf[i].get_text().lstrip().startswith('00｜'))
            for heading in headings:
                # Skip the cover and printed contents; chapter headings start the page.
                prefix=heading.split('｜')[0]+'｜'
                for i in range(first_chapter,len(pdf)):
                    if pdf[i].get_text().lstrip().startswith(prefix):
                        toc.append([1,heading,i+1]);break
            pdf.set_toc(toc)
            metadata=pdf.metadata
            metadata.update(title='从零构建大语言模型：CS336 中文深度学习指南（2024–2026）',subject='CS336 核心知识、原创图解、数学推导与教学代码')
            pdf.set_metadata(metadata)
            # Write through Python as well (see open_pdf): MuPDF cannot address non-UTF-8 file names itself.
            pdf_path.write_bytes(pdf.tobytes(garbage=0, deflate=False))
            result['pdfPages']=len(pdf)
            result['pdfBookmarks']=len(toc)
            result['pdfOutsidePageText']=[]
            for i,pdf_page in enumerate(pdf):
                for block in pdf_page.get_text('blocks'):
                    if block[0]<-1 or block[2]>pdf_page.rect.width+1 or block[1]<-1 or block[3]>pdf_page.rect.height+1:
                        result['pdfOutsidePageText'].append(i+1)
            result['pdfOutsidePageText']=sorted(set(result['pdfOutsidePageText']))
        result['pdfSha256']=sha256(pdf_path.read_bytes()).hexdigest()
        if len(toc)!=len(headings) or result['pdfOutsidePageText']:
            raise RuntimeError('PDF bookmark or page-boundary validation failed')
        glyph_audit=audit(pdf_path,output/'pdf-audit.json',output/'proof',
                          pages=pages,contact_sheets=contact_sheets)
        result['glyphAudit']={key:glyph_audit[key] for key in ['pages','rasterized_pages','exported_pages','passed','missing_glyphs','replacement_or_control_characters','blank_pages','sparse_pages']}
        if not result['glyphAudit']['passed']:
            problems={key:glyph_audit[key] for key in ('missing_glyphs','replacement_or_control_characters',
                       'outside_page','blank_pages','unembedded_fonts') if glyph_audit[key]}
            raise RuntimeError('PDF glyph audit failed: '+str(problems))
        page.emulate_media(media='screen')
        page.set_viewport_size({'width':390,'height':844})
        page.goto((ROOT/'book/guide.html').as_uri(),wait_until='load')
        page.wait_for_function('window.BOOK_READY === true')
        page.evaluate('document.fonts.ready')
        mobile=page.evaluate('({width:document.documentElement.scrollWidth,viewport:innerWidth})')
        result['mobile']=mobile
        if mobile['width']>mobile['viewport']:
            raise RuntimeError('mobile horizontal overflow')
        page.locator('#toc-filter').fill('FlashAttention')
        result['filteredTocEntries']=page.locator('aside nav a:visible').count()
        if result['filteredTocEntries']!=1:
            raise RuntimeError('TOC filtering failed')
        if errors:
            raise RuntimeError('HTML page errors: '+str(errors))

        proof_dir=Path(proof_dir).resolve()
        proof_names=[f'page-{number:03d}.png' for number in glyph_audit['exported_pages']]
        proof_names.extend(Path(name).name for name in glyph_audit['contact_sheets'])
        glyph_audit['file']=display_path(book/pdf_path.name)
        glyph_audit['contact_sheets']=[display_path(proof_dir/Path(name).name)
                                       for name in glyph_audit['contact_sheets']]
        (output/'pdf-audit.json').write_text(json.dumps(glyph_audit,ensure_ascii=False,indent=2), encoding="utf-8")
        (output/'render-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2), encoding="utf-8")
        artifacts=[(pdf_path.name,pdf_path.name),('pdf-audit.json','pdf-audit.json'),('render-check.json','render-check.json')]
        if preview:
            artifacts.insert(0,('preview.png','assets/preview.png'))
        if sha256((book/'guide.html').read_bytes()).hexdigest() != html_digest or chapter_snapshot != {
            path.name:path.read_bytes() for path in (book/'chapters').glob('*.md')
        }:
            raise RuntimeError('HTML or chapters changed during PDF export; original outputs were retained')
        publish_files([(output/source, book/destination) for source,destination in artifacts]
                      + [(output/'proof'/name, proof_dir/name) for name in proof_names])
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_proof_arguments(parser)
    parser.add_argument('--preview',action='store_true',help='also capture and update the README preview screenshot')
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        # CHROMIUM_EXECUTABLE lets a preinstalled browser be used when the bundled one is absent.
        executable = os.environ.get('CHROMIUM_EXECUTABLE')
        # Keep Chromium's sandbox on: the book is local, but there is no reason to drop the protection.
        # Containers without user namespaces can opt out explicitly with CHROMIUM_NO_SANDBOX=1.
        sandbox = os.environ.get('CHROMIUM_NO_SANDBOX') != '1'
        browser = p.chromium.launch(headless=True, chromium_sandbox=sandbox, executable_path=executable)
        try:
            page = browser.new_page(viewport={'width':1440,'height':1050},device_scale_factor=1)
            render_book(page,ROOT/args.proof_dir,pages=args.proof_pages,contact_sheets=args.contact_sheets,preview=args.preview)
        finally:
            browser.close()


if __name__=='__main__':main()

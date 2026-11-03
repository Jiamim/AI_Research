"""Audit every PDF page; export page images and contact sheets only on request."""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import pymupdf
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def display_path(path):
    """Keep repository paths portable while allowing temporary output directories."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_pages(value):
    """Keep page ranges compact until they can be checked against the PDF length."""
    pages = []
    try:
        for item in value.split(','):
            bounds = item.strip().split('-')
            if len(bounds) not in (1, 2):
                raise ValueError
            first, last = int(bounds[0]), int(bounds[-1])
            if first < 1 or last < first:
                raise ValueError
            pages.append(range(first, last + 1))
    except ValueError as error:
        raise argparse.ArgumentTypeError('use positive page numbers/ranges, e.g. 7,12-14') from error
    return tuple(pages)


def select_pages(pages, page_count, contact_sheets):
    if pages is None:
        return range(1, page_count + 1) if contact_sheets else ()
    selected = set()
    message = f'proof pages must be within 1..{page_count}'
    for item in pages:
        if isinstance(item, range):
            if item.step != 1 or not 1 <= item.start < item.stop <= page_count + 1:
                raise ValueError(message)
            selected.update(item)
        elif type(item) is int and 1 <= item <= page_count:
            selected.add(item)
        else:
            raise ValueError(message)
    if not selected:
        raise ValueError(message)
    return selected


def add_proof_arguments(parser):
    parser.add_argument('--proof-dir', default='sources/cache/pdf-proof')
    parser.add_argument('--proof-pages', type=parse_pages, metavar='N,N-M',
                        help='export only these page images; all pages are still audited')
    parser.add_argument('--contact-sheets', action='store_true',
                        help='also export contact sheets (all pages unless --proof-pages is set)')


def open_pdf(path):
    """Open via Python's file API: MuPDF's own path handling assumes UTF-8 names, which fails for the
    Chinese file name on systems whose filesystem encoding is GBK/cp936 (Chinese Windows locales)."""
    return pymupdf.open(stream=Path(path).read_bytes(), filetype='pdf')


def audit(path, report, proof_dir=None, *, pages=None, contact_sheets=False):
    path, report = Path(path).resolve(), Path(report)
    with open_pdf(path) as pdf:
        selected = select_pages(pages, len(pdf), contact_sheets)
        if selected:
            if proof_dir is None:
                raise ValueError('a proof directory is required to export images')
            proof_dir = Path(proof_dir).resolve()
            proof_dir.mkdir(parents=True, exist_ok=True)
        return audit_pages(pdf, path, report, proof_dir, selected, contact_sheets)


def save_contact_sheet(batch, proof_dir, number):
    from PIL import Image, ImageDraw
    sheet = Image.new('RGB', (4 * 355, 3 * 505), '#dbe3ea')
    draw = ImageDraw.Draw(sheet)
    for index, (page, thumbnail) in enumerate(batch):
        x, y = (index % 4) * 355 + 10, (index // 4) * 505 + 24
        sheet.paste(thumbnail, (x, y))
        draw.text((x, y - 18), f'Page {page}', fill='#172f41')
    target = proof_dir / f'contact-{number:02d}.png'
    sheet.save(target)
    return display_path(target)


def audit_pages(pdf, path, report, proof_dir, selected, contact_sheets):
    # Import image encoders only when exporting proof images.
    if selected:
        from PIL import Image
    result=dict(file=display_path(path),sha256=sha256(path.read_bytes()).hexdigest(),pages=len(pdf),
                missing_glyphs=[],replacement_or_control_characters=[],outside_page=[],
                blank_pages=[],sparse_pages=[],fonts=[],rasterized_pages=0,exported_pages=[],contact_sheets=[])
    font_ids=set();thumbnails=[]
    for number,page in enumerate(pdf,1):
        for font in page.get_fonts(full=True):font_ids.add(font[0])
        for span in page.get_texttrace():
            for codepoint,glyph,origin,box in span['chars']:
                char=chr(codepoint)
                if glyph==0 and not char.isspace():
                    result['missing_glyphs'].append(dict(page=number,codepoint=hex(codepoint),font=span['font'],box=box))
                if codepoint in (0,0xfffd) or (codepoint<32 and char not in '\n\t\r'):
                    result['replacement_or_control_characters'].append(dict(page=number,codepoint=hex(codepoint),font=span['font']))
                if box[0]<-1 or box[1]<-1 or box[2]>page.rect.width+1 or box[3]>page.rect.height+1:
                    result['outside_page'].append(dict(page=number,box=box))
        text=page.get_text()
        if number>1 and len(text.strip())<180:result['sparse_pages'].append(dict(page=number,characters=len(text.strip())))
        # Render one page at a time for blank-page detection, without saving images.
        pix=page.get_pixmap(matrix=pymupdf.Matrix(1.25,1.25),alpha=False,colorspace=pymupdf.csRGB)
        rgb=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,3)
        # Exclude the footer so a page containing only a page number is reported.
        crop=rgb[30:-65,30:-30]
        if not crop.size:crop=rgb
        ink=float(np.mean(np.min(crop,axis=-1)<210))
        if ink<.0005:result['blank_pages'].append(number)
        if number in selected:
            image=Image.fromarray(rgb)
            image.save(proof_dir/f'page-{number:03d}.png')
            result['exported_pages'].append(number)
            if contact_sheets:
                image.thumbnail((335,475))
                thumbnails.append((number,image))
                if len(thumbnails) == 12:
                    result['contact_sheets'].append(save_contact_sheet(
                        thumbnails, proof_dir, len(result['contact_sheets']) + 1))
                    thumbnails.clear()
        result['rasterized_pages']+=1
    if thumbnails:
        result['contact_sheets'].append(save_contact_sheet(
            thumbnails, proof_dir, len(result['contact_sheets']) + 1))
        thumbnails.clear()
    for xref in sorted(font_ids):
        subtype=pdf.xref_get_key(xref,'Subtype')[1]
        name,ext,kind,raw=pdf.extract_font(xref)
        # Type3 fonts embed glyph programs directly instead of a FontFile stream.
        charprocs=pdf.xref_get_key(xref,'CharProcs')
        embedded=bool(raw) or (subtype=='/Type3' and charprocs[0] in ('dict','xref'))
        result['fonts'].append(dict(xref=xref,name=name or '(embedded Type3 glyph programs)',subtype=subtype,embedded=embedded,bytes=len(raw)))
    result['unembedded_fonts']=[f for f in result['fonts'] if not f['embedded']]
    result['passed']=not any(result[k] for k in ['missing_glyphs','replacement_or_control_characters','outside_page','blank_pages','unembedded_fonts'])
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(result,ensure_ascii=False,indent=2), encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pdf',default='book/CS336中文学习指南.pdf')
    parser.add_argument('--report',default='book/pdf-audit.json');add_proof_arguments(parser)
    args=parser.parse_args()
    result=audit(ROOT/args.pdf,ROOT/args.report,ROOT/args.proof_dir,
                 pages=args.proof_pages,contact_sheets=args.contact_sheets)
    print(json.dumps({k:result[k] for k in ['pages','rasterized_pages','exported_pages','passed','missing_glyphs','replacement_or_control_characters','outside_page','blank_pages','sparse_pages','unembedded_fonts']},ensure_ascii=False,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()

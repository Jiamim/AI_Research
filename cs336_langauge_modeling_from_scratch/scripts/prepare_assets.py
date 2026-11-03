"""Download licensed renderer/font assets and subset the Chinese font for this book.

Optional build dependency: fonttools[woff]. Assets are embedded into guide.html.
"""
from pathlib import Path
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor
import re
from fontTools import subset

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'book/assets/vendor'
CACHE=ROOT/'sources/cache'


def fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        with urlopen(url,timeout=60) as r:path.write_bytes(r.read())


def woff2_only(css):
    """Keep the WOFF2 source in KaTeX font-face rules; retain the upstream license."""
    def simplify(match):
        alternatives=match.group(1).split(',')
        sources=[part for part in alternatives if "format(\"woff2\")" in part or "format('woff2')" in part]
        return 'src:'+','.join(sources if sources else alternatives)
    return re.sub(r'src:([^;}]+)',simplify,css)


def main():
    base='https://cdn.jsdelivr.net/npm/katex@0.16.22/'
    files={'katex.min.js':'dist/katex.min.js','katex.min.css':'dist/katex.min.css','KATEX-LICENSE.txt':'LICENSE'}
    for local,remote in files.items():fetch(base+remote,OUT/local)
    css_path=OUT/'katex.min.css'
    css_path.write_text(woff2_only(css_path.read_text(encoding="utf-8")), encoding="utf-8")
    font_urls=set(re.findall(r'url\(([^)]+)\)',(OUT/'katex.min.css').read_text(encoding="utf-8")))
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda name:fetch(base+'dist/'+name,OUT/name),font_urls))
    source=CACHE/'NotoSansCJKsc-Regular.otf'
    fetch('https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf',source)
    fetch('https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/LICENSE',OUT/'NOTO-LICENSE.txt')
    chars=''.join(p.read_text(encoding="utf-8") for p in (ROOT/'book/chapters').glob('*.md'))
    chars+=''.join(p.read_text(encoding="utf-8") for p in (ROOT/'book/assets').glob('*.svg'))
    chars+=(ROOT/'scripts/build_book.py').read_text(encoding="utf-8")
    options=subset.Options();options.flavor='woff2'
    font=subset.load_font(str(source),options)
    s=subset.Subsetter(options=options);s.populate(text=chars);s.subset(font)
    subset.save_font(font,str(OUT/'guide-chinese.woff2'),options)
    emoji_source=CACHE/'NotoEmoji.ttf'
    fetch('https://raw.githubusercontent.com/google/fonts/main/ofl/notoemoji/NotoEmoji%5Bwght%5D.ttf',emoji_source)
    fetch('https://raw.githubusercontent.com/google/fonts/main/ofl/notoemoji/OFL.txt',OUT/'NOTO-EMOJI-LICENSE.txt')
    emoji=subset.load_font(str(emoji_source),options)
    es=subset.Subsetter(options=options);es.populate(text=chars);es.subset(emoji)
    subset.save_font(emoji,str(OUT/'guide-emoji.woff2'),options)
    print('Prepared KaTeX, Chinese and monochrome emoji font subsets.')


if __name__=='__main__':main()

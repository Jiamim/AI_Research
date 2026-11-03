"""Verify cached official originals against the pinned manifest; optionally fetch missing."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen
import argparse
import json
try:
    from scripts.artifact_io import atomic_output, file_sha256
except ModuleNotFoundError:
    from artifact_io import atomic_output, file_sha256

ROOT=Path(__file__).resolve().parents[1]


def check_source(document, *, download=False):
    """A failed/mismatched fetch never creates a valid-looking cache entry."""
    try:
        url=document['download_url']
        expected=document['sha256']
        ext='.pdf' if url.endswith('.pdf') else '.py' if url.endswith('.py') else '.html'
        path=ROOT/'sources/cache'/f"{document['id']}{ext}"
        if not path.exists() and download:
            digest=sha256()
            with urlopen(url,timeout=60) as response, atomic_output(path) as stream:
                for block in iter(lambda:response.read(1 << 20),b''):
                    digest.update(block)
                    stream.write(block)
                if digest.hexdigest()!=expected:
                    raise ValueError('download hash mismatch; upstream contents differ from pinned source')
        if not path.exists():return document['id']+': missing'
        if file_sha256(path)!=expected:return document['id']+': hash mismatch'
        return None
    except Exception as error:
        return str(document.get('id','unknown'))+': '+type(error).__name__+': '+str(error)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'sources/manifest.json').read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=8) as ex:
        errors=[r for r in ex.map(lambda d:check_source(d,download=args.download),manifest['documents']) if r]
    print(json.dumps(dict(records=len(manifest['documents']),errors=errors),indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()

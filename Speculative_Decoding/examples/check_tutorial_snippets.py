"""Execute only explicitly marked, maintained tutorial snippets in isolation.

Run: python -m examples.check_tutorial_snippets
Does not execute unmarked code blocks or downloaded research sources.
"""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import argparse
import json
import re
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/投机解码深度调研与实践.md'
PATTERN=re.compile(r'<!-- runnable: ([a-z0-9_]+) -->\s*```python\n(.*?)\n```',re.S)


def check(source=SOURCE):
    results=[]
    found=PATTERN.findall(source.read_text(encoding='utf-8'))
    if not found or len({name for name,_ in found})!=len(found):
        raise ValueError('Need uniquely named executable snippets')
    for name,code in found:
        output=StringIO()
        start=time.perf_counter()
        try:
            with redirect_stdout(output):
                exec(compile(code,f'{source.name}:{name}','exec'),{'__name__':'__tutorial_snippet__'})
        except Exception as error:
            raise RuntimeError(f'Tutorial snippet {name} failed') from error
        results.append(dict(name=name,status='passed',seconds=time.perf_counter()-start,
                            output=output.getvalue()))
    return dict(source=str(source.relative_to(ROOT)),status='passed',count=len(found),snippets=results)


def check_all():
    """Run all explicitly marked examples in the maintained documentation."""
    sources = [p for p in sorted((ROOT/'docs').glob('*.md'))
               if PATTERN.search(p.read_text(encoding='utf-8'))]
    reports = [check(source) for source in sources]
    return dict(status='passed', count=sum(r['count'] for r in reports), documents=reports)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'build/snippets.json')
    parser.add_argument('--source',type=Path,help='Run one document instead of all marked local tutorials')
    args=parser.parse_args()
    import torch
    torch.set_num_threads(1)
    result=check(args.source.resolve()) if args.source else check_all()
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f"Executed {result['count']} tutorial snippets: all passed.")

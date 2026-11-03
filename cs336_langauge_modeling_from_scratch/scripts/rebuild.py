"""Rebuild and validate the book in dependency order without taking screenshots."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.artifact_io import runtime_snapshot, validation_snapshot, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--without-pdf', action='store_true', help='explicitly leave the previous PDF; do not report full acceptance')
    parser.add_argument('--skip-checks', action='store_true', help='use the existing validation record; full acceptance requires it to pass')
    parser.add_argument('--allow-failed-environment-checks', action='store_true',
                        help='still regenerate the book when unit tests pass but the distributed/Triton subprocess checks fail '
                             '(e.g. a sandbox that forbids sockets); the book will then state that those checks failed')
    args = parser.parse_args()
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), stages=[], success=False, complete=False)
    report_path = ROOT/'book/rebuild-check.json'
    logs = ROOT/'outputs/build-logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PYTHONIOENCODING':'utf-8', 'PYTHONDONTWRITEBYTECODE':'1'}
    write_json(report_path,report)

    def run(name, command):
        path = logs/(name+'.log')
        print('Running '+name, flush=True)
        try:
            with path.open('wb') as log:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            status = dict(name=name, success=process.returncode==0, exit_code=process.returncode,
                          log=str(path.relative_to(ROOT)))
        except OSError as error:
            status = dict(name=name, success=False, error=str(error), log=str(path.relative_to(ROOT)))
        report['stages'].append(status)
        write_json(report_path,report)
        print(name+(': passed' if status['success'] else ': failed; see '+str(path.relative_to(ROOT))), flush=True)
        return status['success']

    semantic_ok = None
    if not args.skip_checks:
        semantic_ok=run('semantic-checks',[sys.executable,'scripts/run_checks.py'])
    validation_path=ROOT/'book/validation.json'
    try:
        validation=json.loads(validation_path.read_text(encoding='utf-8')) if validation_path.exists() else {}
    except (OSError, ValueError) as error:
        report['stages'].append(dict(name='semantic-checks',success=False,error='unreadable validation record: '+str(error)))
        write_json(report_path,report)
        raise SystemExit('Validation record is unreadable; run without --skip-checks') from error
    required_counts=('tests_run','failures','errors','skipped')
    if not isinstance(validation,dict) or any(type(validation.get(key)) is not int or validation[key]<0 for key in required_counts) \
            or validation['tests_run']<=validation['skipped'] or type(validation.get('success')) is not bool:
        report['stages'].append(dict(name='semantic-checks',success=False,error='missing or empty semantic test results'))
        write_json(report_path,report)
        raise SystemExit('Validation must contain results from actual semantic tests')
    if validation.get('source_sha256') != validation_snapshot(ROOT) or validation.get('runtime') != runtime_snapshot() \
            or (semantic_ok is False and validation.get('success')):
        report['stages'].append(dict(name='semantic-checks',success=False,error='source or runtime differs from validation; run without --skip-checks'))
        write_json(report_path,report)
        raise SystemExit('No fresh validation for the current source/runtime; run without --skip-checks')
    if not validation or validation.get('failures') or validation.get('errors'):
        print('Semantic checks failed or missing; rebuild stopped.',flush=True)
        raise SystemExit(1)
    # Unit tests passed but an environment check (two-process Gloo, Triton interpreter) did not. Chapter 35
    # embeds this record, so regenerating now would publish "check failed" into the book. In a sandbox that
    # forbids sockets this is an environment limit, not a code defect: stop unless explicitly allowed.
    if validation.get('success') is False and not args.allow_failed_environment_checks:
        failed=[name for name in ('distributed','triton_interpreter')
                if isinstance(validation.get(name),dict) and validation[name].get('success') is False]
        report['stages'].append(dict(name='environment-checks',success=False,failed=failed,
                                     error='unit tests passed but environment checks failed; book not regenerated'))
        write_json(report_path,report)
        print('Unit tests passed, but environment checks failed: '+(', '.join(failed) or 'see book/validation.json')
              +'. The book was not regenerated, so it will not claim these checks failed. Rerun where the checks can run, '
              'or pass --allow-failed-environment-checks to publish the failure.',flush=True)
        raise SystemExit(1)
    for name in ('make_appendices','make_figures','prepare_assets','build_book'):
        if not run(name,[sys.executable,f'scripts/{name}.py']):raise SystemExit(1)
    if not run('text-check',[sys.executable,'scripts/check_book.py','--without-pdf']):raise SystemExit(1)
    if args.without_pdf:
        report['stages'].append(dict(name='render_pdf',success=None,status='not_run',reason='--without-pdf'))
        pdf_ok=False
    else:
        pdf_ok=run('render_pdf',[sys.executable,'scripts/render_pdf.py'])
        if pdf_ok:pdf_ok=run('artifact-check',[sys.executable,'-m','unittest','tests.test_artifacts'])
    report['text_ready']=True
    stable=validation.get('source_sha256')==validation_snapshot(ROOT) and validation.get('runtime')==runtime_snapshot()
    if not stable:
        report['stages'].append(dict(name='validation-freshness',success=False,error='source/runtime changed during rebuild'))
    report['validation_success']=bool(validation.get('success') and stable)
    report['success']=bool(pdf_ok and report['validation_success'])
    report['complete']=report['success']
    write_json(report_path,report)
    print(json.dumps({k:report[k] for k in ('text_ready','validation_success','success','complete')},ensure_ascii=False),flush=True)
    if not report['complete']:raise SystemExit(1)


if __name__=='__main__':main()

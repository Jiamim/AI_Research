"""Run semantic checks and write a machine-generated validation record for the book."""
from datetime import datetime, timezone
from pathlib import Path
import json
import importlib.util
import os
import platform
import subprocess
import sys
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.artifact_io import runtime_snapshot, validation_snapshot, write_json


def semantic_tests(suite):
    """Exclude artifact tests by module, regardless of unittest's suite nesting."""
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from semantic_tests(test)
        elif test.__class__.__module__.split('.')[-1] != 'test_artifacts':
            yield test


def subprocess_check(command, *, cwd, env, timeout):
    """Return failures as data, including startup errors and partial timeout output."""
    def text(value):
        return (value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value or '').strip()
    try:
        process = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True,
                                 encoding='utf-8', errors='replace', timeout=timeout)
    except subprocess.TimeoutExpired as error:
        return dict(success=False, stdout=text(error.stdout),
                    stderr=(text(error.stderr) + f'\nTimed out after {timeout} seconds').strip(), timed_out=True)
    except OSError as error:
        return dict(success=False, stdout='', stderr=f'{type(error).__name__}: {error}')
    return dict(success=process.returncode == 0, stdout=process.stdout.strip(), stderr=process.stderr.strip())


def main():
    # test_artifacts checks that the generated book matches the sources. It must run after the
    # rebuild that consumes this validation record, so it is excluded here and run last instead.
    snapshot=validation_snapshot(ROOT)
    runtime=runtime_snapshot()
    suite=unittest.TestSuite(semantic_tests(unittest.defaultTestLoader.discover(str(ROOT/'tests'))))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    record=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),
                torch=str(torch.__version__),device='cpu',tests_run=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),success=result.wasSuccessful())
    if result.testsRun == len(result.skipped):
        record['success']=False
        record['reason']='no semantic tests were executed'
    if record['success']:
        # Gloo binds to the address the hostname resolves to; on hosts where that fails it cannot even use
        # 127.0.0.1. Pin the loopback interface unless the caller already chose one.
        # Children write UTF-8 so their captured output decodes identically on Windows (GBK/cp1252) consoles.
        env={**os.environ,'PYTHONIOENCODING':'utf-8'}
        if 'GLOO_SOCKET_IFNAME' not in env and platform.system()=='Linux':env['GLOO_SOCKET_IFNAME']='lo'
        record['distributed']=subprocess_check([sys.executable,'-m','examples.distributed'],cwd=ROOT,env=env,timeout=120)
        record['success'] &= record['distributed']['success']
        # The Triton kernel test is skipped by default; rerun it on CPU through the interpreter.
        if importlib.util.find_spec('triton') is None:
            record['triton_interpreter']=dict(status='skipped',success=None,reason='optional triton package is not installed')
        else:
            triton=subprocess_check([sys.executable,'-m','unittest','test_course.AttentionTests.test_triton_softmax_in_interpreter',
                                   'test_regressions.TritonRegressionTests'],
                                  cwd=ROOT/'tests',env={**os.environ,'TRITON_INTERPRET':'1','PYTHONPATH':str(ROOT),'PYTHONIOENCODING':'utf-8'},
                                  timeout=300)
            ok=triton['success'] and 'skipped' not in triton['stderr']
            record['triton_interpreter']=dict(status='passed' if ok else 'failed',success=ok,output=triton['stderr'])
            record['success'] &= ok
    current=validation_snapshot(ROOT)
    record['source_sha256']=snapshot
    record['runtime']=runtime
    if current!=snapshot:
        record['changed_during_validation']=sorted(name for name in snapshot.keys()|current.keys() if snapshot.get(name)!=current.get(name))
        record['success']=False
    if runtime_snapshot()!=runtime:
        record['runtime_changed_during_validation']=True
        record['success']=False
    write_json(ROOT/'book/validation.json', record)
    if not record['success']:raise SystemExit(1)


if __name__=='__main__':main()

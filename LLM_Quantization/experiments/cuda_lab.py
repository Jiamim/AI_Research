#!/usr/bin/env python3
"""从 Colab SSH 运行 T4/A100 CUDA 实验：默认只打印，显式保存到个人运行目录。"""
import argparse
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from experiments.cuda_cases import CASES, DESCRIPTIONS
from experiments.cuda_runtime import CudaBench, PersistenceError, RunReport, Unsupported, environment, require_cuda, skipped


def run_status(results, errors):
    """路径级 failed 行与实验级异常同等对待；只要有失败就不是 completed。"""
    statuses = {r['status'] for r in results}
    if errors or 'failed' in statuses:
        return 'partial' if 'passed' in statuses else 'failed'
    if 'passed' not in statuses:
        return 'skipped'
    return 'completed_with_skips' if 'skipped' in statuses else 'completed'



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--case',choices=['all',*CASES],nargs='+',default=['all'],
                        help='One or more cases; all runs every case in the listed order')
    parser.add_argument('--profile',choices=['smoke','full'],default='smoke')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--write-results',action='store_true')
    parser.add_argument('--output-dir',type=Path,help='Parent for a new timestamped run directory; requires --write-results')
    parser.add_argument('--trace',action='store_true',help='Save one profiler trace per measured variant, outside timed samples')
    args=parser.parse_args()
    if not __debug__:
        parser.error('Run without -O; correctness assertions must remain enabled')
    if args.output_dir is not None and not args.write_results:
        parser.error('--output-dir requires --write-results')
    if args.trace and not args.write_results:
        parser.error('--trace requires --write-results')
    if args.list:
        for name in CASES:
            print(f'{name:13s} {DESCRIPTIONS[name]}')
        return 0
    parameters={**vars(args),'output_dir':str(args.output_dir) if args.output_dir else None}
    report=RunReport(parameters,(args.output_dir or ROOT/'runs/cuda') if args.write_results else None)
    if report.directory:
        print('本轮结果目录：',report.directory,flush=True)
    selected=list(CASES) if 'all' in args.case else list(dict.fromkeys(args.case))
    report.data['selected_cases']=selected
    report.save()
    try:
        device=require_cuda(args.device)
        report.data['environment']=environment(device)
        report.save()
        print('GPU：',report.data['environment']['gpu'],'；profile：',args.profile,flush=True)
        traces=None
        if args.trace:
            traces=report.directory/'traces'
            traces.mkdir()
        for name in selected:
            report.data['active_case']=name
            report.save()
            # 固定每个实验自己的随机流，单独运行与all模式使用同一输入。
            bench=CudaBench(device,args.profile,args.seed,trace_dir=traces,checkpoint=report.add)
            try:
                CASES[name](bench)
            except Unsupported as exc:
                report.add(skipped(name,'backend',str(exc)))
            except PersistenceError:
                raise
            except Exception as exc:
                report.data['errors'].append(dict(case=name,type=type(exc).__name__,message=str(exc),
                                                  traceback=traceback.format_exc()))
                report.add(dict(case=name,variant='execution',status='failed',reason=str(exc)))
            finally:
                import torch
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        report.finish('interrupted')
        return 130
    except Exception as exc:
        report.data['errors'].append(dict(case=report.data['active_case'],type=type(exc).__name__,
                                          message=str(exc),traceback=traceback.format_exc()))
        report.finish('failed')
        print(f'FAILED: {exc}',file=sys.stderr)
        return 1
    status=run_status(report.data['results'],report.data['errors'])
    report.finish(status)
    print('运行状态：',status,flush=True)
    if report.directory:
        print('查看：',report.directory/'summary.md',flush=True)
    return 1 if status in ('partial','failed') else 0


if __name__=='__main__':
    raise SystemExit(main())

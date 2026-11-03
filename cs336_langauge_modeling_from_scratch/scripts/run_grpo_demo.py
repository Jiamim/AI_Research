"""Compare advantage normalizations on the tiny-LM GRPO loop over several seeds.

Writes book/grpo-results.json. CPU-only, a few minutes with parallel workers.
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.artifact_io import experiment_provenance, write_json
from examples.posttrain import grpo_arithmetic_demo


def job(args):
    method, seed = args
    torch.set_num_threads(2)
    result = grpo_arithmetic_demo(method=method, seed=seed, verbose=False)
    print(method, seed, result["log"][-1]["train_accuracy"], result["log"][-1]["heldout_accuracy"], flush=True)
    return dict(method=method, seed=seed, log=result["log"])


def main():
    jobs = [(m, s) for m in ("grpo", "dr_grpo", "maxrl") for s in range(3)]
    with ProcessPoolExecutor(9, mp_context=mp.get_context('spawn')) as pool:
        runs = list(pool.map(job, jobs))
    summary = {}
    for method in ("grpo", "dr_grpo", "maxrl"):
        finals = [r["log"][-1] for r in runs if r["method"] == method]
        summary[method] = {key: [round(f[key], 4) for f in finals]
                           for key in ("train_accuracy", "heldout_accuracy", "entropy")}
        # Every optimizer update's pre-clip norm, not just the one logged every 10 steps.
        logs = [[entry for entry in r["log"][1:]] for r in runs if r["method"] == method]
        summary[method]["clipped_updates"] = [sum(e["clipped_updates"] for e in log) for log in logs]
        summary[method]["updates"] = [sum(e["updates"] for e in log) for log in logs]
        summary[method]["max_grad_norm"] = [round(max(e["max_grad_norm"] for e in log), 2) for log in logs]
    record = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), torch=str(torch.__version__),
                  setup=dict(steps=200, prompts_per_step=32, group=8, lr=1e-3, epochs_per_rollout=2,
                             pretrain_steps=120, policy="125,248-parameter TinyLM"),
                  initial=runs[0]["log"][0], summary=summary, runs=runs)
    record['provenance'] = experiment_provenance(ROOT)
    write_json(ROOT/'book/grpo-results.json', record)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

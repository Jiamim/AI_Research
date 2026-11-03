"""Recompute the small deterministic CPU experiments quoted in the core chapters.

Writes book/experiment-results.json: the synthetic-arithmetic model at 120 and
1000 steps (curves + per-role NLL + exact-answer accuracy), the scaling design
study, resource-accounting tables, and the inference demos.
"""
import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.artifact_io import experiment_provenance, write_json
from examples.train import run
from examples.evaluation import arithmetic_diagnostics
from examples.scaling import seed_study
from examples.accounting import report
from examples.inference import speculative_distribution_check, paged_cache_demo
from examples.systems import accumulation_demo


def main():
    record = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), torch=str(torch.__version__), device="cpu")
    arithmetic = {}
    for steps in (120, 1000):
        out = ROOT / f"outputs/arithmetic_{steps}"
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = run(steps=steps, out=str(out))
        arithmetic[str(steps)] = dict(history=metrics["history"], parameters=metrics["parameters"],
                                      diagnostics=arithmetic_diagnostics(out / "checkpoint.pt"))
    record["arithmetic"] = arithmetic
    record["scaling"] = seed_study()
    record["accounting"] = report()
    record["inference"] = dict(speculative=speculative_distribution_check(), paged_cache=paged_cache_demo())
    record["accumulation"] = accumulation_demo()
    record['provenance'] = experiment_provenance(ROOT)
    write_json(ROOT/'book/experiment-results.json', record)
    for steps, result in arithmetic.items():
        d = result["diagnostics"]
        print(steps, "steps: accuracy", d["all_100_prompt_accuracy"],
              {k: round(v["nll"], 3) for k, v in d["position_metrics"].items()})


if __name__ == "__main__":
    main()

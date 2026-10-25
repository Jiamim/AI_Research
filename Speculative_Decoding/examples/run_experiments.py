"""Explicitly refresh offline experiment JSON; figure builds only read it.

Run: python -m examples.run_experiments
"""
import importlib.metadata
import json
from pathlib import Path
import platform

import torch

from .speculative_sampling import one_step_experiment
from .tiny_transformer import (
    TinyTransformer, autoregressive_generate, demo, lookup_generate, summarize_trace,
)
from .ngram_speculation import demo as ngram_demo

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = dict(
        environment={"python": platform.python_version(),
                     **{name: importlib.metadata.version(name) for name in ("numpy", "torch", "matplotlib")}},
        one_step=one_step_experiment(), tiny_transformer=demo(),
        speed_model={"kind": "analytical; not GPU measurements", "draft_cost": .08,
                     "nonideal_verify_slope": .035},
    )
    torch.manual_seed(10)
    model = TinyTransformer().double().eval()
    prompt = [1, 2, 3]
    expected = autoregressive_generate(model, prompt, 24)
    output, trace = lookup_generate(model, prompt, 24, gamma=4)
    if output != expected:
        raise AssertionError("Lookup differs from target greedy generation")
    result["lookup_transformer"] = dict(greedy_equal=True, weights="random", device="cpu",
                                      stats=summarize_trace(trace), trace=trace,
                                      note="Correctness only; no wall-clock measurement.")
    (ROOT / "data/experiment_results.json").write_text(json.dumps(result, indent=2) + "\n")
    (ROOT / "data/ngram_results.json").write_text(json.dumps(ngram_demo(), indent=2) + "\n")
    print("Updated probability, neural/lookup Transformer, and n-gram experiment JSON.")


if __name__ == "__main__":
    main()

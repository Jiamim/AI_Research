"""Optional real-model demo. Needs a working Transformers install and checkpoints.

Not GPU-benchmarked in the provided CPU environment. Checkpoint downloads are
triggered only if you run this file yourself. Supports a shared tokenizer only.
"""
import argparse
import time
import statistics
import json
import math
import platform
from pathlib import Path
import torch


def benchmark(run, repeats):
    """Check every warmup/measurement against one target-only reference."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    reference, _ = run(False)
    warm_assisted, _ = run(True)
    if not torch.equal(reference, warm_assisted):
        raise AssertionError("Greedy mismatch during warmup")
    times = {False: [], True: []}
    for repeat in range(repeats):
        for assisted in ([False, True] if repeat % 2 == 0 else [True, False]):
            output, elapsed = run(assisted)
            if not torch.equal(reference, output):
                raise AssertionError(f"Greedy mismatch at repeat {repeat}, assisted={assisted}")
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("Measured durations must be finite and positive")
            times[assisted].append(elapsed)
    return reference, times


def timing_summary(times, generated_tokens):
    """Distinguish typical request latency from pooled sequential token rates.

    Every run has the same verified output length. Pooled token/s is total
    generated tokens / total measured seconds, not the mean of per-run rates.
    It is still a sequential single-request metric, not serving throughput.
    """
    if len(times[False]) != len(times[True]) or not times[False]:
        raise ValueError("Need paired, nonempty timing samples")
    if any(not math.isfinite(t) or t <= 0 for ts in times.values() for t in ts):
        raise ValueError("Measured durations must be finite and positive")
    base, spec = (statistics.median(times[flag]) for flag in [False, True])
    return dict(
        baseline_seconds=base, assisted_seconds=spec, e2e_speedup=base/spec,
        speedup_definition="ratio of median end-to-end request durations",
        aggregate_speedup=sum(times[False])/sum(times[True]),
        pooled_baseline_tokens_per_second=generated_tokens*len(times[False])/sum(times[False]),
        pooled_assisted_tokens_per_second=generated_tokens*len(times[True])/sum(times[True]),
        paired_speedups=[a/b for a,b in zip(times[False],times[True])],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--prompt", default="Explain speculative decoding in three sentences.")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--target-revision", default=None)
    parser.add_argument("--draft-revision", default=None)
    parser.add_argument("--output", type=Path, help="Save raw timings and environment as JSON")
    args = parser.parse_args()
    if args.tokens < 1 or args.gamma < 1 or args.repeats < 1:
        parser.error("tokens, gamma, and repeats must be positive")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.target, revision=args.target_revision)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft, revision=args.draft_revision)
    if (tokenizer.get_vocab() != draft_tokenizer.get_vocab()
            or tokenizer.all_special_ids != draft_tokenizer.all_special_ids
            or tokenizer(args.prompt)["input_ids"] != draft_tokenizer(args.prompt)["input_ids"]):
        raise ValueError("Need matching vocabulary, special IDs and prompt tokenization")
    # Equal vocabulary maps are necessary but not sufficient: use a known
    # compatible pair with the same normalization and special-token behavior.
    target = AutoModelForCausalLM.from_pretrained(args.target, revision=args.target_revision, dtype=dtype).to(device).eval()
    draft = AutoModelForCausalLM.from_pretrained(args.draft, revision=args.draft_revision, dtype=dtype).to(device).eval()
    draft.generation_config.num_assistant_tokens = args.gamma
    draft.generation_config.num_assistant_tokens_schedule = "constant"
    draft.generation_config.assistant_confidence_threshold = 0.0
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    kwargs = dict(max_new_tokens=args.tokens, do_sample=False, num_beams=1,
                  num_return_sequences=1, return_dict_in_generate=False, use_cache=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        kwargs["pad_token_id"] = tokenizer.eos_token_id

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def run(assisted):
        sync(); start = time.perf_counter()
        with torch.inference_mode():
            output = target.generate(**inputs, **kwargs, **({"assistant_model": draft} if assisted else {}))
        sync()
        return output, time.perf_counter() - start

    output, times = benchmark(run, args.repeats)
    generated = output.shape[1] - inputs["input_ids"].shape[1]
    import transformers
    result = dict(device=device, dtype=str(dtype), torch_version=torch.__version__,
                  python_version=platform.python_version(), cuda_version=torch.version.cuda,
                  device_name=torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
                  device_memory_bytes=torch.cuda.get_device_properties(0).total_memory if device == "cuda" else None,
                  transformers_version=transformers.__version__, target=args.target, draft=args.draft,
                  target_revision=args.target_revision, draft_revision=args.draft_revision,
                  target_commit=getattr(target.config, "_commit_hash", None),
                  draft_commit=getattr(draft.config, "_commit_hash", None),
                  gamma=args.gamma, prompt=args.prompt, requested_tokens=args.tokens,
                  input_tokens=int(inputs["input_ids"].shape[1]),
                  generated_tokens=generated, **timing_summary(times, generated), greedy_equal=True,
                  generation_kwargs=kwargs,
                  raw_baseline_seconds=times[False], raw_assisted_seconds=times[True],
                  note="Includes prefill; both models resident; not isolated decode or serving throughput.")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("Timing includes prefill and generation; this is NOT isolated decode TPOT or serving throughput.")
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

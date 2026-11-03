"""Synthetic data and distillation on the arithmetic toy, with a known-correct verifier.

Teacher: the 1000-step model from chapter 20 (100% exact answers when greedy).
Student: a fresh, smaller TinyLM. Four ways to train the student on the same
budget of answer tokens are compared. Every arm uses the same loss structure and source
prompt pool: cross-entropy on the three next-byte targets inside the prompt plus a loss on the answer
byte. Filtering changes prompt frequencies, so its sampled prompt sequence is not identical:

* real: ground-truth "a+b=c" rows (the oracle data source);
* teacher_sampled: rows whose answers the teacher generates at temperature T;
* teacher_filtered: the same samples kept only if a verifier marks them correct
  (rejection sampling, the core of most synthetic-data pipelines);
* soft_kd: token-level KL to the teacher's full answer distribution at the same temperature T,
  so it carries the same information as the samples, without the sampling noise.

The point is measurable: unverified sampling at high temperature injects wrong
answers, and filtering removes them. Every arm keeps the answer-loss coefficient
fixed; otherwise loss weighting would confound the comparison of teacher signals.
"""
import contextlib
import io
import json
import math
import random
import tempfile
import torch
from torch.nn import functional as F
from .model import Config, TinyLM, generate
from .optim import AdamW, cosine_lr
from .train import run

FACTS = [(a, b) for a in range(10) for b in range(10)]
STUDENT = dict(width=48, layers=2, heads=4, kv_heads=2, hidden=128)


def teacher(steps=1000, seed=7):
    with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
        run(steps=steps, out=folder, seed=seed)
        saved = torch.load(f"{folder}/checkpoint.pt", weights_only=True)
    model = TinyLM(Config(**saved["config"]))
    model.load_state_dict(saved["model"])
    return model.eval()


@torch.no_grad()
def exact_accuracy(model, facts=FACTS):
    prompts = torch.tensor([list(f"{a}+{b}=".encode()) for a, b in facts])
    answers = model(prompts)[:, -1].argmax(-1)
    targets = torch.tensor([ord(str((a + b) % 10)) for a, b in facts])
    return (answers == targets).float().mean().item()


@torch.no_grad()
def teacher_rows(model, facts, temperature, generator):
    """Teacher completions as raw 6-byte rows (latin-1 keeps every byte a single character)."""
    prompts = torch.tensor([list(f"{a}+{b}=".encode()) for a, b in facts])
    out = generate(model, prompts, 2, temperature=temperature, generator=generator)
    return [bytes(row.tolist()).decode("latin-1") for row in out]


DIGITS = "0123456789"


def is_correct(row):
    # ASCII digits only: rows are latin-1 decoded, and str.isdigit() also accepts "²", which int() rejects.
    return len(row) == 6 and row[0] in DIGITS and row[1] == "+" and row[2] in DIGITS and row[3] == "=" \
        and row[4] in DIGITS and row[5] == "\n" and int(row[4]) == (int(row[0]) + int(row[2])) % 10


def train_student(rows, teacher_model=None, kd_temperature=1.0, steps=300, seed=0, lr=3e-3, batch=32):
    """Train on 6-byte rows. The loss is CE on '+', the second operand, and '=' plus a loss on the
    answer byte: CE with the row's answer (hard target), or, if teacher_model is given, KL to
    the teacher's answer distribution softened by kd_temperature (the row's answer is unused)."""
    if not rows or any(not isinstance(row,str) or len(row.encode('latin-1')) != 6 for row in rows):
        raise ValueError('nonempty six-byte training rows required')
    if type(steps) is not int or type(batch) is not int or min(steps,batch) < 1 \
            or not math.isfinite(kd_temperature) or kd_temperature <= 0:
        raise ValueError('positive integer steps/batch and finite positive teacher temperature required')
    torch.manual_seed(seed)
    student = TinyLM(Config(**STUDENT))
    opt = AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = cosine_lr(step, lr, lr * 0.1, min(20, steps - 1), steps)
        ids = torch.tensor([list(rng.choice(rows).encode("latin-1")) for _ in range(batch)])
        prompts = ids[:, :4]
        logits = student(prompts)
        prefix_loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), prompts[:, 1:].reshape(-1))
        if teacher_model is None:
            answer_loss = F.cross_entropy(logits[:, -1], ids[:, 4])
        else:
            with torch.no_grad():
                target = (teacher_model(prompts)[:, -1] / kd_temperature).log_softmax(-1)
            answer_loss = F.kl_div(logits[:, -1].log_softmax(-1), target, log_target=True, reduction="batchmean")
        loss = prefix_loss + answer_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
    return exact_accuracy(student.eval())


def experiment(temperature=1.5, samples_per_fact=4, seeds=(0, 1, 2, 3, 4)):
    if not math.isfinite(temperature) or temperature <= 0 or type(samples_per_fact) is not int or samples_per_fact < 1 or not seeds:
        raise ValueError('positive temperature/sample count and at least one seed required')
    model = teacher()
    g = torch.Generator().manual_seed(0)
    facts = [f for f in FACTS for _ in range(samples_per_fact)]
    sampled = teacher_rows(model, facts, temperature, g)
    filtered = [r for r in sampled if is_correct(r)]
    if not filtered:
        raise ValueError('the teacher produced no accepted data; cannot train the filtered-data baseline')
    real = [f"{a}+{b}={(a + b) % 10}\n" for a, b in facts]
    well_formed = sum(len(r) == 6 and r[4] in DIGITS and r[5] == "\n" for r in sampled)
    # Rejection sampling keeps more samples of facts the teacher already finds easy.
    kept = [sum(f == g and is_correct(r) for g, r in zip(facts, sampled)) for f in FACTS]
    results = dict(teacher_accuracy=exact_accuracy(model), temperature=temperature,
                   sampled_rows=len(sampled), sampled_well_formed_fraction=well_formed / len(sampled),
                   sampled_correct_fraction=len(filtered) / len(sampled), filtered_rows=len(filtered),
                   facts_by_kept_samples={k: kept.count(k) for k in range(samples_per_fact + 1)}, students={})
    for name, kwargs in [("real", dict(rows=real)), ("teacher_sampled", dict(rows=sampled)),
                         ("teacher_filtered", dict(rows=filtered)), ("soft_kd", dict(rows=real, teacher_model=model, kd_temperature=temperature))]:
        results["students"][name] = [round(train_student(seed=s, **kwargs), 3) for s in seeds]
    return results


if __name__ == "__main__":
    print(json.dumps(experiment(), indent=2))

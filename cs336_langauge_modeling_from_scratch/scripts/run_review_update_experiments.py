"""Record the controlled review-update experiments without expensive LM training."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.artifact_io import experiment_provenance, write_json
from examples.numerical_tools import demo, corrected_kl, mismatch_stats


def main():
    logits = torch.tensor([.3, -.5, .8], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([.2, .5, .3], dtype=torch.float64)
    old = torch.tensor([.6, .1, .3], dtype=torch.float64)
    corrected = corrected_kl(logits, reference, old)
    p = logits.softmax(-1)
    exact = (p * (p.log() - reference.log())).sum()
    a = torch.autograd.grad(corrected, logits, retain_graph=True)[0]
    b = torch.autograd.grad(exact, logits)[0]
    result = demo()
    result.update(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  scope="Controlled CPU distributions/matrices; not full LM training or GPU performance",
                  corrected_kl=dict(value=corrected.item(), exact=exact.item(),
                                    gradient_max_error=(a-b).abs().max().item()),
                  mismatch_example=mismatch_stats(torch.tensor([0., 0., 2.]), torch.zeros(3), torch.ones(3)))
    result['provenance'] = experiment_provenance(ROOT)
    write_json(ROOT/'book/review-update-results.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

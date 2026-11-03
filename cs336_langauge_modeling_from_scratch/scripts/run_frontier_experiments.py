"""Recompute the experiments quoted in the frontier chapters (29-34).

Writes book/frontier-results.json. The synthetic-data experiment trains small
models and takes about a minute on CPU; everything else is instant.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.artifact_io import experiment_provenance, write_json
from examples.architectures import report as architecture_report
from examples.long_context import demo as long_context_demo
from examples.rl_objectives import demo as rl_demo
from examples.agents import demo as agent_demo
from examples.low_precision import demo as low_precision_demo
from examples.synthetic_data import experiment as synthetic_experiment


def main():
    torch.set_num_threads(4)
    index = json.loads((ROOT / "sources/model-configs/index.json").read_text(encoding="utf-8"))
    record = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), torch=str(torch.__version__), device="cpu",
                  model_configs=dict(retrieved=index["retrieved"],
                                     revisions={k: v["revision"][:12] for k, v in index["models"].items()}),
                  architectures=architecture_report(), long_context=long_context_demo(), rl_objectives=rl_demo(),
                  agents=agent_demo(), low_precision=low_precision_demo(), synthetic_data=synthetic_experiment())
    record['provenance'] = experiment_provenance(ROOT)
    write_json(ROOT/'book/frontier-results.json', record)
    print(json.dumps(record["synthetic_data"], indent=1))


if __name__ == "__main__":
    main()

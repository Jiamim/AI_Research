"""Record executable derivations used throughout the guide."""
from datetime import datetime, timezone
from pathlib import Path
import sys
import json
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.artifact_io import experiment_provenance, write_json
from examples.mechanisms import demo

if __name__=='__main__':
    torch.set_num_threads(2)
    result=demo()
    result.update(timestamp_utc=datetime.now(timezone.utc).isoformat(),torch=str(torch.__version__),device='cpu')
    result['provenance'] = experiment_provenance(ROOT)
    write_json(ROOT/'book/mechanism-results.json', result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

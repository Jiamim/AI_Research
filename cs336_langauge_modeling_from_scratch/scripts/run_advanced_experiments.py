"""Record deterministic small experiments used in the six advanced chapters."""
from pathlib import Path
from datetime import datetime,timezone
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.artifact_io import experiment_provenance, write_json
from examples.research import reweight_domains,pareto_front,gradient_transfer_demo
from examples.retrieval import demo as retrieval_demo
from examples.reasoning import verifier_experiment
from examples.compression import demo as compression_demo
from examples.interpretability import demo as patching_demo


def main():
    record=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                scope='Synthetic or fixed-corpus CPU experiments; no pretrained-LM performance claims',
                research=dict(domain_weights=reweight_domains([.5,.3,.2],[.1,.8,.2]).tolist(),
                              pareto_indices=pareto_front([[80,200],[81,300],[79,250]],[True,False]),
                              gradient_transfer=gradient_transfer_demo()),
                retrieval=retrieval_demo(),verifier=verifier_experiment(),
                compression=compression_demo(),patching=patching_demo())
    record['provenance'] = experiment_provenance(ROOT)
    write_json(ROOT/'book/advanced-experiments.json', record)
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

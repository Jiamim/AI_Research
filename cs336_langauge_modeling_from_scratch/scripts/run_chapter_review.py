"""Record the controlled experiments introduced by chapter-by-chapter review."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.artifact_io import experiment_provenance, write_json
from examples.numerical_tools import chapter_interfaces_demo, chapter_invariants_demo, chapter_review_demo

if __name__ == '__main__':
    result = chapter_review_demo()
    result['chapter_invariants'] = chapter_invariants_demo()
    result['interfaces'] = chapter_interfaces_demo()
    result['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
    result['provenance'] = experiment_provenance(ROOT)
    write_json(ROOT/'book/chapter-review-results.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

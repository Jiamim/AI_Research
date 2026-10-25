"""Reproducible run metadata and atomic text output; no writes on import."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import stat
import tempfile
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_RESULTS = ROOT / "experiments" / "results"


def run_metadata(parameters, extra_sources=()):
    # Hash all experiment modules: a snippet may depend on any shared helper.
    paths = set((ROOT / "experiments").glob("*.py"))
    paths.update((ROOT / "experiments" / "demos").glob("*.py"))
    paths.update(ROOT / name for name in ("requirements.txt", "requirements-demos.txt", "requirements-cuda.txt", "constraints-cpu.txt"))
    paths.update(ROOT / name for name in extra_sources)
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(paths) if path.is_file()}
    versions = {}
    for name in ("numpy", "matplotlib", "nbformat"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "dependencies": versions,
        "parameters": parameters,
        "source_files_sha256": hashes,
        "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
    }


def write_text(path, text):
    """Replace one complete file, leaving the previous version on write failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        temporary.chmod(stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, payload):
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def file_link(label, target, output_dir):
    relative = os.path.relpath(target, Path(output_dir).resolve())
    return f"[{label}]({quote(relative, safe='/._-')})"

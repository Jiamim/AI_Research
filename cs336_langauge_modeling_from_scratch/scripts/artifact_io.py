"""Bounded file hashing and publication that preserves old outputs on ordinary I/O failures."""
from contextlib import contextmanager
from hashlib import sha256
import json
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import stat
import sys
import tempfile


def file_sha256(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def source_snapshot(root, folders=('examples', 'scripts', 'tests')):
    root = Path(root)
    return {str(path.relative_to(root)):file_sha256(path)
            for folder in folders for path in sorted((root/folder).glob('*.py'))}


def validation_snapshot(root):
    root = Path(root)
    snapshot = source_snapshot(root)
    inputs = [path for path in (root/'book/chapters').glob('*.md') if not path.name.startswith(('35-','36-'))]
    inputs.extend((root/'sources/model-configs').glob('*.json'))
    snapshot.update({str(path.relative_to(root)):file_sha256(path) for path in sorted(inputs)})
    return snapshot


def runtime_snapshot():
    """Runtime identity for reusing validation; absent optional packages are significant."""
    packages = {}
    for name in ('torch', 'numpy', 'regex', 'triton', 'transformers', 'markdown-it-py',
                 'beautifulsoup4', 'PyMuPDF', 'fonttools', 'Pillow', 'playwright'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=platform.python_version(), implementation=platform.python_implementation(),
                system=platform.system(), machine=platform.machine(), byteorder=sys.byteorder,
                packages=packages, environment={name:os.environ.get(name) for name in
                    ('PYTHONHASHSEED', 'TRITON_INTERPRET', 'GLOO_SOCKET_IFNAME',
                     'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})


def experiment_provenance(root):
    root = Path(root)
    sources = source_snapshot(root, ('examples','scripts'))
    sources.update({str(path.relative_to(root)):file_sha256(path) for path in sorted((root/'sources/model-configs').glob('*.json'))})
    return dict(python=platform.python_version(),
                packages={name:importlib.metadata.version(name) for name in ('torch','numpy')},
                source_sha256=sources)


@contextmanager
def atomic_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.write-', delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    with atomic_output(path) as stream:
        stream.write(encoded)


def publish_files(files):
    """Replace a group after generation; roll back earlier replacements if a later one fails.

    Not a crash-atomic filesystem transaction: readers must still verify the manifest/hash.
    Backup files are retained if a rollback itself fails.
    """
    files = [(Path(source), Path(target)) for source, target in files]
    targets = {target.resolve() for _, target in files}
    if len(targets) != len(files):
        raise ValueError('publication destinations must be unique')
    if any(source.resolve() in targets for source, _ in files):
        raise ValueError('publication sources must be staged separately from destinations')
    backups, published = {}, []
    rollback_failed = False
    try:
        for source, target in files:
            if not source.is_file():
                raise FileNotFoundError(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            backups[target] = None
            if target.exists():
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.previous-', delete=False) as stream:
                    backups[target] = Path(stream.name)
                shutil.copy2(target, backups[target])
        for source, target in files:
            with atomic_output(target) as stream, source.open('rb') as incoming:
                shutil.copyfileobj(incoming, stream, length=1 << 20)
            published.append(target)
    except BaseException:
        try:
            for target in reversed(published):
                if backups[target] is None:
                    target.unlink(missing_ok=True)
                else:
                    backups[target].replace(target)
        except OSError:
            rollback_failed = True
            raise
        raise
    finally:
        if not rollback_failed:
            for backup in backups.values():
                if backup is not None:
                    backup.unlink(missing_ok=True)

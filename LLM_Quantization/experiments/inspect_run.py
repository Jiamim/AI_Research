#!/usr/bin/env python3
"""Validate a protocol-v4 Colab report ZIP or directory without extracting it.

Usage: python experiments/inspect_run.py /path/to/report.zip
Hashes establish internal consistency, not authenticity or benchmark quality.
"""

import argparse
import hashlib
import json
import math
from numbers import Real
from pathlib import Path, PurePosixPath
import statistics
import zipfile


def validate_files(files):
    """Inspect a mapping of relative file names to bytes; raise on inconsistency."""
    def read(name):
        def reject_constant(value):
            raise ValueError(f'Non-finite JSON number in {name}: {value}')

        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f'Duplicate JSON key in {name}: {key}')
                result[key] = value
            return result

        value = json.loads(files[name], parse_constant=reject_constant, object_pairs_hook=unique_pairs)
        # JSON exponent overflow (for example 1e999) bypasses parse_constant.
        def check_numbers(item):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f'Non-finite JSON number in {name}')
            if isinstance(item, dict):
                for child in item.values():
                    check_numbers(child)
            elif isinstance(item, list):
                for child in item:
                    check_numbers(child)
        check_numbers(value)
        return value

    def positive_integer(value, label):
        if type(value) is not int or value <= 0:
            raise ValueError(f'{label} must be a positive integer')
        return value

    def finite_number(value, label, minimum=0, strict=False):
        if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                or value < minimum or (strict and value == minimum)):
            raise ValueError(f'{label} must be finite and {">" if strict else ">="} {minimum}')
        return value

    def matches(value, expected, label, rel_tol=1e-6):
        finite_number(value, label)
        if not math.isclose(value, expected, rel_tol=rel_tol, abs_tol=1e-8):
            raise ValueError(f'{label} is inconsistent with its source measurements')

    manifest = read('artifact_manifest.json')
    expected = manifest['files_sha256']
    if set(files) - {'artifact_manifest.json'} != set(expected):
        raise ValueError('Archive inventory differs from artifact_manifest.json')
    for name, checksum in expected.items():
        if hashlib.sha256(files[name]).hexdigest() != checksum:
            raise ValueError(f'Artifact hash mismatch: {name}')
    metadata = read('metadata.json')
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported artifact manifest schema')
    if metadata['protocol_version'] != manifest['protocol_version']:
        raise ValueError('Manifest and metadata protocol versions differ')
    if metadata['protocol_version'] != 4:
        raise ValueError('This inspector supports protocol v4 only')
    if hashlib.sha256(files['executed_cells.txt']).hexdigest() != metadata['executed_cells_sha256']:
        raise ValueError('Executed source hash mismatch')
    plan = [row['mode'] for row in metadata['plan']]
    if not plan or any(not isinstance(mode, str) or not mode for mode in plan) or plan[0] != 'fp16':
        raise ValueError('The nonempty experiment plan must start with FP16')
    new_tokens = positive_integer(metadata['new_tokens'], 'new_tokens')
    repeats = positive_integer(metadata['timing_repeats'], 'timing_repeats')
    results = read('results.json') if 'results.json' in files else []
    modes = [row['mode'] for row in results]
    failures = read('failures.json') if 'failures.json' in files else []
    failed = [row['mode'] for row in failures]
    if len(plan) != len(set(plan)) or len(modes) != len(set(modes)) or len(failed) != len(set(failed)):
        raise ValueError('Duplicate experiment mode')
    if set(modes) & set(failed) or not set(modes + failed) <= set(plan):
        raise ValueError('Completed and failed modes must be disjoint subsets of the plan')
    if metadata['completed_modes'] != modes or metadata['failures'] != failures:
        raise ValueError('Metadata status differs from result/failure records')
    complete = set(modes) == set(plan) and not failures
    if metadata['status'] not in ('completed', 'partial', 'interrupted'):
        raise ValueError('Report has not reached a terminal state')
    if (metadata['status'] == 'completed') != complete:
        raise ValueError('Completion status does not match the result inventory')
    if results:
        baseline = next((row for row in results if row['mode'] == 'fp16'), None)
        if baseline is None:
            raise ValueError('Missing FP16 baseline')
        for row in results:
            positive_integer(row['scored_tokens'], 'scored_tokens')
            positive_integer(row['generated_tokens'], 'generated_tokens')
            if row['scored_tokens'] != baseline['scored_tokens']:
                raise ValueError('Methods scored different numbers of tokens')
            if row['generated_tokens'] != new_tokens:
                raise ValueError('Generation length differs from the timing protocol')
            if not isinstance(row['generation_seconds_all'], list) or len(row['generation_seconds_all']) != repeats:
                raise ValueError('Missing timing repetitions')
            if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0
                   for t in row['generation_seconds_all']):
                raise ValueError('Timing repetitions must be finite positive seconds')
            median = statistics.median(row['generation_seconds_all'])
            if 'generation_seconds_median' in row:
                matches(row['generation_seconds_median'], median, 'generation_seconds_median')
            if 'generation_tokens_per_s' in row:
                matches(row['generation_tokens_per_s'], new_tokens / median, 'generation_tokens_per_s')
            for field in ('ppl', 'nll_sum', 'logits_kl_vs_fp16', 'model_footprint_mib',
                          'peak_generation_allocated_mib', 'preparation_seconds', 'weight_nmse'):
                if field in row and not (field == 'weight_nmse' and row[field] is None):
                    finite_number(row[field], field, minimum=1 if field == 'ppl' else 0)
            if 'ppl' in row and 'nll_sum' in row:
                matches(math.log(row['ppl']), row['nll_sum'] / row['scored_tokens'], 'log(ppl)', rel_tol=1e-5)
            if metadata.get('measure_phases', False) or 'phase_seconds_all' in row:
                times = row.get('phase_seconds_all')
                if (new_tokens < 2 or not isinstance(times, list) or len(times) != repeats
                        or any(not isinstance(pair, list) or len(pair) != 2 for pair in times)):
                    raise ValueError('Missing or malformed prefill/decode timing repetitions')
                for pair in times:
                    for value in pair:
                        finite_number(value, 'phase timing', strict=True)
                if positive_integer(row['decode_tokens'], 'decode_tokens') != new_tokens - 1:
                    raise ValueError('Decode token count differs from the phase timing protocol')
                prefill = statistics.median(pair[0] for pair in times)
                decode = statistics.median(pair[1] for pair in times)
                matches(row['prefill_forward_seconds_median'], prefill, 'prefill_forward_seconds_median')
                matches(row['decode_seconds_median'], decode, 'decode_seconds_median')
                matches(row['decode_tokens_per_s'], (new_tokens - 1) / decode, 'decode_tokens_per_s')
    return {'status': metadata['status'], 'completed': modes, 'failed': failed,
            'profile': metadata['profile'], 'protocol_version': metadata['protocol_version'],
            'verified_files': len(expected)}


def inspect(path):
    path = Path(path)
    if path.is_dir():
        files = {p.relative_to(path).as_posix(): p.read_bytes()
                 for p in path.rglob('*') if p.is_file()}
    else:
        with zipfile.ZipFile(path) as archive:
            names = [info.filename for info in archive.infolist() if not info.is_dir()]
            if len(names) != len(set(names)):
                raise ValueError('ZIP contains duplicate file names')
            if any(PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts for name in names):
                raise ValueError('ZIP must contain relative report paths')
            files = {name: archive.read(name) for name in names}
    return validate_files(files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(args.report), ensure_ascii=False, indent=2))
    except (KeyError, TypeError, ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'Invalid report: {exc}\n')

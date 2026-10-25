"""Validate archive integrity and partial-run semantics without a model."""

import hashlib
import json
import math
import unittest

from experiments.inspect_run import validate_files


def report(status='partial'):
    results = [dict(mode='fp16', scored_tokens=2040, generated_tokens=64,
                    generation_seconds_all=[1., 1.1, .9])]
    failures = [dict(mode='awq4_g128', message='backend unavailable')]
    metadata = dict(protocol_version=4, profile='smoke', status=status,
                    plan=[dict(mode='fp16'), dict(mode='awq4_g128')],
                    completed_modes=['fp16'], failures=failures,
                    new_tokens=64, timing_repeats=3,
                    executed_cells_sha256=hashlib.sha256(b'example source').hexdigest())
    files = {name: json.dumps(data).encode() for name, data in (
        ('metadata.json', metadata), ('results.json', results), ('failures.json', failures))}
    files['executed_cells.txt'] = b'example source'
    return seal(files)


def seal(files):
    files['artifact_manifest.json'] = json.dumps(dict(schema_version=1, protocol_version=4, files_sha256={
        name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        if name != 'artifact_manifest.json'})).encode()
    return files


class RunArtifactTests(unittest.TestCase):
    def test_partial_run_is_valid_and_reports_failure(self):
        result = validate_files(report())
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['failed'], ['awq4_g128'])

    def test_corruption_missing_file_and_false_completion_rejected(self):
        for action in ('corrupt', 'missing', 'complete'):
            files = report('completed' if action == 'complete' else 'partial')
            if action == 'corrupt':
                files['results.json'] += b' '
            if action == 'missing':
                del files['executed_cells.txt']
            with self.subTest(action=action), self.assertRaises(ValueError):
                validate_files(files)

    def test_matching_hashes_do_not_hide_wrong_measurement_protocol(self):
        files = report()
        results = json.loads(files['results.json'])
        results[0]['generated_tokens'] = 63
        files['results.json'] = json.dumps(results).encode()
        with self.assertRaisesRegex(ValueError, 'Generation length'):
            validate_files(seal(files))

    def test_invalid_timings_are_not_valid_benchmarks(self):
        for value in (0, -1, float('inf'), float('nan'), True):
            files = report()
            results = json.loads(files['results.json'])
            results[0]['generation_seconds_all'][0] = value
            files['results.json'] = json.dumps(results).encode()
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, '[Ff]inite'):
                validate_files(seal(files))

    def test_invalid_counts_and_nonfinite_quality_are_rejected(self):
        for field, value in (('scored_tokens', -1), ('scored_tokens', True),
                             ('generated_tokens', 64.0), ('ppl', -2),
                             ('ppl', float('nan')), ('ppl', None), ('nll_sum', float('inf'))):
            files = report()
            rows = json.loads(files['results.json'])
            rows[0][field] = value
            files['results.json'] = json.dumps(rows).encode()
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_files(seal(files))

    def test_resealed_derived_metrics_must_match_raw_measurements(self):
        fields = dict(ppl=2., nll_sum=2040 * math.log(2.),
                      generation_seconds_median=1., generation_tokens_per_s=64.,
                      phase_seconds_all=[[.1, .7], [.2, .8], [.3, .9]],
                      prefill_forward_seconds_median=.2, decode_seconds_median=.8,
                      decode_tokens=63, decode_tokens_per_s=63/.8)
        files = report()
        row = json.loads(files['results.json'])[0]
        row.update(fields)
        files['results.json'] = json.dumps([row]).encode()
        validate_files(seal(files))
        for field in fields:
            changed = dict(row)
            changed[field] = [[.1, 0.]] * 3 if field == 'phase_seconds_all' else 999
            files['results.json'] = json.dumps([changed]).encode()
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_files(seal(files))

    def test_duplicate_json_keys_are_rejected_even_when_resealed(self):
        files = report()
        files['metadata.json'] = files['metadata.json'].replace(b'"new_tokens": 64',
                                                              b'"new_tokens": 1, "new_tokens": 64')
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
            validate_files(seal(files))

    def test_exponent_overflow_is_not_accepted_as_valid_json_measurement(self):
        files = report()
        files['results.json'] = files['results.json'].replace(b'"mode": "fp16"',
                                                            b'"mode": "fp16", "aux_metric": 1e999')
        with self.assertRaisesRegex(ValueError, 'Non-finite JSON'):
            validate_files(seal(files))

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiment_results import (
    aggregate_standard_runs,
    build_identity,
    make_envelope,
    result_directory,
    scan_results,
    write_experiment,
    write_results_summary,
)


class IdentityTests(unittest.TestCase):
    def test_identity_is_stable_and_changes_with_data_or_parameters(self):
        first = build_identity({"advbench": "data-a"}, {"batch_size": 8})
        same = build_identity({"advbench": "data-a"}, {"batch_size": 8})
        changed_parameter = build_identity({"advbench": "data-a"}, {"batch_size": 16})
        changed_data = build_identity({"advbench": "data-b"}, {"batch_size": 8})
        self.assertEqual(first["run_id"], same["run_id"])
        self.assertNotEqual(first["run_id"], changed_parameter["run_id"])
        self.assertNotEqual(first["run_id"], changed_data["run_id"])

    def test_same_identity_updates_only_its_own_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identities = [
                build_identity({"advbench": "data"}, {"batch_size": size})
                for size in (8, 16)
            ]
            for identity in identities:
                envelope = make_envelope(
                    experiment_kind="text", script="test.py", identity=identity,
                    config={}, datasets={}, runs=[],
                )
                target = result_directory(root, identity)
                write_experiment(envelope, target)
                write_experiment(envelope, target)
            self.assertEqual(len(list(root.glob("advbench/*/result.json"))), 2)


class AggregationTests(unittest.TestCase):
    def test_aggregates_totals_peak_micro_and_weighted_macro(self):
        runs = [
            {"dataset": "a", "backend": "vllm", "cache_policy": "native", "order": "default", "status": "ok",
             "metrics": {"total_tokens": 100, "hit_tokens": 20, "peak_cache_tokens": 30, "peak_cache_bytes": 1024,
                         "micro_hit_rate": .2, "macro_hit_rate": .1, "num_requests": 1}},
            {"dataset": "a", "backend": "vllm", "cache_policy": "native", "order": "default", "status": "ok",
             "metrics": {"total_tokens": 300, "hit_tokens": 180, "peak_cache_tokens": 40, "peak_cache_bytes": 2048,
                         "micro_hit_rate": .6, "macro_hit_rate": .5, "num_requests": 3}},
        ]
        row = aggregate_standard_runs(runs)[0]
        self.assertEqual(row["total_tokens"], 400)
        self.assertEqual(row["hit_tokens"], 200)
        self.assertEqual(row["peak_cache_tokens"], 40)
        self.assertEqual(row["peak_cache_bytes"], 2048)
        self.assertEqual(row["micro_hit_rate"], .5)
        self.assertEqual(row["macro_hit_rate"], .4)


class SummaryTests(unittest.TestCase):
    def test_scans_new_result_and_skips_unrelated_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = build_identity({"chartqa": "data"}, {"batch_size": 8})
            run = {"dataset": "chartqa", "backend": "vllm", "cache_policy": "native", "order": "grouped",
                   "status": "ok", "metrics": {"total_tokens": 10, "hit_tokens": 4, "peak_cache_tokens": 8,
                   "peak_cache_bytes": None, "micro_hit_rate": .4, "macro_hit_rate": .3, "num_requests": 2}}
            envelope = make_envelope(
                experiment_kind="multimodal", script="test.py", identity=identity,
                config={}, datasets={}, runs=[run],
            )
            write_experiment(envelope, result_directory(root, identity))
            (root / "summary.json").write_text(json.dumps({"not": "a result"}), encoding="utf-8")
            rows, warnings = scan_results(root)
            self.assertEqual(len(rows), 1)
            self.assertFalse(warnings)
            output = root / "index.md"
            write_results_summary(root, output)
            report = output.read_text(encoding="utf-8")
            self.assertIn("Total Tokens", report)
            self.assertIn("40.00%", report)
            self.assertIn("N/A", report)


if __name__ == "__main__":
    unittest.main()

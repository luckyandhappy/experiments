import asyncio
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import run_experiment


class VLLMVersionTests(unittest.TestCase):
    def test_accepts_only_026(self):
        self.assertEqual(run_experiment.validate_vllm_version("0.26.0"), "0.26.0")
        self.assertEqual(
            run_experiment.validate_vllm_version("0.26.3+cu130"),
            "0.26.3+cu130",
        )
        for version in ("0.25.9", "0.27.0", "main"):
            with self.subTest(version=version), self.assertRaises(RuntimeError):
                run_experiment.validate_vllm_version(version)

    def test_import_is_backend_lazy(self):
        command = (
            "import sys, run_experiment; "
            "assert 'sglang' not in sys.modules; "
            "assert 'scheduler' not in sys.modules; "
            "assert 'xxxtrie' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", command],
            cwd=".",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_vllm_cannot_skip_its_only_experiment(self):
        completed = subprocess.run(
            [
                sys.executable,
                "run_experiment.py",
                "--backend",
                "vllm",
                "--skip-baseline",
            ],
            cwd=".",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("不能同时指定 --skip-baseline", completed.stderr)


class VLLMBaselineTests(unittest.IsolatedAsyncioTestCase):
    async def test_vllm_requests_use_strict_batches(self):
        events = []

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeEngine:
            async def generate(self, prompt, sampling_params, request_id):
                events.append(("start", request_id))
                await asyncio.sleep(0.01)
                events.append(("finish", request_id))
                yield types.SimpleNamespace(
                    finished=True,
                    num_cached_tokens=0,
                    num_cache_creation_tokens=len(prompt["prompt_token_ids"]),
                )

        requests = [(('data', i), [1, 2, i]) for i in range(5)]
        await run_experiment._send_requests_vllm(
            FakeEngine(), FakeSamplingParams, requests, tokenizer_eos_id=2,
            batch_size=2,
        )

        third_start = events.index(("start", "data:0002"))
        self.assertLess(events.index(("finish", "data:0000")), third_start)
        self.assertLess(events.index(("finish", "data:0001")), third_start)

    async def test_vllm_apc_self_test_requires_full_native_block(self):
        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeEngine:
            def __init__(self):
                self.reset_called = False

            async def generate(self, prompt, sampling_params, request_id):
                hit = 16 if request_id.endswith("0001") else 0
                yield types.SimpleNamespace(
                    finished=True,
                    num_cached_tokens=hit,
                    num_cache_creation_tokens=16,
                )

            async def reset_prefix_cache(self):
                self.reset_called = True
                return True

        engine = FakeEngine()
        result = await run_experiment._run_vllm_apc_self_test(
            engine, FakeSamplingParams, tokenizer_eos_id=2,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["observed_hit_tokens"], 16)
        self.assertTrue(engine.reset_called)

    async def test_vllm_main_writes_baseline_only_result(self):
        class FakeTokenizer:
            eos_token_id = 2

            def encode(self, prompt, add_special_tokens=False):
                return [1, 2]

        class FakeTokenizerClass:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return FakeTokenizer()

        baseline_result = {
            "backend": "vllm",
            "backend_version": "0.26.1",
            "metrics": {
                "aggregate_hit_rate_micro_percent": 0.0,
                "aggregate_hit_rate_macro_percent": 0.0,
                "cache_creation_tokens": 0,
                "cache_capacity_tokens": None,
                "peak_cache_tokens": None,
                "cache_capacity_bytes": None,
                "peak_cache_mib": None,
                "cache_bytes_available": False,
                "backend_metrics": {
                    "prefix_cache_hit_tokens": 0,
                    "prefix_cache_query_tokens": 2,
                    "native_micro_hit_rate_percent": 0.0,
                    "peak_kv_cache_usage_percent": 0.0,
                },
            },
        }
        run_vllm = mock.AsyncMock(return_value=baseline_result)
        run_trie = mock.AsyncMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            output = f"{tmpdir}/results"
            argv = [
                "run_experiment.py",
                "--backend",
                "vllm",
                "--output-dir",
                output,
                "--datasets",
                "advbench",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    run_experiment,
                    "load_datasets",
                    return_value={"advbench": ["prompt"]},
                ),
                mock.patch.object(
                    run_experiment,
                    "_load_tokenizer_cls",
                    return_value=FakeTokenizerClass,
                ),
                mock.patch.object(
                    run_experiment,
                    "run_vllm_baseline_experiment",
                    run_vllm,
                ),
                mock.patch.object(
                    run_experiment,
                    "aggregate_vllm_baselines",
                    return_value=baseline_result,
                ),
                mock.patch.object(
                    run_experiment,
                    "run_trie_experiment",
                    run_trie,
                ),
            ):
                await run_experiment.main()

            result_paths = list(Path(output).glob("advbench/*/result.json"))
            self.assertEqual(len(result_paths), 1)
            with result_paths[0].open("r", encoding="utf-8") as result_file:
                result = json.load(result_file)
            self.assertTrue((Path(output) / "results_report.md").is_file())

        run_vllm.assert_awaited_once()
        run_trie.assert_not_awaited()
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["config"]["backend"], "vllm")
        self.assertEqual(result["runs"][0]["backend"], "vllm")
        self.assertEqual(result["runs"][0]["details"], baseline_result)
        self.assertEqual(result["summary"]["rows"][0]["total_tokens"], 2)

    def test_old_output_file_option_is_removed(self):
        completed = subprocess.run(
            [sys.executable, "run_experiment.py", "--output", "result.json"],
            cwd=".", capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("unrecognized arguments: --output", completed.stderr)

    async def test_vllm_uses_native_apc_without_cstrie_arguments(self):
        state = {"complete": False}

        class FakeTokenizer:
            eos_token_id = 2

        class FakeTokenizerClass:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return FakeTokenizer()

        class FakeAsyncEngineArgs:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeEngine:
            instance = None

            def __init__(self, engine_args):
                self.engine_args = engine_args
                self.sampler_env = run_experiment.os.environ.get(
                    "VLLM_USE_FLASHINFER_SAMPLER"
                )
                self.model_runner_env = run_experiment.os.environ.get(
                    "VLLM_USE_V2_MODEL_RUNNER"
                )
                self.tokenizer_parallelism_env = run_experiment.os.environ.get(
                    "TOKENIZERS_PARALLELISM"
                )
                self.requests = []
                self.did_shutdown = False
                type(self).instance = self

            @classmethod
            def from_engine_args(cls, engine_args):
                return cls(engine_args)

            async def generate(self, prompt, sampling_params, request_id):
                self.requests.append((prompt, sampling_params.kwargs, request_id))
                await asyncio.sleep(0)
                yield types.SimpleNamespace(
                    finished=False,
                    num_cached_tokens=999,
                    num_cache_creation_tokens=999,
                )
                state["complete"] = True
                cached_tokens, creation_tokens = {
                    "data:0000": (1, 2),
                    "data:0001": (3, 1),
                }[request_id]
                yield types.SimpleNamespace(
                    finished=True,
                    num_cached_tokens=cached_tokens,
                    num_cache_creation_tokens=creation_tokens,
                )

            def shutdown(self):
                self.did_shutdown = True

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.AsyncEngineArgs = FakeAsyncEngineArgs
        fake_vllm.AsyncLLMEngine = FakeEngine
        fake_vllm.SamplingParams = FakeSamplingParams

        def read_metrics():
            cache_configs = (
                {
                    "kv_cache_size_tokens": "1000",
                    "kv_cache_memory_bytes": "2000000",
                },
            )
            if state["complete"]:
                return run_experiment._VLLMMetricSnapshot(
                    values={
                        "vllm:prompt_tokens_total": 7.0,
                        "vllm:prefix_cache_queries": 7.0,
                        "vllm:prefix_cache_hits": 3.0,
                        "vllm:kv_cache_usage_perc": 0.5,
                    },
                    cache_configs=cache_configs,
                )
            return run_experiment._VLLMMetricSnapshot(
                values={
                    "vllm:prompt_tokens_total": 0.0,
                    "vllm:prefix_cache_queries": 0.0,
                    "vllm:prefix_cache_hits": 0.0,
                    "vllm:kv_cache_usage_perc": 0.0,
                },
                cache_configs=cache_configs,
            )

        config = run_experiment.ExperimentConfig(
            model_path="fake-model",
            context_length=4096,
            batch_size=2,
            max_input_tokens=1024,
            metrics_log_path="unused.jsonl",
            scheduler="heuristic",
            gpu_memory_utilization=0.75,
        )
        requests = [(('data', 0), [1, 2, 3]), (('data', 1), [1, 2, 4, 5])]

        with (
            mock.patch.dict(
                run_experiment.os.environ,
                {"VLLM_USE_FLASHINFER_SAMPLER": "1"},
            ),
            mock.patch.dict(sys.modules, {"vllm": fake_vllm}),
            mock.patch.object(
                run_experiment.importlib.metadata,
                "version",
                return_value="0.26.1",
            ),
            mock.patch.object(
                run_experiment,
                "_load_tokenizer_cls",
                return_value=FakeTokenizerClass,
            ),
            mock.patch.object(
                run_experiment,
                "_read_vllm_metric_snapshot",
                side_effect=read_metrics,
            ),
            mock.patch.object(
                run_experiment,
                "_run_vllm_apc_self_test",
                new=mock.AsyncMock(
                    return_value={
                        "passed": True,
                        "expected_min_hit_tokens": 16,
                        "observed_hit_tokens": 16,
                    }
                ),
            ),
        ):
            result = await run_experiment.run_vllm_baseline_experiment(
                config,
                requests,
            )

        engine = FakeEngine.instance
        self.assertTrue(engine.did_shutdown)
        self.assertEqual(engine.engine_args.kwargs["max_num_seqs"], 2)
        self.assertEqual(engine.engine_args.kwargs["max_model_len"], 4096)
        self.assertEqual(engine.engine_args.kwargs["block_size"], 16)
        self.assertNotIn("prefix_match_unit", engine.engine_args.kwargs)
        self.assertEqual(engine.engine_args.kwargs["gpu_memory_utilization"], 0.75)
        self.assertTrue(engine.engine_args.kwargs["enable_prefix_caching"])
        self.assertFalse(engine.engine_args.kwargs["async_scheduling"])
        self.assertEqual(
            [request[0] for request in engine.requests],
            [
                {"prompt_token_ids": [1, 2, 3]},
                {"prompt_token_ids": [1, 2, 4, 5]},
            ],
        )
        for _, sampling_params, _ in engine.requests:
            self.assertEqual(sampling_params["max_tokens"], 1)
            self.assertEqual(sampling_params["temperature"], 0.0)
            self.assertNotIn("custom_cache_prefix_len", sampling_params)
            self.assertNotIn("bootstrap_host", sampling_params)

        self.assertEqual(result["backend"], "vllm")
        self.assertEqual(result["backend_version"], "0.26.1")
        self.assertEqual(result["cache_policy"], "native")
        self.assertEqual(result["cache_match_mode"], "block")
        self.assertEqual(result["block_size"], 16)
        self.assertEqual(result["cache_granularity_tokens"], 16)
        self.assertEqual(result["sampler_backend"], "native")
        self.assertFalse(result["flashinfer_sampler_enabled"])
        self.assertEqual(engine.sampler_env, "0")
        self.assertEqual(engine.model_runner_env, "0")
        self.assertEqual(engine.tokenizer_parallelism_env, "false")
        self.assertEqual(result["model_runner"], "v1")
        self.assertFalse(result["async_scheduling"])
        self.assertFalse(result["tokenizers_parallelism"])
        self.assertEqual(result["metrics"]["total_input_tokens_measured"], 7)
        self.assertEqual(result["metrics"]["total_hit_tokens_measured"], 4)
        self.assertEqual(result["metrics"]["cache_hit_tokens"], 4)
        self.assertEqual(result["metrics"]["cache_creation_tokens"], 3)
        self.assertAlmostEqual(
            result["metrics"]["aggregate_hit_rate_micro"],
            4 / 7,
        )
        self.assertAlmostEqual(
            result["metrics"]["aggregate_hit_rate_macro"],
            ((1 / 3) + (3 / 4)) / 2,
        )
        self.assertEqual(result["metrics"]["cache_capacity_tokens"], 1000)
        self.assertEqual(result["metrics"]["peak_cache_tokens"], 500)
        self.assertEqual(result["metrics"]["peak_full_tokens"], 500)
        self.assertEqual(result["metrics"]["cache_capacity_bytes"], 2000000)
        self.assertEqual(result["metrics"]["peak_cache_bytes"], 1000000)
        self.assertTrue(result["metrics"]["cache_bytes_available"])
        self.assertAlmostEqual(
            result["metrics"]["backend_metrics"]["native_micro_hit_rate"],
            3 / 7,
        )
        self.assertIsNone(result["metrics"]["peak_radix_bytes"])

    async def test_request_metrics_use_only_finished_output_and_track_missing(self):
        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeEngine:
            async def generate(self, prompt, sampling_params, request_id):
                yield types.SimpleNamespace(
                    finished=False,
                    num_cached_tokens=10,
                    num_cache_creation_tokens=10,
                )
                yield types.SimpleNamespace(
                    finished=True,
                    num_cached_tokens=None,
                    num_cache_creation_tokens=None,
                )

        metrics = await run_experiment._send_requests_vllm(
            llm=FakeEngine(),
            sampling_params_cls=FakeSamplingParams,
            flat_seqs=[(("data", 0), [1, 2, 3])],
            tokenizer_eos_id=2,
            batch_size=1,
        )

        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].input_tokens, 3)
        self.assertEqual(metrics[0].cached_tokens, 0)
        self.assertEqual(metrics[0].cache_creation_tokens, 0)
        self.assertTrue(metrics[0].missing_cached_tokens)
        self.assertTrue(metrics[0].missing_cache_creation_tokens)

    def test_metric_reader_preserves_cache_config_labels(self):
        fake_metrics = [
            types.SimpleNamespace(
                name="vllm:cache_config_info",
                labels={
                    "kv_cache_size_tokens": "2048",
                    "kv_cache_memory_bytes": "None",
                },
                value=1.0,
            ),
            types.SimpleNamespace(
                name="vllm:kv_cache_usage_perc",
                labels={"engine": "0"},
                value=0.25,
            ),
            types.SimpleNamespace(
                name="vllm:kv_cache_usage_perc",
                labels={"engine": "1"},
                value=0.5,
            ),
        ]
        fake_reader = types.SimpleNamespace(
            get_metrics_snapshot=lambda: fake_metrics,
        )

        with mock.patch.object(
            run_experiment.importlib,
            "import_module",
            return_value=fake_reader,
        ):
            snapshot = run_experiment._read_vllm_metric_snapshot()

        self.assertEqual(
            run_experiment._metric_value(snapshot, "kv_cache_usage_perc"),
            0.5,
        )
        self.assertEqual(
            run_experiment._vllm_cache_config_int(
                (snapshot,),
                "kv_cache_size_tokens",
            ),
            2048,
        )
        self.assertIsNone(
            run_experiment._vllm_cache_config_int(
                (snapshot,),
                "kv_cache_memory_bytes",
            )
        )


class PrefixOpportunityTests(unittest.TestCase):
    def test_analysis_respects_batch_visibility_and_match_unit(self):
        seqs = [[1, 2, 3], [1, 2, 4], [1, 2, 3, 5]]
        token_result = run_experiment.analyze_prefix_opportunities(
            seqs, match_unit=1, batch_size=2
        )
        block_result = run_experiment.analyze_prefix_opportunities(
            seqs, match_unit=2, batch_size=2
        )
        self.assertEqual(token_result["potential_hit_tokens"], 3)
        self.assertEqual(block_result["potential_hit_tokens"], 2)


if __name__ == "__main__":
    unittest.main()

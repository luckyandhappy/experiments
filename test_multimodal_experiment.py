from __future__ import annotations

from contextlib import chdir
import json
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import multimodal_experiment as multimodal_module
import run_experiment
from multimodal_experiment import (
    MAX_MEDIA,
    MME_CATEGORIES,
    ManifestSample,
    MultimodalSample,
    adapter_names,
    build_manifest,
    encoder_reuse_opportunity,
    get_adapter,
    manifest_samples,
    prepare_requests,
    sample_by_media,
    validate_samples,
)
from run_multimodal_experiment import (
    _run_sglang_requests,
    _sglang_image_uri,
    parse_sglang_encoder_metrics,
    run_vllm,
)
from scheduler import ScheduledRequest
from xxxtrie import XXXTrieNode


class FakeTokenizer:
    eos_token_id = 2
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {"<|image_pad|>": 99}.get(token, 0)


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, return_dict=False
    ):
        question = messages[0]["content"][1]["text"]
        if not tokenize:
            return f"<image>{question}"
        suffix = 20 if question.endswith("one?") else 21
        return {"input_ids": [[10, 99, 99, 11, suffix]]}


def make_vqav2(root: Path) -> Path:
    dataset = root / "vqav2"
    images = dataset / "val2014"
    images.mkdir(parents=True)
    payload = {
        "questions": [
            {"question_id": 3, "image_id": 20, "question": "Image twenty?"},
            {"question_id": 1, "image_id": 10, "question": "Question one?"},
            {"question_id": 2, "image_id": 10, "question": "Question two?"},
        ]
    }
    (dataset / "v2_OpenEnded_mscoco_val2014_questions.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (images / "COCO_val2014_000000000010.jpg").write_bytes(b"image-ten")
    (images / "COCO_val2014_000000000020.jpg").write_bytes(b"image-twenty")
    return dataset


def make_chartqa(root: Path) -> Path:
    dataset = root / "chartqa"
    split = dataset / "ChartQA Dataset" / "test"
    images = split / "png"
    images.mkdir(parents=True)
    (split / "test_human.json").write_text(
        json.dumps(
            [
                {"imgname": "chart-a.png", "query": "Question one?", "label": "1"},
                {"imgname": "chart-a.png", "query": "Question two?", "label": "2"},
            ]
        ),
        encoding="utf-8",
    )
    (split / "test_augmented.json").write_text(
        json.dumps(
            [{"imgname": "chart-b.png", "query": "Image twenty?", "label": "yes"}]
        ),
        encoding="utf-8",
    )
    (images / "chart-a.png").write_bytes(b"chart-a")
    (images / "chart-b.png").write_bytes(b"chart-b")
    return dataset


def make_mme(root: Path) -> Path:
    dataset = root / "mme" / "MME_Benchmark_release_version"
    categories = {
        "perception": (
            "existence",
            "count",
            "position",
            "color",
            "posters",
            "celebrity",
            "scene",
            "landmark",
            "artwork",
            "OCR",
        ),
        "cognition": (
            "commonsense_reasoning",
            "numerical_calculation",
            "text_translation",
            "code_reasoning",
        ),
    }
    for domain, domain_categories in categories.items():
        for category in domain_categories:
            stem = f"{domain}-{category}"
            category_dir = dataset / domain / category
            images = category_dir / "images"
            questions = category_dir / "questions_answers_YN"
            images.mkdir(parents=True)
            questions.mkdir()
            (images / f"{stem}.jpg").write_bytes(stem.encode())
            (questions / f"{stem}.txt").write_text(
                "Is this question one?\tYes\nIs this question two?\tNo\n",
                encoding="utf-8",
            )
    return root / "mme"


def make_mme_flat_files(root: Path) -> Path:
    dataset = root / "mme" / "MME_Benchmark_release_version"
    for domain, categories in MME_CATEGORIES.items():
        for category in categories:
            stem = f"{domain}-{category}"
            category_dir = dataset / domain / category
            category_dir.mkdir(parents=True)
            (category_dir / f"{stem}.jpg").write_bytes(stem.encode())
            (category_dir / f"{stem}.txt").write_text(
                "Is this question one?\tYes\nIs this question two?\tNo\n",
                encoding="utf-8",
            )
    return root / "mme"


class DatasetAdapterTests(unittest.TestCase):
    def test_registry_contains_first_party_adapters(self):
        self.assertEqual(adapter_names(), ("chartqa", "mme", "vqav2"))

    def test_vqav2_image_sampling_keeps_all_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = make_vqav2(root)
            manifest = build_manifest(get_adapter("vqav2"), dataset, "val", 42)
        self.assertEqual(manifest["num_media"], 2)
        self.assertEqual(manifest["num_samples"], 3)
        self.assertEqual({item["dataset"] for item in manifest["records"]}, {"vqav2"})

    def test_chartqa_combines_human_and_augmented(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_chartqa(Path(directory))
            manifest = build_manifest(get_adapter("chartqa"), dataset, "test", 42)
        self.assertEqual(manifest["num_samples"], 3)
        self.assertEqual(
            {item["category"] for item in manifest["records"]}, {"human", "augmented"}
        )
        self.assertTrue(
            all(item["prompt"].startswith("Answer the question using the chart.") for item in manifest["records"])
        )

    def test_mme_defaults_to_full_and_limit_is_stratified(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_mme(Path(directory))
            adapter = get_adapter("mme")
            full = build_manifest(adapter, dataset, "all", 42)
        self.assertEqual(full["num_media"], 14)
        self.assertEqual(full["num_samples"], 28)

    def test_mme_supports_questions_and_images_beside_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_mme_flat_files(Path(directory))
            manifest = build_manifest(get_adapter("mme"), dataset, "all", 42)
        self.assertEqual(manifest["num_media"], 14)
        self.assertEqual(manifest["num_samples"], 28)
        self.assertTrue(
            all("questions_answers_YN" not in row["image_path"] for row in manifest["records"])
        )

    def test_mme_requires_all_fourteen_official_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_mme(Path(directory))
            missing = (
                dataset
                / "MME_Benchmark_release_version"
                / "perception"
                / "existence"
                / "questions_answers_YN"
            )
            for path in missing.iterdir():
                path.unlink()
            missing.rmdir()
            with self.assertRaisesRegex(FileNotFoundError, "perception/existence"):
                build_manifest(get_adapter("mme"), dataset, "all", 42)

    def test_manifest_checksum_excludes_absolute_root(self):
        with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
            left = build_manifest(
                get_adapter("vqav2"), make_vqav2(Path(left_dir)), "val", 42
            )
            right = build_manifest(
                get_adapter("vqav2"), make_vqav2(Path(right_dir)), "val", 42
            )
        self.assertEqual(left["records_sha256"], right["records_sha256"])

    def test_validation_rejects_duplicate_ids_and_missing_images(self):
        sample = MultimodalSample("demo", "same", "m", "q", "/missing.jpg")
        with self.assertRaisesRegex(ValueError, "重复 sample_id"):
            validate_samples([sample, sample], "demo")


class MultimodalCacheKeyTests(unittest.TestCase):
    def _requests(self):
        samples = [
            ManifestSample(
                MultimodalSample("chartqa", "1", "a", "Question one?", "/a.png"),
                "Question one?",
                "hash-a",
            ),
            ManifestSample(
                MultimodalSample("chartqa", "2", "a", "Question two?", "/a.png"),
                "Question two?",
                "hash-a",
            ),
            ManifestSample(
                MultimodalSample("chartqa", "3", "b", "Question one?", "/b.png"),
                "Question one?",
                "hash-b",
            ),
        ]
        return prepare_requests(samples, FakeProcessor())

    def test_same_media_shares_visual_keys_and_dataset_request_ids(self):
        requests = self._requests()
        self.assertEqual(requests[0].cache_keys[1:3], requests[1].cache_keys[1:3])
        self.assertNotEqual(requests[0].cache_keys[1], requests[2].cache_keys[1])
        self.assertTrue(all(request.request_id[0] == "chartqa" for request in requests))
        root = XXXTrieNode.build_vertical(
            {"chartqa": [list(request.cache_keys) for request in requests]}
        )
        first = root.children[("text", 10)]
        self.assertIn(("image", 99, "hash-a"), first.children)
        self.assertIn(("chartqa", 2), first.request_ids)

    def test_ordering_and_reuse_opportunity(self):
        requests = self._requests()
        opportunity = encoder_reuse_opportunity(requests)
        self.assertEqual(opportunity["potential_hits"], 1)

    def test_fixed_media_cap_keeps_all_questions(self):
        self.assertEqual(MAX_MEDIA, 10_000)
        samples = [
            MultimodalSample("chartqa", f"{media}:{question}", str(media), "q", "/x")
            for media in range(4)
            for question in range(2)
        ]
        with mock.patch.object(multimodal_module, "MAX_MEDIA", 3):
            selected, media_ids = sample_by_media(
                samples, get_adapter("chartqa"), seed=42
            )
            repeated, repeated_ids = sample_by_media(
                samples, get_adapter("chartqa"), seed=42
            )
        self.assertEqual(len(media_ids), 3)
        self.assertEqual(len(selected), 6)
        self.assertEqual(media_ids, repeated_ids)
        self.assertEqual(selected, repeated)


class VLLMAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_same_native_apc_contract_as_text_experiment(self):
        copied_images = []

        class FakeImage:
            size = (40, 30)

            def load(self):
                return None

            def copy(self):
                copied = types.SimpleNamespace(closed=False)
                copied.close = lambda: setattr(copied, "closed", True)
                copied_images.append(copied)
                return copied

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

        fake_image_module = types.SimpleNamespace(open=lambda path: FakeImage())

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeEngineArgs:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeEngine:
            instance = None

            def __init__(self, args):
                self.args = args
                self.sampler_env = run_experiment.os.environ.get(
                    "VLLM_USE_FLASHINFER_SAMPLER"
                )
                self.model_runner_env = run_experiment.os.environ.get(
                    "VLLM_USE_V2_MODEL_RUNNER"
                )
                self.tokenizer_parallelism_env = run_experiment.os.environ.get(
                    "TOKENIZERS_PARALLELISM"
                )
                self.shutdown_called = False
                type(self).instance = self

            @classmethod
            def from_engine_args(cls, args):
                return cls(args)

            async def generate(self, prompt, sampling, request_id):
                yield types.SimpleNamespace(
                    finished=True,
                    num_cached_tokens=2,
                    num_cache_creation_tokens=3,
                )

            def shutdown(self):
                self.shutdown_called = True

        fake_vllm = types.SimpleNamespace(
            AsyncEngineArgs=FakeEngineArgs,
            AsyncLLMEngine=FakeEngine,
            SamplingParams=FakeSamplingParams,
        )

        def import_module(name):
            if name == "vllm":
                return fake_vllm
            if name == "PIL.Image":
                return fake_image_module
            raise ImportError(name)

        snapshot = run_experiment._VLLMMetricSnapshot(
            values={}, cache_configs=({"kv_cache_size_tokens": "1000"},)
        )
        requests = MultimodalCacheKeyTests()._requests()[:2]
        with (
            mock.patch.dict(
                run_experiment.os.environ,
                {"VLLM_USE_FLASHINFER_SAMPLER": "1"},
            ),
            mock.patch(
                "run_multimodal_experiment.importlib.import_module",
                side_effect=import_module,
            ),
            mock.patch.object(
                run_experiment, "validate_vllm_version", return_value="0.26.0"
            ),
            mock.patch.object(
                run_experiment,
                "_run_vllm_apc_self_test",
                new=mock.AsyncMock(
                    return_value={"passed": True, "observed_hit_tokens": 16}
                ),
            ),
            mock.patch.object(
                run_experiment, "_read_vllm_metric_snapshot", return_value=snapshot
            ),
        ):
            result = await run_vllm(
                None,
                model_path="fake-model",
                context_length=4096,
                batch_size=2,
                gpu_memory_utilization=0.8,
                eos_token_id=2,
                request_factory=lambda: requests,
            )

        engine = FakeEngine.instance
        self.assertEqual(engine.sampler_env, "0")
        self.assertEqual(engine.model_runner_env, "0")
        self.assertEqual(engine.tokenizer_parallelism_env, "false")
        self.assertEqual(engine.args.kwargs["block_size"], 16)
        self.assertNotIn("prefix_match_unit", engine.args.kwargs)
        self.assertTrue(engine.args.kwargs["enable_prefix_caching"])
        self.assertFalse(engine.args.kwargs["async_scheduling"])
        self.assertEqual(
            engine.args.kwargs["limit_mm_per_prompt"],
            {"image": 1, "video": 0},
        )
        self.assertTrue(engine.args.kwargs["skip_mm_profiling"])
        self.assertNotIn("mm_processor_kwargs", engine.args.kwargs)
        self.assertTrue(engine.shutdown_called)
        self.assertTrue(all(image.closed for image in copied_images))
        self.assertEqual(result["kv_cache"]["cache_hit_tokens"], 4)
        self.assertEqual(result["kv_cache"]["cache_creation_tokens"], 6)
        self.assertEqual(result["backend_metrics"]["cache_match_mode"], "block")
        self.assertEqual(result["backend_metrics"]["block_size"], 16)
        self.assertEqual(
            result["backend_metrics"]["observed_max_image_pixels"], 1200
        )
        self.assertTrue(result["backend_metrics"]["skip_mm_profiling"])
        self.assertEqual(result["backend_metrics"]["model_runner"], "v1")
        self.assertFalse(result["backend_metrics"]["async_scheduling"])
        self.assertFalse(result["backend_metrics"]["tokenizers_parallelism"])
        self.assertEqual(
            result["backend_metrics"]["cache_self_test"]["observed_hit_tokens"],
            16,
        )
        self.assertFalse(result["backend_metrics"]["flashinfer_sampler_enabled"])


class SGLangAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_passes_media_hash_and_cstrie_cache_policy(self):
        class FakeEngine:
            def __init__(self):
                self.calls = []

            async def async_generate(self, **kwargs):
                self.calls.append(kwargs)
                return {"text": "x"}

        requests = MultimodalCacheKeyTests()._requests()[:2]
        batches = [
            [ScheduledRequest(requests[0].request_id, "prefill", 4)],
            [ScheduledRequest(requests[1].request_id, "normal")],
        ]
        engine = FakeEngine()
        latencies = await _run_sglang_requests(
            engine, requests, batch_size=2, eos_token_id=2, scheduled_batches=batches
        )
        self.assertEqual(len(latencies), 2)
        self.assertEqual(engine.calls[0]["mm_hashes"], ["hash-a"])
        self.assertEqual(engine.calls[0]["image_data"], Path("/a.png").as_uri())
        self.assertEqual(
            engine.calls[0]["sampling_params"]["custom_params"],
            {"custom_cache_prefix_len": 4},
        )
        self.assertIn("chartqa", engine.calls[0]["rid"])
        self.assertIn("bootstrap_host", engine.calls[1])

    def test_relative_image_uri_is_portable_across_working_directories(self):
        relative = Path("data/chartqa/ChartQA Dataset/test/png/chart 1.png")
        with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
            left = Path(left_dir)
            right = Path(right_dir)
            with chdir(left):
                left_uri = _sglang_image_uri(str(relative))
            with chdir(right):
                right_uri = _sglang_image_uri(str(relative))

        self.assertEqual(left_uri, (left / relative).resolve().as_uri())
        self.assertEqual(right_uri, (right / relative).resolve().as_uri())
        self.assertNotEqual(left_uri, right_uri)
        self.assertIn("ChartQA%20Dataset", left_uri)
        self.assertIn("chart%201.png", left_uri)

    async def test_parses_optional_encoder_cache_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(
                "\n".join(
                    (
                        json.dumps({"encoder_cache_hit": False}),
                        json.dumps({"encoder_cache_hits": 3, "encoder_cache_misses": 1}),
                    )
                ),
                encoding="utf-8",
            )
            metrics = parse_sglang_encoder_metrics(path)
        self.assertTrue(metrics["available"])
        self.assertEqual(metrics["hit_rate"], 0.75)


class CLITests(unittest.TestCase):
    def test_prepare_only_builds_independent_dataset_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_vqav2(root)
            make_chartqa(root)
            make_mme(root)
            output = root / "output"
            completed = subprocess.run(
                [
                    sys.executable,
                    "run_multimodal_experiment.py",
                    "--datasets",
                    "vqav2",
                    "chartqa",
                    "mme",
                    "--data-root",
                    str(root),
                    "--output-dir",
                    str(output),
                    "--prepare-only",
                ],
                cwd=".",
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            for dataset in ("vqav2", "chartqa", "mme"):
                manifests = list(output.glob(f"*/*/artifacts/{dataset}/manifest.json"))
                self.assertEqual(len(manifests), 1)
            self.assertEqual(len(list(output.glob("*/*/result.json"))), 1)
            self.assertTrue((output / "results_report.md").is_file())

    def test_dataset_path_and_split_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chartqa = make_chartqa(root)
            output = root / "output"
            completed = subprocess.run(
                [
                    sys.executable,
                    "run_multimodal_experiment.py",
                    "--datasets",
                    "chartqa",
                    "--data-root",
                    str(root / "does-not-exist"),
                    "--dataset-path",
                    f"chartqa={chartqa}",
                    "--split",
                    "chartqa=test",
                    "--output-dir",
                    str(output),
                    "--prepare-only",
                ],
                cwd=".",
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifests = list(output.glob("chartqa/*/artifacts/chartqa/manifest.json"))
            self.assertEqual(len(manifests), 1)

    def test_old_vqa_entrypoint_is_removed(self):
        self.assertFalse(Path("run_vqa_experiment.py").exists())


if __name__ == "__main__":
    unittest.main()

"""Run reproducible multimodal cache experiments with Qwen3-VL.

The script intentionally keeps multimodal dependencies lazy so manifest creation and
unit tests work on machines without a GPU inference stack.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import run_experiment
from scheduler import ScheduledRequest, schedule_heuristic
from multimodal_experiment import (
    PreparedMultimodalRequest,
    adapter_names,
    aggregate_runs,
    build_manifest,
    encoder_reuse_opportunity,
    get_adapter,
    manifest_samples,
    prepare_requests,
    summarize_latencies,
    write_manifest,
)
from xxxtrie import XXXTrieNode

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_OUTPUT_DIR = Path("results/multimodal")


def _max_image_pixels(
    requests: Sequence[PreparedMultimodalRequest], image_module: Any
) -> int:
    """Return the real workload's largest image size for result metadata."""
    maximum = 0
    visited = set()
    for request in requests:
        if request.image_sha256 in visited:
            continue
        visited.add(request.image_sha256)
        with image_module.open(request.image_path) as image:
            width, height = image.size
        maximum = max(maximum, int(width) * int(height))
    if maximum <= 0:
        raise ValueError("vLLM 多模态实验至少需要一张有效图片")
    return maximum


def load_processor(model_path: str) -> Any:
    try:
        transformers = importlib.import_module("transformers")
        processor_cls = transformers.AutoProcessor
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("运行多模态推理需要安装支持 Qwen3-VL 的 transformers") from exc
    return processor_cls.from_pretrained(model_path, trust_remote_code=True)


def _sampling_params(eos_token_id: Optional[int], cstrie_prefix: Optional[int] = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "max_new_tokens": 1,
        "temperature": 0.0,
        "skip_special_tokens": True,
    }
    if eos_token_id is not None:
        params["stop_token_ids"] = [int(eos_token_id)]
    if cstrie_prefix is not None:
        params["custom_params"] = {"custom_cache_prefix_len": cstrie_prefix}
    return params


def _sglang_image_uri(image_path: str) -> str:
    """Convert a local image path to SGLang's portable file URI format."""
    return Path(image_path).expanduser().resolve().as_uri()


async def _run_sglang_requests(
    llm: Any,
    requests: Sequence[PreparedMultimodalRequest],
    batch_size: int,
    eos_token_id: Optional[int],
    scheduled_batches: Optional[Sequence[Sequence[ScheduledRequest]]] = None,
) -> List[float]:
    by_id = {request.request_id: request for request in requests}
    latencies: List[float] = []
    semaphore = asyncio.Semaphore(batch_size)

    async def send(
        request: PreparedMultimodalRequest, task: Optional[ScheduledRequest]
    ) -> None:
        async with semaphore:
            prefix = task.cache_prefix_len if task and task.kind == "prefill" else None
            params = _sampling_params(eos_token_id, prefix)
            kwargs: Dict[str, Any] = {}
            if task and task.kind == "normal":
                kwargs["bootstrap_host"] = run_experiment._FAKE_BOOTSTRAP_HOST
            started = time.perf_counter()
            await llm.async_generate(
                prompt=request.prompt,
                image_data=_sglang_image_uri(request.image_path),
                mm_hashes=[request.image_sha256],
                sampling_params=params,
                stream=False,
                rid=(
                    f"{task.kind[0].upper()}:{request.dataset}:{request.sample_id}"
                    if task
                    else f"{request.dataset}:{request.sample_id}"
                ),
                **kwargs,
            )
            latencies.append(time.perf_counter() - started)

    if scheduled_batches is None:
        for start in range(0, len(requests), batch_size):
            await asyncio.gather(*(send(request, None) for request in requests[start : start + batch_size]))
    else:
        for batch in scheduled_batches:
            await asyncio.gather(*(send(by_id[task.request_id], task) for task in batch))
    return latencies


async def run_sglang(
    requests: Sequence[PreparedMultimodalRequest],
    model_path: str,
    context_length: int,
    batch_size: int,
    gpu_memory_utilization: float,
    metrics_log: Path,
    use_cstrie: bool,
    eos_token_id: Optional[int],
) -> Dict[str, Any]:
    sgl = run_experiment._load_sglang()
    metrics_log.parent.mkdir(parents=True, exist_ok=True)
    metrics_log.unlink(missing_ok=True)
    os.environ["SGLANG_CUSTOM_METRICS_LOG"] = str(metrics_log)
    os.environ["SGLANG_CACHE_EXP_MODE"] = "1"

    scheduled_batches: Optional[List[List[ScheduledRequest]]] = None
    trie_info: Optional[Dict[str, Any]] = None
    if use_cstrie:
        dataset = requests[0].dataset if requests else "multimodal"
        seq_map = {dataset: [list(request.cache_keys) for request in requests]}
        root = XXXTrieNode.build_vertical(seq_map)
        scheduled_batches = schedule_heuristic(root, batch_size)
        tasks = [task for batch in scheduled_batches for task in batch]
        if len(tasks) != len(requests) or {task.request_id for task in tasks} != {
            request.request_id for request in requests
        }:
            raise AssertionError("CSTrie 调度未完整覆盖多模态请求")
        trie_info = {
            "num_nodes_with_requests": len(root.collect_leaves()),
            "num_batches": len(scheduled_batches),
            "num_prefill_requests": sum(task.kind == "prefill" for task in tasks),
        }

    llm = sgl.Engine(
        model_path=model_path,
        tp_size=1,
        mem_fraction_static=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="auto",
        context_length=context_length,
        max_running_requests=max(batch_size, 4),
        chunked_prefill_size=256,
        disable_cuda_graph=True,
        disable_radix_cache=False,
        log_level="info",
        attention_backend="triton",
    )
    started = time.perf_counter()
    try:
        latencies = await _run_sglang_requests(
            llm, requests, batch_size, eos_token_id, scheduled_batches
        )
        elapsed = time.perf_counter() - started
    finally:
        llm.shutdown()
    parsed = run_experiment.parse_metrics_log(str(metrics_log), rid_prefix="")
    encoder_metrics = parse_sglang_encoder_metrics(metrics_log)
    return _build_run_result(
        requests,
        elapsed,
        latencies,
        cache_metrics=parsed,
        backend_metrics={
            "sglang_metrics_log": str(metrics_log),
            "encoder_cache": encoder_metrics,
        },
        trie_info=trie_info,
    )


def _snapshot_deltas(before: Any, after: Any, keyword: str) -> Dict[str, float]:
    names = set(before.values) | set(after.values)
    return {
        name: after.values.get(name, 0.0) - before.values.get(name, 0.0)
        for name in sorted(names)
        if keyword in name.lower()
    }


def parse_sglang_encoder_metrics(path: Path) -> Dict[str, Any]:
    """Parse the optional encoder-cache contract emitted by the custom fork."""
    if not path.is_file():
        return {"available": False}
    event_hits = 0
    event_misses = 0
    cumulative_hits: Optional[int] = None
    cumulative_misses: Optional[int] = None
    peak_entries: Optional[int] = None
    capacity_embeddings: Optional[int] = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(event, dict):
                continue
            if isinstance(event.get("encoder_cache_hit"), bool):
                if event["encoder_cache_hit"]:
                    event_hits += 1
                else:
                    event_misses += 1
            numeric = {
                key: int(event[key])
                for key in (
                    "encoder_cache_hits",
                    "encoder_cache_misses",
                    "encoder_cache_entries",
                    "encoder_cache_capacity_embeddings",
                )
                if isinstance(event.get(key), (int, float))
            }
            if "encoder_cache_hits" in numeric:
                cumulative_hits = max(cumulative_hits or 0, numeric["encoder_cache_hits"])
            if "encoder_cache_misses" in numeric:
                cumulative_misses = max(
                    cumulative_misses or 0, numeric["encoder_cache_misses"]
                )
            if "encoder_cache_entries" in numeric:
                peak_entries = max(peak_entries or 0, numeric["encoder_cache_entries"])
            if "encoder_cache_capacity_embeddings" in numeric:
                capacity_embeddings = max(
                    capacity_embeddings or 0,
                    numeric["encoder_cache_capacity_embeddings"],
                )
    hits = cumulative_hits if cumulative_hits is not None else event_hits
    misses = cumulative_misses if cumulative_misses is not None else event_misses
    available = any(
        value is not None
        for value in (
            cumulative_hits,
            cumulative_misses,
            peak_entries,
            capacity_embeddings,
        )
    ) or bool(event_hits or event_misses)
    total = hits + misses
    return {
        "available": available,
        "hits": hits if available else None,
        "misses": misses if available else None,
        "hit_rate": hits / total if available and total else None,
        "peak_entries": peak_entries,
        "capacity_embeddings": capacity_embeddings,
    }


async def run_vllm(
    requests: Optional[Sequence[PreparedMultimodalRequest]],
    model_path: str,
    context_length: int,
    batch_size: int,
    gpu_memory_utilization: float,
    eos_token_id: Optional[int],
    request_factory: Optional[Callable[[], Sequence[PreparedMultimodalRequest]]] = None,
) -> Dict[str, Any]:
    run_experiment._configure_vllm_native_runtime()
    version = run_experiment.validate_vllm_version()
    try:
        vllm = importlib.import_module("vllm")
        image_module = importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise RuntimeError("vLLM 多模态实验需要 vllm 0.26.x 和 Pillow") from exc

    engine_args = vllm.AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=context_length,
        max_num_seqs=batch_size,
        async_scheduling=False,
        block_size=run_experiment.VLLM_BLOCK_SIZE,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 1, "video": 0},
        # vLLM 0.26 can hang indefinitely while profiling Qwen3-VL's encoder.
        # Skipping that startup-only estimate does not alter real image inputs.
        skip_mm_profiling=True,
        trust_remote_code=True,
        dtype="auto",
        enforce_eager=True,
        disable_log_stats=False,
    )
    llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
    if eos_token_id is None:
        shutdown = llm.shutdown()
        if hasattr(shutdown, "__await__"):
            await shutdown
        raise ValueError("processor.tokenizer.eos_token_id 不能为 None")
    try:
        cache_self_test = await run_experiment._run_vllm_apc_self_test(
            llm, vllm.SamplingParams, int(eos_token_id)
        )
    except Exception:
        shutdown = llm.shutdown()
        if hasattr(shutdown, "__await__"):
            await shutdown
        raise
    try:
        if requests is None:
            if request_factory is None:
                raise ValueError("requests 和 request_factory 不能同时为 None")
            requests = list(request_factory())
        observed_max_image_pixels = _max_image_pixels(requests, image_module)
    except Exception:
        shutdown = llm.shutdown()
        if hasattr(shutdown, "__await__"):
            await shutdown
        raise
    sampling_kwargs: Dict[str, Any] = {
        "max_tokens": 1,
        "temperature": 0.0,
        "skip_special_tokens": True,
    }
    if eos_token_id is not None:
        sampling_kwargs["stop_token_ids"] = [int(eos_token_id)]
    sampling = vllm.SamplingParams(**sampling_kwargs)
    before = run_experiment._read_vllm_metric_snapshot()
    peak_kv_usage = run_experiment._metric_value(before, "kv_cache_usage_perc")
    stop_sampling = asyncio.Event()
    latencies: List[float] = []
    request_metrics: List[run_experiment._VLLMRequestMetrics] = []

    async def sample_kv_usage() -> None:
        nonlocal peak_kv_usage
        while not stop_sampling.is_set():
            snapshot = run_experiment._read_vllm_metric_snapshot()
            peak_kv_usage = max(
                peak_kv_usage,
                run_experiment._metric_value(snapshot, "kv_cache_usage_perc"),
            )
            try:
                await asyncio.wait_for(stop_sampling.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass

    async def send(
        request: PreparedMultimodalRequest, images: Mapping[str, Any]
    ) -> run_experiment._VLLMRequestMetrics:
        prompt = {
            "prompt": request.prompt,
            "multi_modal_data": {"image": images[request.image_sha256]},
        }
        started = time.perf_counter()
        final = None
        async for output in llm.generate(
            prompt,
            sampling,
            request_id=f"{request.dataset}:{request.sample_id}",
        ):
            if bool(getattr(output, "finished", False)):
                final = output
        latencies.append(time.perf_counter() - started)
        if final is None:
            raise RuntimeError(f"sample_id={request.sample_id} 没有 finished 输出")
        cached_value = getattr(final, "num_cached_tokens", None)
        creation_value = getattr(final, "num_cache_creation_tokens", None)
        return run_experiment._VLLMRequestMetrics(
            request_id=f"{request.dataset}:{request.sample_id}",
            input_tokens=len(request.input_ids),
            cached_tokens=max(0, int(cached_value or 0)),
            cache_creation_tokens=max(0, int(creation_value or 0)),
            missing_cached_tokens=cached_value is None,
            missing_cache_creation_tokens=creation_value is None,
        )

    started = time.perf_counter()
    sampler_task = asyncio.create_task(sample_kv_usage())
    try:
        for offset in range(0, len(requests), batch_size):
            batch = requests[offset : offset + batch_size]
            images: Dict[str, Any] = {}
            try:
                for request in batch:
                    if request.image_sha256 not in images:
                        with image_module.open(request.image_path) as image:
                            image.load()
                            images[request.image_sha256] = image.copy()
                request_metrics.extend(
                    await asyncio.gather(*(send(request, images) for request in batch))
                )
            finally:
                for image in images.values():
                    image.close()
        elapsed = time.perf_counter() - started
    finally:
        stop_sampling.set()
        await sampler_task
        # 进程内 Prometheus collector 可能在 shutdown 时注销，必须先取快照。
        after = run_experiment._read_vllm_metric_snapshot()
        peak_kv_usage = max(
            peak_kv_usage,
            run_experiment._metric_value(after, "kv_cache_usage_perc"),
        )
        shutdown = llm.shutdown()
        if hasattr(shutdown, "__await__"):
            await shutdown

    total_input_tokens = sum(item.input_tokens for item in request_metrics)
    cached_tokens = sum(item.cached_tokens for item in request_metrics)
    created_tokens = sum(item.cache_creation_tokens for item in request_metrics)
    per_request_hit_rates = [
        item.cached_tokens / item.input_tokens
        for item in request_metrics
        if item.input_tokens
    ]
    micro_hit_rate = cached_tokens / total_input_tokens if total_input_tokens else 0.0
    macro_hit_rate = (
        sum(per_request_hit_rates) / len(per_request_hit_rates)
        if per_request_hit_rates
        else 0.0
    )
    missing_cached_tokens = sum(item.missing_cached_tokens for item in request_metrics)
    missing_creation_tokens = sum(
        item.missing_cache_creation_tokens for item in request_metrics
    )
    encoder_metrics = _snapshot_deltas(before, after, "encoder_cache")
    vllm_metric_deltas = {
        name: after.values.get(name, 0.0) - before.values.get(name, 0.0)
        for name in sorted(set(before.values) | set(after.values))
        if any(term in name.lower() for term in ("prefix_cache", "prompt_token", "encoder_cache"))
    }
    cache_capacity_tokens = run_experiment._vllm_cache_config_int(
        (after, before), "kv_cache_size_tokens"
    )
    cache_capacity_bytes = run_experiment._vllm_cache_config_int(
        (after, before), "kv_cache_memory_bytes"
    )
    cache_queries = max(
        0.0,
        run_experiment._metric_value(after, "prefix_cache_queries")
        - run_experiment._metric_value(before, "prefix_cache_queries"),
    )
    cache_hits = max(
        0.0,
        run_experiment._metric_value(after, "prefix_cache_hits")
        - run_experiment._metric_value(before, "prefix_cache_hits"),
    )
    peak_cache_tokens = (
        round(cache_capacity_tokens * peak_kv_usage)
        if cache_capacity_tokens is not None
        else None
    )
    peak_cache_bytes = (
        round(cache_capacity_bytes * peak_kv_usage)
        if cache_capacity_bytes is not None
        else None
    )
    return _build_run_result(
        requests,
        elapsed,
        latencies,
        cache_metrics={
            "cache_hit_tokens": cached_tokens,
            "cache_creation_tokens": created_tokens,
            "cache_capacity_tokens": cache_capacity_tokens,
            "cache_capacity_bytes": cache_capacity_bytes,
            "peak_cache_tokens": peak_cache_tokens,
            "peak_cache_bytes": peak_cache_bytes,
            "aggregate_hit_rate_micro": micro_hit_rate,
            "aggregate_hit_rate_micro_percent": micro_hit_rate * 100.0,
            "aggregate_hit_rate_macro": macro_hit_rate,
            "aggregate_hit_rate_macro_percent": macro_hit_rate * 100.0,
            "total_input_tokens_measured": total_input_tokens,
            "total_hit_tokens_measured": cached_tokens,
        },
        backend_metrics={
            "vllm_version": version,
            "cache_policy": "native",
            "cache_match_mode": "block",
            "block_size": run_experiment.VLLM_BLOCK_SIZE,
            "skip_mm_profiling": True,
            "model_runner": "v1",
            "async_scheduling": False,
            "tokenizers_parallelism": False,
            "observed_max_image_pixels": observed_max_image_pixels,
            "cache_granularity_tokens": run_experiment.VLLM_BLOCK_SIZE,
            "sampler_backend": "native",
            "flashinfer_sampler_enabled": False,
            "cache_self_test": cache_self_test,
            "prefix_cache_query_tokens": int(cache_queries),
            "prefix_cache_hit_tokens": int(cache_hits),
            "native_micro_hit_rate": cache_hits / cache_queries if cache_queries else 0.0,
            "peak_kv_cache_usage": peak_kv_usage,
            "peak_kv_cache_usage_percent": peak_kv_usage * 100.0,
            "missing_num_cached_tokens": missing_cached_tokens,
            "missing_num_cache_creation_tokens": missing_creation_tokens,
            "encoder_cache_metrics_available": bool(encoder_metrics),
            "encoder_cache_metric_deltas": encoder_metrics,
            "metric_deltas": vllm_metric_deltas,
        },
    )


def _build_run_result(
    requests: Sequence[PreparedMultimodalRequest],
    elapsed: float,
    latencies: Sequence[float],
    cache_metrics: Mapping[str, Any],
    backend_metrics: Mapping[str, Any],
    trie_info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    prompt_tokens = sum(len(request.input_ids) for request in requests)
    return {
        "status": "ok",
        "elapsed_seconds": elapsed,
        "num_requests": len(requests),
        "total_prompt_tokens": prompt_tokens,
        "requests_per_second": len(requests) / elapsed if elapsed else 0.0,
        "prompt_tokens_per_second": prompt_tokens / elapsed if elapsed else 0.0,
        "ttft_proxy": {
            **summarize_latencies(latencies),
            "note": "max_new_tokens=1；完成时延作为 TTFT 的近似值",
        },
        "kv_cache": dict(cache_metrics),
        "encoder_cache": {
            **encoder_reuse_opportunity(requests),
            "note": "potential_hits 是 workload 理论机会；实测计数仅见 backend_metrics",
        },
        "backend_metrics": dict(backend_metrics),
        "trie": dict(trie_info) if trie_info is not None else None,
    }


def _write_report(result: Mapping[str, Any], path: Path) -> None:
    lines = [
        f"# {result['manifest']['dataset']} Multimodal Cache Experiment",
        "",
        f"- Model: `{result['config']['model_path']}`",
        f"- Media: {result['manifest']['num_media']}",
        f"- Samples: {result['manifest']['num_samples']}",
        f"- Manifest SHA-256: `{result['manifest']['records_sha256']}`",
        "",
        "| Order | Backend | Successful runs | Median seconds | Median requests/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for order, backends in result["experiments"].items():
        for backend, value in backends.items():
            summary = value["summary"]
            median = summary["median"]
            lines.append(
                f"| {order} | {backend} | {summary['num_successful']}/{summary['num_runs']} "
                f"| {median['elapsed_seconds'] if median['elapsed_seconds'] is not None else 'N/A'} "
                f"| {median['requests_per_second'] if median['requests_per_second'] is not None else 'N/A'} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_overrides(values: Sequence[str], option: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for value in values:
        name, separator, configured = value.partition("=")
        if not separator or not name or not configured:
            raise ValueError(f"{option} 必须使用 NAME=VALUE 格式: {value!r}")
        if name in result:
            raise ValueError(f"{option} 重复指定数据集 {name!r}")
        result[name] = configured
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可扩展的多模态缓存性能实验")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument(
        "--datasets", nargs="+", choices=adapter_names(), default=adapter_names()
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--dataset-path", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--split", action="append", default=[], metavar="NAME=SPLIT")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("sglang", "cstrie", "vllm"),
        default=("sglang", "cstrie", "vllm"),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.repetitions <= 0 or args.batch_size <= 0:
        parser.error("--repetitions 和 --batch-size 必须大于 0")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization 必须在 (0, 1] 范围内")
    try:
        args.dataset_paths = _parse_overrides(args.dataset_path, "--dataset-path")
        args.splits = _parse_overrides(args.split, "--split")
    except ValueError as exc:
        parser.error(str(exc))
    selected = set(args.datasets)
    for option, overrides in (
        ("--dataset-path", args.dataset_paths),
        ("--split", args.splits),
    ):
        unknown = set(overrides) - selected
        if unknown:
            parser.error(f"{option} 指定了未选择的数据集: {sorted(unknown)}")
    return args


def _dataset_root(args: argparse.Namespace, dataset: str) -> Path:
    configured = args.dataset_paths.get(dataset)
    return Path(configured) if configured is not None else args.data_root / dataset


def _write_cross_dataset_summary(
    results: Mapping[str, Mapping[str, Any]], output_dir: Path
) -> None:
    summary = {
        dataset: {
            "manifest": result["manifest"],
            "experiments": {
                order: {
                    backend: value["summary"]
                    for backend, value in backends.items()
                }
                for order, backends in result.get("experiments", {}).items()
            },
        }
        for dataset, result in results.items()
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Multimodal Cache Experiment Summary",
        "",
        "| Dataset | Order | Backend | Successful runs | Median seconds | Median requests/s |",
        "|---|---|---|---:|---:|---:|",
    ]
    for dataset, result in results.items():
        for order, backends in result.get("experiments", {}).items():
            for backend, value in backends.items():
                item = value["summary"]
                median = item["median"]
                lines.append(
                    f"| {dataset} | {order} | {backend} "
                    f"| {item['num_successful']}/{item['num_runs']} "
                    f"| {median['elapsed_seconds'] if median['elapsed_seconds'] is not None else 'N/A'} "
                    f"| {median['requests_per_second'] if median['requests_per_second'] is not None else 'N/A'} |"
                )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def _run_dataset(
    args: argparse.Namespace, manifest: Mapping[str, Any], processor: Any
) -> tuple[Dict[str, Any], bool]:
    dataset = str(manifest["dataset"])
    dataset_dir = args.output_dir / dataset
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    result: Dict[str, Any] = {
        "config": {
            "model_path": args.model_path,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "seed": args.seed,
            "repetitions": args.repetitions,
            "orders": ["grouped"],
            "backends": list(args.backends),
        },
        "manifest": {key: value for key, value in manifest.items() if key != "records"},
        "experiments": {},
    }
    result_path = dataset_dir / "results.json"
    has_errors = False
    order = "grouped"
    samples = manifest_samples(manifest)
    requests: Optional[List[PreparedMultimodalRequest]] = None
    result["experiments"][order] = {}
    # Starting vLLM before another backend tokenizes or initializes CUDA keeps
    # its EngineCore spawn path isolated and reproducible.
    execution_backends = sorted(args.backends, key=lambda backend: backend != "vllm")
    for backend in execution_backends:
        runs: List[Dict[str, Any]] = []
        for repetition in range(args.repetitions):
            print(
                f"[RUN] dataset={dataset} order={order} backend={backend} "
                f"repetition={repetition + 1}"
            )
            try:
                if backend == "vllm":
                    run = await run_vllm(
                        None,
                        args.model_path,
                        args.context_length,
                        args.batch_size,
                        args.gpu_memory_utilization,
                        eos_token_id,
                        request_factory=lambda: prepare_requests(samples, processor),
                    )
                else:
                    if requests is None:
                        requests = prepare_requests(samples, processor)
                    run = await run_sglang(
                        requests,
                        args.model_path,
                        args.context_length,
                        args.batch_size,
                        args.gpu_memory_utilization,
                        dataset_dir / f"{order}_{backend}_{repetition + 1}.jsonl",
                        backend == "cstrie",
                        eos_token_id,
                    )
            except Exception as exc:
                has_errors = True
                run = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                if not args.continue_on_error:
                    runs.append(run)
                    result["experiments"][order][backend] = {
                        "runs": runs,
                        "summary": aggregate_runs(runs),
                    }
                    result_path.write_text(
                        json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    raise
            runs.append(run)
            result["experiments"][order][backend] = {
                "runs": runs,
                "summary": aggregate_runs(runs),
            }
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    _write_report(result, dataset_dir / "report.md")
    return result, has_errors


async def main() -> None:
    args = parse_args()
    if "vllm" in args.backends and not args.prepare_only:
        # Must run before AutoProcessor and prepare_requests create the Rust
        # tokenizer worker pool used by the parent process.
        run_experiment._configure_vllm_native_runtime()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests: Dict[str, Dict[str, Any]] = {}
    for dataset in args.datasets:
        adapter = get_adapter(dataset)
        split = args.splits.get(dataset, adapter.default_split)
        manifest = build_manifest(
            adapter,
            _dataset_root(args, dataset),
            split,
            args.seed,
        )
        dataset_dir = args.output_dir / dataset
        manifest_path = dataset_dir / "manifest.json"
        manifests[dataset] = manifest
        write_manifest(manifest, manifest_path)
        print(
            f"[DATA] {dataset}: {manifest['num_media']} media, "
            f"{manifest['num_samples']} samples -> {manifest_path}"
        )
    if args.prepare_only:
        prepared = {
            dataset: {
                key: value for key, value in manifest.items() if key != "records"
            }
            for dataset, manifest in manifests.items()
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(prepared, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return

    processor = load_processor(args.model_path)
    results: Dict[str, Dict[str, Any]] = {}
    has_errors = False
    for dataset in args.datasets:
        result, dataset_has_errors = await _run_dataset(
            args, manifests[dataset], processor
        )
        results[dataset] = result
        has_errors = has_errors or dataset_has_errors
        _write_cross_dataset_summary(results, args.output_dir)
    print(f"[DONE] {args.output_dir / 'summary.json'}")
    if has_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())

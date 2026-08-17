"""Shared result storage and reporting for CSTrie experiments."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

SCHEMA_VERSION = 1
RESULT_FILENAME = "result.json"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def dataset_key(dataset_names: Iterable[str]) -> str:
    names = sorted({str(name) for name in dataset_names})
    if not names:
        raise ValueError("至少需要一个数据集")
    return "+".join(names)


def build_identity(
    dataset_hashes: Mapping[str, str], parameters: Mapping[str, Any]
) -> Dict[str, Any]:
    datasets = [
        {"name": name, "content_sha256": str(dataset_hashes[name])}
        for name in sorted(dataset_hashes)
    ]
    parameters_sha256 = content_sha256(parameters)
    full_sha256 = content_sha256(
        {"datasets": datasets, "parameters": parameters}
    )
    return {
        "dataset_key": dataset_key(dataset_hashes),
        "datasets": datasets,
        "parameters_sha256": parameters_sha256,
        "full_sha256": full_sha256,
        "run_id": full_sha256[:12],
    }


def result_directory(root: Path, identity: Mapping[str, Any]) -> Path:
    return root / str(identity["dataset_key"]) / str(identity["run_id"])


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    )


def _number(source: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = source.get(key)
        if value is not None and not isinstance(value, bool):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def normalize_cache_metrics(
    source: Mapping[str, Any], *, fallback_total_tokens: Optional[int] = None,
    num_requests: Optional[int] = None,
) -> Dict[str, Any]:
    """Normalize vLLM/SGLang/CSTrie metrics without inventing unavailable data."""
    total = _number(
        source, "total_tokens", "total_input_tokens_measured", "total_prompt_tokens"
    )
    if total is None and fallback_total_tokens is not None:
        total = float(fallback_total_tokens)
    hits = _number(
        source, "hit_tokens", "total_hit_tokens_measured", "cache_hit_tokens", "prefill_hit"
    )
    peak_tokens = _number(source, "peak_cache_tokens", "peak_full_tokens")
    peak_bytes = _number(source, "peak_cache_bytes", "peak_radix_bytes")
    peak_mib = _number(source, "peak_cache_mib", "peak_radix_mib")
    if peak_bytes is None and peak_mib is not None:
        peak_bytes = peak_mib * 1024 * 1024
    if peak_mib is None and peak_bytes is not None:
        peak_mib = peak_bytes / (1024 * 1024)
    micro = _number(source, "micro_hit_rate", "aggregate_hit_rate_micro")
    macro = _number(source, "macro_hit_rate", "aggregate_hit_rate_macro")
    if micro is None:
        percent = _number(source, "aggregate_hit_rate_micro_percent")
        micro = percent / 100.0 if percent is not None else None
    if macro is None:
        percent = _number(source, "aggregate_hit_rate_macro_percent")
        macro = percent / 100.0 if percent is not None else None
    if micro is None and total and hits is not None:
        micro = hits / total
    requests = num_requests
    if requests is None:
        value = _number(source, "num_requests", "total_request_cache_events")
        requests = int(value) if value is not None else None
    return {
        "total_tokens": int(total) if total is not None else None,
        "hit_tokens": int(hits) if hits is not None else None,
        "peak_cache_tokens": int(peak_tokens) if peak_tokens is not None else None,
        "peak_cache_bytes": int(peak_bytes) if peak_bytes is not None else None,
        "peak_cache_mib": peak_mib,
        "micro_hit_rate": micro,
        "macro_hit_rate": macro,
        "num_requests": requests,
    }


def aggregate_standard_runs(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        key = (
            str(run.get("dataset", "unknown")),
            str(run.get("backend", "unknown")),
            str(run.get("cache_policy", "unknown")),
            str(run.get("order", "default")),
        )
        groups[key].append(run)
    rows: List[Dict[str, Any]] = []
    for (dataset, backend, policy, order), items in sorted(groups.items()):
        successful = [item for item in items if item.get("status") == "ok"]
        metrics = [item.get("metrics", {}) for item in successful]
        totals = [m.get("total_tokens") for m in metrics if m.get("total_tokens") is not None]
        hits = [m.get("hit_tokens") for m in metrics if m.get("hit_tokens") is not None]
        total_tokens = sum(totals) if len(totals) == len(metrics) and metrics else None
        hit_tokens = sum(hits) if len(hits) == len(metrics) and metrics else None
        peak_tokens_values = [m.get("peak_cache_tokens") for m in metrics if m.get("peak_cache_tokens") is not None]
        peak_bytes_values = [m.get("peak_cache_bytes") for m in metrics if m.get("peak_cache_bytes") is not None]
        micro = (
            hit_tokens / total_tokens
            if total_tokens not in (None, 0) and hit_tokens is not None
            else None
        )
        macro_parts = [
            (m.get("macro_hit_rate"), m.get("num_requests"))
            for m in metrics
            if m.get("macro_hit_rate") is not None and m.get("num_requests")
        ]
        macro_weight = sum(weight for _, weight in macro_parts)
        macro = (
            sum(rate * weight for rate, weight in macro_parts) / macro_weight
            if macro_weight else None
        )
        peak_bytes = max(peak_bytes_values) if peak_bytes_values else None
        rows.append({
            "dataset": dataset,
            "backend": backend,
            "cache_policy": policy,
            "order": order,
            "status": "ok" if len(successful) == len(items) else ("error" if not successful else "partial"),
            "successful_runs": len(successful),
            "total_runs": len(items),
            "total_tokens": total_tokens,
            "hit_tokens": hit_tokens,
            "peak_cache_tokens": max(peak_tokens_values) if peak_tokens_values else None,
            "peak_cache_bytes": peak_bytes,
            "peak_cache_mib": peak_bytes / (1024 * 1024) if peak_bytes is not None else None,
            "micro_hit_rate": micro,
            "macro_hit_rate": macro,
        })
    return rows


def make_envelope(
    *, experiment_kind: str, script: str, identity: Mapping[str, Any],
    config: Mapping[str, Any], datasets: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]], status: str = "completed",
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": identity["run_id"],
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "experiment": {"kind": experiment_kind, "script": script, "status": status},
        "identity": dict(identity),
        "config": dict(config),
        "datasets": dict(datasets),
        "runs": list(runs),
        "summary": {"rows": aggregate_standard_runs(runs)},
    }


def _fmt_int(value: Any) -> str:
    return "N/A" if value is None else f"{int(value):,}"


def _fmt_rate(value: Any) -> str:
    return "N/A" if value is None else f"{float(value) * 100:.2f}%"


def _fmt_size(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2f} MiB"


def summary_markdown(rows: Sequence[Mapping[str, Any]], title: str) -> str:
    lines = [
        f"# {title}", "",
        "| Dataset | Backend | Policy | Status | Runs | Total Tokens | Hit Tokens | Peak Cache Tokens | Peak Cache Size | Micro | Macro |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('dataset', 'N/A')} | {row.get('backend', 'N/A')} "
            f"| {row.get('cache_policy', 'N/A')} | {row.get('status', 'N/A')} "
            f"| {row.get('successful_runs', 0)}/{row.get('total_runs', 0)} "
            f"| {_fmt_int(row.get('total_tokens'))} | {_fmt_int(row.get('hit_tokens'))} "
            f"| {_fmt_int(row.get('peak_cache_tokens'))} | {_fmt_size(row.get('peak_cache_mib'))} "
            f"| {_fmt_rate(row.get('micro_hit_rate'))} | {_fmt_rate(row.get('macro_hit_rate'))} |"
        )
    return "\n".join(lines) + "\n"


def print_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    print("\n" + "=" * 132)
    print("实验结果汇总")
    print("=" * 132)
    header = (
        f"{'Dataset':<16} {'Backend/Policy':<24} {'Runs':>9} {'Total Tokens':>14} "
        f"{'Hit Tokens':>14} {'Peak Tokens':>14} {'Peak Size':>14} {'Micro':>9} {'Macro':>9}"
    )
    print(header)
    print("-" * 132)
    for row in rows:
        backend = f"{row.get('backend', 'N/A')}/{row.get('cache_policy', 'N/A')}"
        runs = f"{row.get('successful_runs', 0)}/{row.get('total_runs', 0)}"
        print(
            f"{str(row.get('dataset', 'N/A')):<16} {backend:<24} {runs:>9} "
            f"{_fmt_int(row.get('total_tokens')):>14} {_fmt_int(row.get('hit_tokens')):>14} "
            f"{_fmt_int(row.get('peak_cache_tokens')):>14} {_fmt_size(row.get('peak_cache_mib')):>14} "
            f"{_fmt_rate(row.get('micro_hit_rate')):>9} {_fmt_rate(row.get('macro_hit_rate')):>9}"
        )


def write_experiment(envelope: Mapping[str, Any], directory: Path) -> None:
    atomic_write_json(directory / RESULT_FILENAME, envelope)
    rows = envelope.get("summary", {}).get("rows", [])
    atomic_write_text(directory / "report.md", summary_markdown(rows, "Cache Experiment Report"))


def _legacy_text_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    config = payload.get("config", {})
    names = config.get("datasets") or ["unknown"]
    rows: List[Dict[str, Any]] = []
    for field, default_backend, policy in (
        ("baseline", config.get("backend", "sglang"), "native"),
        ("trie", "cstrie", "cstrie"),
    ):
        value = payload.get(field)
        if not isinstance(value, Mapping):
            continue
        backend = str(value.get("backend", default_backend))
        per_dataset = value.get("per_dataset")
        sources = per_dataset if isinstance(per_dataset, Mapping) else {"+".join(names): value}
        for name, item in sources.items():
            metrics_source = item.get("metrics", item) if isinstance(item, Mapping) else {}
            metrics = normalize_cache_metrics(metrics_source, num_requests=item.get("num_requests") if isinstance(item, Mapping) else None)
            rows.append({
                "dataset": name, "backend": backend, "cache_policy": policy,
                "order": "default", "status": "ok", "successful_runs": 1,
                "total_runs": 1, **{key: metrics[key] for key in metrics if key != "num_requests"},
            })
    return rows


def _legacy_multimodal_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    dataset = str(payload.get("manifest", {}).get("dataset", "unknown"))
    rows: List[Dict[str, Any]] = []
    for order, backends in payload.get("experiments", {}).items():
        for backend, value in backends.items():
            runs = []
            for repetition, run in enumerate(value.get("runs", []), 1):
                cache = run.get("kv_cache", {}) if isinstance(run, Mapping) else {}
                runs.append({
                    "dataset": dataset, "backend": backend,
                    "cache_policy": "cstrie" if backend == "cstrie" else "native",
                    "order": order, "repetition": repetition,
                    "status": run.get("status", "error"),
                    "metrics": normalize_cache_metrics(
                        cache,
                        fallback_total_tokens=run.get("total_prompt_tokens"),
                        num_requests=run.get("num_requests"),
                    ),
                })
            rows.extend(aggregate_standard_runs(runs))
    return rows


def rows_from_payload(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if payload.get("schema_version") == SCHEMA_VERSION:
        return list(payload.get("summary", {}).get("rows", []))
    if "experiment_meta" in payload and "config" in payload:
        return _legacy_text_rows(payload)
    if "manifest" in payload and "experiments" in payload:
        return _legacy_multimodal_rows(payload)
    return []


def scan_results(root: Path) -> tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    candidates = set(root.rglob(RESULT_FILENAME))
    candidates.update(
        path for path in root.rglob("results.json") if "artifacts" not in path.parts
    )
    candidates.update(root.rglob("*_CSTrie.json"))
    candidates.update(root.rglob("results_formal.json"))
    for path in sorted(candidates):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            parsed = rows_from_payload(payload)
        except (OSError, ValueError, TypeError) as exc:
            warnings.append(f"{path}: {exc}")
            continue
        if not parsed and payload.get("schema_version") == SCHEMA_VERSION:
            parsed = [{
                "dataset": payload.get("identity", {}).get("dataset_key", "unknown"),
                "backend": "N/A", "cache_policy": "N/A", "order": "N/A",
                "status": payload.get("experiment", {}).get("status", "unknown"),
                "successful_runs": 0, "total_runs": 0,
                "total_tokens": None, "hit_tokens": None,
                "peak_cache_tokens": None, "peak_cache_bytes": None,
                "peak_cache_mib": None, "micro_hit_rate": None,
                "macro_hit_rate": None,
            }]
        for row in parsed:
            enriched = dict(row)
            enriched["result_path"] = str(path)
            enriched["run_id"] = payload.get("run_id", f"legacy:{path.stem}")
            rows.append(enriched)
    return rows, warnings


def write_results_summary(root: Path, output: Optional[Path] = None) -> List[str]:
    rows, warnings = scan_results(root)
    target = output or root / "results_report.md"
    lines = [
        "# CSTrie Experiment Results", "",
        "| Dataset | Run ID | Backend | Policy | Status | Runs | Total Tokens | Hit Tokens | Peak Cache Tokens | Peak Cache Size | Micro | Macro | Result |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        path = Path(row["result_path"])
        try:
            display = str(path.relative_to(root))
        except ValueError:
            display = str(path)
        lines.append(
            f"| {row.get('dataset', 'N/A')} | `{row.get('run_id', 'N/A')}` "
            f"| {row.get('backend', 'N/A')} | {row.get('cache_policy', 'N/A')} "
            f"| {row.get('status', 'N/A')} | {row.get('successful_runs', 0)}/{row.get('total_runs', 0)} "
            f"| {_fmt_int(row.get('total_tokens'))} | {_fmt_int(row.get('hit_tokens'))} "
            f"| {_fmt_int(row.get('peak_cache_tokens'))} | {_fmt_size(row.get('peak_cache_mib'))} "
            f"| {_fmt_rate(row.get('micro_hit_rate'))} | {_fmt_rate(row.get('macro_hit_rate'))} "
            f"| [{display}]({display}) |"
        )
    text = "\n".join(lines) + "\n"
    atomic_write_text(target, text)
    return warnings

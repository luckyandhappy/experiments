"""Persistent SGLang HiCache bundles used by the CSTrie experiments.

The orchestration helpers in this module intentionally do not require SGLang at
import time.  ``BundleHiCacheFile`` is loaded dynamically inside SGLang worker
processes, where the local fork is available.
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from experiment_results import atomic_write_json, content_sha256


BUNDLE_SCHEMA_VERSION = 1
BUNDLE_KIND = "sglang_hicache_decoder_kv"
MANIFEST_NAME = "manifest.json"
PAGE_SUFFIX = ".bin"


def _utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def _small_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(model_path: str | Path) -> Dict[str, Any]:
    """Build a portable, inexpensive model/tokenizer fingerprint.

    Small configuration/tokenizer files are content hashed.  Large weight files
    use their relative path and size so this check does not warm every model
    shard into the OS page cache immediately before an experiment.
    """

    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {root}")
    small_names = {
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
    small_files: List[Dict[str, Any]] = []
    for name in sorted(small_names):
        path = root / name
        if path.is_file():
            small_files.append(
                {
                    "path": name,
                    "size": path.stat().st_size,
                    "sha256": _small_file_sha256(path),
                }
            )
    weight_files: List[Dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in {
            ".safetensors",
            ".bin",
            ".pt",
            ".pth",
        }:
            weight_files.append({"path": path.name, "size": path.stat().st_size})
    payload = {"small_files": small_files, "weight_files": weight_files}
    if not small_files:
        raise ValueError(f"模型目录缺少可用于缓存校验的配置文件: {root}")
    return {**payload, "sha256": hashlib.sha256(_canonical_json(payload)).hexdigest()}


def _sglang_revision() -> Dict[str, Optional[str]]:
    try:
        version = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        version = None
    revision: Optional[str] = None
    try:
        import sglang  # type: ignore

        module_path = Path(sglang.__file__).resolve()
        repo = next((p for p in module_path.parents if (p / ".git").exists()), None)
        if repo is not None:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )
            revision = completed.stdout.strip() or None
    except (ImportError, OSError, StopIteration, subprocess.SubprocessError):
        pass
    return {"version": version, "revision": revision}


def build_compatibility(
    model_path: str | Path,
    *,
    page_size: int,
    dtype: str = "auto",
    attention_backend: str = "triton",
    multimodal: bool = False,
) -> Dict[str, Any]:
    model = model_fingerprint(model_path)
    return {
        "model_sha256": model["sha256"],
        "model": model,
        "sglang": _sglang_revision(),
        "cache_abi": {
            "tp_size": 1,
            "pp_size": 1,
            "page_size": int(page_size),
            "dtype": dtype,
            "attention_backend": attention_backend,
            "multimodal": bool(multimodal),
        },
    }


def stable_served_model_name(compatibility: Mapping[str, Any]) -> str:
    return f"cstrie-cache-{str(compatibility['model_sha256'])[:20]}"


def hicache_engine_kwargs(compatibility: Mapping[str, Any]) -> Dict[str, Any]:
    """Engine arguments needed before a storage backend can be attached."""

    return {
        "enable_hierarchical_cache": True,
        "enable_cache_report": True,
        "served_model_name": stable_served_model_name(compatibility),
    }


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def load_manifest(root: str | Path) -> Dict[str, Any]:
    bundle_root = Path(root).expanduser().resolve()
    path = _manifest_path(bundle_root)
    if not path.is_file():
        raise FileNotFoundError(f"缓存包缺少 {MANIFEST_NAME}: {path.parent}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"缓存包 manifest 无法解析: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"缓存包 manifest 必须是 JSON 对象: {path}")
    if value.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"缓存包 schema 不兼容: {value.get('schema_version')!r}, "
            f"需要 {BUNDLE_SCHEMA_VERSION}"
        )
    if value.get("kind") != BUNDLE_KIND:
        raise ValueError(f"缓存包类型不兼容: {value.get('kind')!r}")
    if not isinstance(value.get("bundle_id"), str) or not value["bundle_id"]:
        raise ValueError(f"缓存包 manifest 缺少 bundle_id: {path}")
    if not isinstance(value.get("compatibility"), dict):
        raise ValueError(f"缓存包 manifest 缺少 compatibility: {path}")
    if not isinstance(value.get("shards"), dict):
        raise ValueError(f"缓存包 manifest 缺少 shards: {path}")
    for relative, expected in value["shards"].items():
        if not isinstance(relative, str) or not isinstance(expected, dict):
            raise ValueError(f"缓存包 shard 清单格式非法: {path}")
        directory = (bundle_root / relative).resolve()
        try:
            directory.relative_to(bundle_root)
        except ValueError as exc:
            raise ValueError(f"缓存包 shard 路径越界: {relative!r}") from exc
        actual = _page_inventory(directory)
        if actual != expected:
            raise ValueError(
                f"缓存包 shard 清单与磁盘内容不一致: {relative}\n"
                f"  manifest={expected}\n  actual={actual}"
            )
    expected_inventory = content_sha256(value["shards"])
    if value.get("inventory_sha256") != expected_inventory:
        raise ValueError(f"缓存包 inventory_sha256 校验失败: {path}")
    expected_bundle_id = content_sha256(
        {
            "compatibility": value["compatibility"],
            "shards": value["shards"],
            "inventory_sha256": expected_inventory,
        }
    )
    if value["bundle_id"] != expected_bundle_id:
        raise ValueError(f"缓存包 bundle_id 校验失败: {path}")
    return value


def validate_compatibility(
    manifest: Mapping[str, Any], compatibility: Mapping[str, Any], *, label: str
) -> None:
    actual = manifest.get("compatibility")
    if actual != dict(compatibility):
        raise ValueError(
            f"{label}缓存包与当前模型/SGLang 缓存 ABI 不兼容\n"
            f"  bundle={content_sha256(actual)}\n"
            f"  current={content_sha256(compatibility)}"
        )


def inspect_import_bundle(
    path: Optional[Path], compatibility: Optional[Mapping[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    manifest = load_manifest(path)
    if compatibility is not None:
        validate_compatibility(manifest, compatibility, label="导入")
    return manifest


def cache_identity_parameters(
    import_manifest: Optional[Mapping[str, Any]], *, export_enabled: bool
) -> Dict[str, Any]:
    return {
        "cache_import_bundle_id": (
            import_manifest.get("bundle_id") if import_manifest else None
        ),
        "cache_import_enabled": import_manifest is not None,
        "cache_export_enabled": bool(export_enabled),
        "cache_kind": BUNDLE_KIND if import_manifest or export_enabled else None,
    }


def shard_relative_path(experiment_kind: str, backend: str, dataset: str) -> Path:
    values = (experiment_kind, backend, dataset)
    if any(not value or value in {".", ".."} or "/" in value for value in values):
        raise ValueError(f"非法缓存分片标识: {values!r}")
    return Path("shards", *values)


def _page_inventory(directory: Path) -> Dict[str, Any]:
    entries = []
    if directory.is_dir():
        for path in sorted(directory.glob(f"*{PAGE_SUFFIX}")):
            if path.is_file():
                entries.append({"name": path.name, "size": path.stat().st_size})
    return {
        "page_count": len(entries),
        "bytes": sum(item["size"] for item in entries),
        "inventory_sha256": content_sha256(entries),
    }


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(
                f"缓存页冲突且大小不同: {source.name} "
                f"({source.stat().st_size} != {destination.stat().st_size})"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass
class CacheHitStats:
    storage_hit_tokens: int = 0
    requests_with_storage_hits: int = 0
    responses_observed: int = 0

    def observe(self, response: Any) -> None:
        self.responses_observed += 1
        if not isinstance(response, Mapping):
            return
        meta = response.get("meta_info")
        if not isinstance(meta, Mapping):
            return
        details = meta.get("cached_tokens_details")
        if not isinstance(details, Mapping):
            return
        try:
            storage = max(0, int(details.get("storage", 0) or 0))
        except (TypeError, ValueError):
            storage = 0
        self.storage_hit_tokens += storage
        self.requests_with_storage_hits += storage > 0

    def merge(self, other: "CacheHitStats") -> None:
        self.storage_hit_tokens += other.storage_hit_tokens
        self.requests_with_storage_hits += other.requests_with_storage_hits
        self.responses_observed += other.responses_observed

    def as_dict(self) -> Dict[str, int]:
        return {
            "storage_hit_tokens": self.storage_hit_tokens,
            "requests_with_storage_hits": self.requests_with_storage_hits,
            "responses_observed": self.responses_observed,
        }


@dataclass
class CacheRun:
    relative_shard: Path
    import_shard: Optional[Path]
    export_stage: Optional[Path]
    read_only: bool
    stats: CacheHitStats = field(default_factory=CacheHitStats)
    attached: bool = False

    def dynamic_backend_config(self) -> str:
        return json.dumps(
            {
                "backend_name": "cstrie_bundle_file",
                "module_path": "cache_storage_backend",
                "class_name": "BundleHiCacheFile",
                "cstrie_import_path": (
                    str(self.import_shard) if self.import_shard is not None else None
                ),
                "cstrie_export_path": (
                    str(self.export_stage) if self.export_stage is not None else None
                ),
                "cstrie_read_only": self.read_only,
                # SGLang defaults to 256 tokens, which silently disables storage
                # reads for short-prompt datasets such as advbench.  One cache
                # page is the smallest useful prefetch for these experiments.
                "prefetch_threshold": 1,
                "hicache_storage_pass_prefix_keys": True,
            },
            ensure_ascii=False,
        )


class CacheBundle:
    def __init__(
        self,
        *,
        import_root: Optional[Path],
        export_root: Optional[Path],
        compatibility: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ) -> None:
        if import_root is None and export_root is None:
            raise ValueError("CacheBundle 至少需要 import_root 或 export_root")
        self.import_root = import_root.expanduser().resolve() if import_root else None
        self.export_root = export_root.expanduser().resolve() if export_root else None
        self.compatibility = dict(compatibility)
        self.provenance = dict(provenance)
        self.import_manifest = (
            inspect_import_bundle(self.import_root, self.compatibility)
            if self.import_root
            else None
        )
        if self.import_manifest:
            imported_hashes = self.import_manifest.get("provenance", {}).get(
                "dataset_hashes", {}
            )
            current_hashes = self.provenance.get("dataset_hashes", {})
            for dataset in sorted(set(imported_hashes) & set(current_hashes)):
                if imported_hashes[dataset] != current_hashes[dataset]:
                    print(
                        f"[CACHE] 数据集 {dataset} 内容指纹与导入包不同；"
                        "继续使用内容寻址缓存，未匹配前缀将正常计算"
                    )
        self.existing_export_manifest: Optional[Dict[str, Any]] = None
        if self.export_root and _manifest_path(self.export_root).exists():
            self.existing_export_manifest = load_manifest(self.export_root)
            validate_compatibility(
                self.existing_export_manifest, self.compatibility, label="导出"
            )
        elif self.export_root and self.export_root.exists() and any(
            self.export_root.iterdir()
        ):
            raise ValueError(
                f"导出目录非空但缺少 {MANIFEST_NAME}，拒绝混入未知缓存页: "
                f"{self.export_root}"
            )
        self._staging_root: Optional[Path] = None
        self._runs: List[CacheRun] = []
        if self.export_root:
            self.export_root.parent.mkdir(parents=True, exist_ok=True)
            self._staging_root = Path(
                tempfile.mkdtemp(
                    prefix=f".{self.export_root.name}.staging-",
                    dir=self.export_root.parent,
                )
            )
            atexit.register(self.abort_staging)

    @property
    def enabled(self) -> bool:
        return True

    def prepare_run(
        self, experiment_kind: str, backend: str, dataset: str, run_label: str
    ) -> Optional[CacheRun]:
        relative = shard_relative_path(experiment_kind, backend, dataset)
        import_shard = self.import_root / relative if self.import_root else None
        if import_shard is not None and not import_shard.is_dir():
            print(f"[CACHE] 导入包缺少分片 {relative}，该运行按冷启动处理")
            import_shard = None
        export_stage = None
        if self._staging_root is not None:
            export_stage = self._staging_root / run_label / relative
            export_stage.mkdir(parents=True, exist_ok=True)
        if import_shard is None and export_stage is None:
            return None
        run = CacheRun(
            relative_shard=relative,
            import_shard=import_shard,
            export_stage=export_stage,
            read_only=export_stage is None,
        )
        self._runs.append(run)
        return run

    async def attach(self, llm: Any, run: Optional[CacheRun]) -> None:
        if run is None:
            return
        manager = getattr(llm, "tokenizer_manager", None)
        attach = getattr(manager, "attach_hicache_storage", None)
        if not callable(attach):
            raise RuntimeError("当前 SGLang 缺少动态 attach_hicache_storage 接口")
        result = await attach(
            hicache_storage_backend="dynamic",
            hicache_storage_backend_extra_config_json=run.dynamic_backend_config(),
            hicache_storage_prefetch_policy="wait_complete",
            hicache_write_policy="write_through",
        )
        if not getattr(result, "success", False):
            raise RuntimeError(
                f"挂载 KV 缓存失败: {getattr(result, 'message', result)!s}"
            )
        run.attached = True

    async def detach(
        self, llm: Any, run: Optional[CacheRun], *, timeout_seconds: float = 60.0
    ) -> None:
        if run is None or not run.attached:
            return
        manager = getattr(llm, "tokenizer_manager", None)
        detach = getattr(manager, "detach_hicache_storage", None)
        if not callable(detach):
            raise RuntimeError("当前 SGLang 缺少动态 detach_hicache_storage 接口")
        deadline = time.monotonic() + timeout_seconds
        last_message = ""
        while True:
            result = await detach()
            if getattr(result, "success", False):
                run.attached = False
                return
            last_message = str(getattr(result, "message", result))
            if time.monotonic() >= deadline:
                raise RuntimeError(f"等待 KV 缓存写入完成超时: {last_message}")
            await asyncio.sleep(0.1)

    def run_summary(self, run: Optional[CacheRun]) -> Dict[str, Any]:
        if run is None:
            return {
                "enabled": False,
                "storage_hit_tokens": 0,
                "requests_with_storage_hits": 0,
                "responses_observed": 0,
            }
        source = _page_inventory(run.import_shard) if run.import_shard else None
        delta = _page_inventory(run.export_stage) if run.export_stage else None
        return {
            "enabled": True,
            "read_only": run.read_only,
            "shard": run.relative_shard.as_posix(),
            "import": source,
            "export_delta": delta,
            **run.stats.as_dict(),
        }

    def finalize(self) -> Optional[Dict[str, Any]]:
        if self.export_root is None:
            return None
        shards = {run.relative_shard for run in self._runs}
        if self.import_root:
            import_shards_root = self.import_root / "shards"
            if import_shards_root.is_dir():
                for directory in import_shards_root.glob("*/*/*"):
                    if directory.is_dir():
                        shards.add(directory.relative_to(self.import_root))
        self.export_root.mkdir(parents=True, exist_ok=True)
        for relative in sorted(shards, key=lambda p: p.as_posix()):
            destination = self.export_root / relative
            destination.mkdir(parents=True, exist_ok=True)
            sources: List[Path] = []
            if self.import_root and (self.import_root / relative).is_dir():
                sources.append(self.import_root / relative)
            if (
                self.existing_export_manifest is not None
                and (self.export_root / relative).is_dir()
            ):
                # Existing destination pages are already present and validated by size.
                pass
            sources.extend(
                run.export_stage
                for run in self._runs
                if run.relative_shard == relative and run.export_stage is not None
            )
            for source_dir in sources:
                for page in sorted(source_dir.glob(f"*{PAGE_SUFFIX}")):
                    if page.is_file():
                        _link_or_copy(page, destination / page.name)
        shard_manifest: Dict[str, Dict[str, Any]] = {}
        for directory in sorted(
            (self.export_root / "shards").glob("*/*/*")
            if (self.export_root / "shards").is_dir()
            else []
        ):
            if directory.is_dir():
                shard_manifest[directory.relative_to(self.export_root).as_posix()] = (
                    _page_inventory(directory)
                )
        inventory_sha256 = content_sha256(shard_manifest)
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "kind": BUNDLE_KIND,
            "created_at": _utc_now(),
            "bundle_id": content_sha256(
                {
                    "compatibility": self.compatibility,
                    "shards": shard_manifest,
                    "inventory_sha256": inventory_sha256,
                }
            ),
            "compatibility": self.compatibility,
            "provenance": self.provenance,
            "visual_encoder_cache_persisted": False,
            "inventory_sha256": inventory_sha256,
            "shards": shard_manifest,
        }
        atomic_write_json(_manifest_path(self.export_root), manifest)
        self.abort_staging()
        return manifest

    def abort_staging(self) -> None:
        if self._staging_root and self._staging_root.exists():
            shutil.rmtree(self._staging_root)
        self._staging_root = None

"""SGLang dynamic storage backend for CSTrie cache bundles."""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
from sglang.srt.mem_cache.hicache_storage import HiCacheFile


PAGE_SUFFIX = ".bin"


class BundleHiCacheFile(HiCacheFile):
    """HiCache file backend with a read-only source and a writable overlay."""

    def __init__(self, storage_config: Any, _factory_kwargs: Any = None):
        extra = dict(getattr(storage_config, "extra_config", None) or {})
        import_value = extra.get("cstrie_import_path")
        export_value = extra.get("cstrie_export_path")
        self._import_path = Path(import_value).resolve() if import_value else None
        self._read_only = bool(extra.get("cstrie_read_only", False))
        self._temporary_primary: Optional[Path] = None
        if export_value:
            primary = Path(export_value).resolve()
        elif self._read_only and self._import_path is not None:
            # Never initialize HiCacheFile's LRU evictor on the imported source:
            # size limits can evict existing files during backend construction.
            self._temporary_primary = Path(
                tempfile.mkdtemp(prefix="cstrie-cache-readonly-")
            ).resolve()
            primary = self._temporary_primary
        else:
            primary = self._import_path
        if primary is None:
            raise ValueError("动态缓存后端缺少 import/export 路径")

        sanitized = {
            key: value
            for key, value in extra.items()
            if not key.startswith("cstrie_")
            and key not in {"backend_name", "module_path", "class_name"}
        }
        config = copy.copy(storage_config)
        config.extra_config = sanitized
        env_name = "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"
        previous = os.environ.pop(env_name, None)
        try:
            super().__init__(config, file_path=str(primary))
        finally:
            if previous is not None:
                os.environ[env_name] = previous
        self._primary_path = Path(self.file_path).resolve()
        self._read_paths = [self._primary_path]
        if self._import_path and self._import_path != self._primary_path:
            self._read_paths.append(self._import_path)
        # A multimodal shard can contain hundreds of thousands of pages.  An
        # os.scandir() for every batch_exists_v2 call makes imports effectively
        # quadratic in the number of pages, so index each immutable/read path
        # once when it is attached.  Writes keep the primary index up to date.
        self._file_indexes = {
            directory: self._index_directory(directory)
            for directory in self._read_paths
        }

    @staticmethod
    def _index_directory(directory: Path) -> set[str]:
        if not directory.is_dir():
            return set()
        with os.scandir(directory) as entries:
            return {
                entry.name
                for entry in entries
                if entry.is_file() and entry.name.endswith(PAGE_SUFFIX)
            }

    def _find_path(self, key: str) -> Optional[Path]:
        filename = f"{self._get_suffixed_key(key)}{PAGE_SUFFIX}"
        return next(
            (
                directory / filename
                for directory in self._read_paths
                if filename in self._file_indexes[directory]
            ),
            None,
        )

    def get(self, key: str, target_location: Any, target_sizes: Any = None):
        tensor_path = self._find_path(key)
        if tensor_path is None:
            return None
        expected = target_location.numel() * target_location.element_size()
        with tensor_path.open("rb", buffering=0) as handle:
            buffer = memoryview(target_location.view(torch.uint8).contiguous().numpy())
            if handle.readinto(buffer) != expected:
                raise IOError(f"Short read for {key}")
        if tensor_path.parent == self._primary_path:
            self._evictor.touch(self._get_suffixed_key(key), str(tensor_path))
        return target_location

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        return [
            self.get(key, target)
            for key, target in zip(keys, target_locations or [None] * len(keys))
        ]

    def set(self, key, value=None, target_location=None, target_sizes=None):
        if self._read_only:
            return True
        written = super().set(key, value, target_location, target_sizes)
        if written:
            filename = f"{self._get_suffixed_key(key)}{PAGE_SUFFIX}"
            self._file_indexes[self._primary_path].add(filename)
        return written

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None):
        if self._read_only:
            return True
        return super().batch_set(keys, values, target_locations, target_sizes)

    def exists(self, key: str) -> bool:
        return self._find_path(key) is not None

    def _collect_existing_component_keys(self, keys, pool_transfers=None):
        target_files = {f"{self._get_component_key(key)}{PAGE_SUFFIX}" for key in keys}
        for transfer in pool_transfers or []:
            for key in keys:
                target_files.add(
                    f"{self._get_component_key(key, transfer.name)}{PAGE_SUFFIX}"
                )
        existing = set()
        for directory in self._read_paths:
            existing.update(target_files & self._file_indexes[directory])
        return existing

    def clear(self):
        if self._read_only:
            return True
        cleared = super().clear()
        if cleared:
            self._file_indexes[self._primary_path].clear()
        return cleared

    def close(self):
        if self._temporary_primary is not None:
            shutil.rmtree(self._temporary_primary, ignore_errors=True)
            self._temporary_primary = None

"""Dataset adapters and shared helpers for multimodal cache experiments."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Type

RequestID = Tuple[str, int]
CacheKey = Hashable
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MME_CATEGORIES = {
    "perception": {
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
    },
    "cognition": {
        "commonsense_reasoning",
        "numerical_calculation",
        "text_translation",
        "code_reasoning",
    },
}


@dataclass(frozen=True)
class MultimodalSample:
    dataset: str
    sample_id: str
    media_id: str
    question: str
    image_path: str
    category: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManifestSample:
    sample: MultimodalSample
    prompt: str
    media_sha256: str


@dataclass(frozen=True)
class PreparedMultimodalRequest:
    request_id: RequestID
    dataset: str
    sample_id: str
    media_id: str
    question: str
    image_path: str
    image_sha256: str
    category: Optional[str]
    prompt: str
    input_ids: Tuple[int, ...]
    cache_keys: Tuple[CacheKey, ...]


class DatasetAdapter(ABC):
    name: str
    default_split: str
    default_num_media: Optional[int] = 200
    stratify_limited_sampling: bool = False

    @abstractmethod
    def load(self, dataset_root: Path, split: str) -> List[MultimodalSample]:
        """Load the official dataset layout into the shared sample schema."""

    def format_prompt(self, sample: MultimodalSample) -> str:
        return sample.question


_ADAPTERS: Dict[str, Type[DatasetAdapter]] = {}


def register_adapter(adapter_cls: Type[DatasetAdapter]) -> Type[DatasetAdapter]:
    name = adapter_cls.name
    if not name or name in _ADAPTERS:
        raise ValueError(f"非法或重复的数据集适配器: {name!r}")
    _ADAPTERS[name] = adapter_cls
    return adapter_cls


def get_adapter(name: str) -> DatasetAdapter:
    try:
        return _ADAPTERS[name]()
    except KeyError as exc:
        raise ValueError(f"未知数据集 {name!r}；支持: {sorted(_ADAPTERS)}") from exc


def adapter_names() -> Tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@register_adapter
class VQAv2Adapter(DatasetAdapter):
    name = "vqav2"
    default_split = "val"

    def load(self, dataset_root: Path, split: str) -> List[MultimodalSample]:
        if split != "val":
            raise ValueError("VQAv2 缓存实验当前只支持 val split")
        questions_path = dataset_root / "v2_OpenEnded_mscoco_val2014_questions.json"
        payload = _read_json(questions_path)
        questions = payload.get("questions") if isinstance(payload, dict) else None
        if not isinstance(questions, list):
            raise ValueError(f"VQAv2 文件缺少 questions 数组: {questions_path}")
        samples: List[MultimodalSample] = []
        for item in questions:
            try:
                question_id = str(int(item["question_id"]))
                image_id = int(item["image_id"])
                question = str(item["question"]).strip()
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"非法 VQAv2 条目: {item!r}") from exc
            samples.append(
                MultimodalSample(
                    dataset=self.name,
                    sample_id=question_id,
                    media_id=str(image_id),
                    question=question,
                    image_path=str(
                        dataset_root / "val2014" / f"COCO_val2014_{image_id:012d}.jpg"
                    ),
                    metadata={"question_id": int(question_id), "image_id": image_id},
                )
            )
        return samples


@register_adapter
class ChartQAAdapter(DatasetAdapter):
    name = "chartqa"
    default_split = "test"

    def load(self, dataset_root: Path, split: str) -> List[MultimodalSample]:
        base = dataset_root / "ChartQA Dataset"
        if not base.is_dir():
            base = dataset_root
        split_dir = base / split
        samples: List[MultimodalSample] = []
        for source in ("human", "augmented"):
            annotation_path = split_dir / f"{split}_{source}.json"
            rows = _read_json(annotation_path)
            if not isinstance(rows, list):
                raise ValueError(f"ChartQA 标注必须是数组: {annotation_path}")
            for index, item in enumerate(rows):
                try:
                    image_name = str(item["imgname"]).strip()
                    question = str(item["query"]).strip()
                except (KeyError, TypeError) as exc:
                    raise ValueError(f"非法 ChartQA 条目: {item!r}") from exc
                samples.append(
                    MultimodalSample(
                        dataset=self.name,
                        sample_id=f"{source}:{index}",
                        media_id=image_name,
                        question=question,
                        image_path=str(split_dir / "png" / image_name),
                        category=source,
                        metadata={"source": source, "label": item.get("label")},
                    )
                )
        return samples

    def format_prompt(self, sample: MultimodalSample) -> str:
        return f"Answer the question using the chart.\n{sample.question}"


def _resolve_mme_image(image_dirs: Sequence[Path], stem: str) -> Path:
    for images_dir in image_dirs:
        direct = images_dir / stem
        if direct.is_file():
            return direct
        for extension in IMAGE_EXTENSIONS:
            candidate = images_dir / f"{stem}{extension}"
            if candidate.is_file():
                return candidate
    return image_dirs[0] / f"{stem}.jpg"


def _parse_mme_question_file(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.rsplit("\t", 1)
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise ValueError(f"MME 问答行必须为 question<TAB>answer: {path}:{line_number}")
        pairs.append((parts[0].strip(), parts[1].strip()))
    if not pairs:
        raise ValueError(f"MME 问答文件为空: {path}")
    return pairs


@register_adapter
class MMEAdapter(DatasetAdapter):
    name = "mme"
    default_split = "all"
    default_num_media = None
    stratify_limited_sampling = True

    def load(self, dataset_root: Path, split: str) -> List[MultimodalSample]:
        if split != "all":
            raise ValueError("经典 MME 使用 all split")
        base = dataset_root / "MME_Benchmark_release_version"
        if not base.is_dir():
            base = dataset_root
        expected_categories = {
            (domain, category)
            for domain, categories in MME_CATEGORIES.items()
            for category in categories
        }
        category_layouts: Dict[Tuple[str, str], Tuple[List[Path], List[Path]]] = {}
        missing_categories = set()
        for domain, category in sorted(expected_categories):
            category_dir = base / domain / category
            nested_questions = category_dir / "questions_answers_YN"
            if nested_questions.is_dir():
                question_files = sorted(nested_questions.glob("*.txt"))
                image_dirs = [category_dir / "images", category_dir]
            elif category_dir.is_dir():
                # Some complete MME releases store each question .txt beside
                # its image instead of using questions_answers_YN/images.
                question_files = sorted(category_dir.glob("*.txt"))
                image_dirs = [category_dir, category_dir / "images"]
            else:
                question_files = []
                image_dirs = [category_dir]
            if question_files:
                category_layouts[(domain, category)] = (question_files, image_dirs)
            else:
                missing_categories.add((domain, category))
        if missing_categories:
            formatted = [f"{domain}/{category}" for domain, category in missing_categories]
            raise FileNotFoundError(f"MME 缺少官方子任务目录: {sorted(formatted)}")
        samples: List[MultimodalSample] = []
        for (domain, category), (question_files, image_dirs) in sorted(
            category_layouts.items()
        ):
            for question_file in question_files:
                image_path = _resolve_mme_image(image_dirs, question_file.stem)
                media_id = f"{domain}/{category}/{question_file.stem}"
                for index, (question, answer) in enumerate(
                    _parse_mme_question_file(question_file)
                ):
                    samples.append(
                        MultimodalSample(
                            dataset=self.name,
                            sample_id=f"{media_id}:{index}",
                            media_id=media_id,
                            question=question,
                            image_path=str(image_path),
                            category=f"{domain}/{category}",
                            metadata={"answer": answer, "domain": domain},
                        )
                    )
        return samples


def validate_samples(samples: Sequence[MultimodalSample], dataset: str) -> None:
    if not samples:
        raise ValueError(f"数据集 {dataset} 没有样本")
    seen: set[str] = set()
    missing: List[str] = []
    for sample in samples:
        if sample.dataset != dataset:
            raise ValueError(f"适配器 {dataset} 返回了错误 dataset={sample.dataset!r}")
        if not sample.sample_id or sample.sample_id in seen:
            raise ValueError(f"数据集 {dataset} 存在空或重复 sample_id: {sample.sample_id!r}")
        if not sample.media_id or not sample.question.strip():
            raise ValueError(f"样本 {sample.sample_id} 缺少 media_id 或 question")
        seen.add(sample.sample_id)
        if not Path(sample.image_path).is_file():
            missing.append(sample.image_path)
    if missing:
        preview = "\n".join(missing[:10])
        suffix = f"\n... 另有 {len(missing) - 10} 个" if len(missing) > 10 else ""
        raise FileNotFoundError(f"数据集 {dataset} 图片缺失:\n{preview}{suffix}")


def _select_media(samples: Sequence[MultimodalSample], limit: int, seed: int) -> List[str]:
    media_ids = sorted({sample.media_id for sample in samples})
    if limit > len(media_ids):
        raise ValueError(f"请求抽取 {limit} 个媒体，但数据集只有 {len(media_ids)} 个")
    return random.Random(seed).sample(media_ids, limit)


def _select_media_stratified(
    samples: Sequence[MultimodalSample], limit: int, seed: int
) -> List[str]:
    by_category: Dict[str, List[str]] = {}
    for sample in samples:
        category = sample.category or "uncategorized"
        by_category.setdefault(category, []).append(sample.media_id)
    rng = random.Random(seed)
    queues: Dict[str, List[str]] = {}
    for category, values in by_category.items():
        unique = sorted(set(values))
        rng.shuffle(unique)
        queues[category] = unique
    if limit > sum(len(values) for values in queues.values()):
        raise ValueError("媒体抽样上限超过数据集媒体总数")
    selected: List[str] = []
    categories = sorted(queues)
    while len(selected) < limit:
        progressed = False
        for category in categories:
            if queues[category] and len(selected) < limit:
                selected.append(queues[category].pop())
                progressed = True
        if not progressed:
            break
    return selected


def sample_by_media(
    samples: Sequence[MultimodalSample],
    adapter: DatasetAdapter,
    num_media: Optional[int],
    seed: int,
) -> Tuple[List[MultimodalSample], List[str]]:
    effective_limit = adapter.default_num_media if num_media is None else num_media
    all_media = sorted({sample.media_id for sample in samples})
    if effective_limit is None:
        selected_media = all_media
    else:
        if effective_limit <= 0:
            raise ValueError("num_media 必须大于 0")
        selector = _select_media_stratified if adapter.stratify_limited_sampling else _select_media
        selected_media = selector(samples, effective_limit, seed)
    grouped: Dict[str, List[MultimodalSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.media_id, []).append(sample)
    selected_samples = [
        sample
        for media_id in selected_media
        for sample in sorted(grouped[media_id], key=lambda item: item.sample_id)
    ]
    return selected_samples, selected_media


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    adapter: DatasetAdapter,
    dataset_root: Path,
    split: str,
    num_media: Optional[int],
    seed: int,
) -> Dict[str, Any]:
    loaded = adapter.load(dataset_root, split)
    validate_samples(loaded, adapter.name)
    selected, selected_media = sample_by_media(loaded, adapter, num_media, seed)
    media_hashes = {
        media_id: sha256_file(Path(next(s.image_path for s in selected if s.media_id == media_id)))
        for media_id in selected_media
    }
    records = [
        {
            **asdict(sample),
            "prompt": adapter.format_prompt(sample),
            "media_sha256": media_hashes[sample.media_id],
        }
        for sample in selected
    ]
    canonical_records = [
        {key: value for key, value in record.items() if key != "image_path"}
        for record in records
    ]
    canonical = json.dumps(canonical_records, ensure_ascii=False, sort_keys=True).encode()
    return {
        "format_version": 2,
        "dataset": adapter.name,
        "split": split,
        "dataset_root": str(dataset_root.resolve()),
        "seed": seed,
        "sampling": {
            "requested_num_media": num_media,
            "default_num_media": adapter.default_num_media,
            "stratified": adapter.stratify_limited_sampling and num_media is not None,
        },
        "num_media": len(selected_media),
        "num_samples": len(records),
        "selected_media_ids": selected_media,
        "records_sha256": hashlib.sha256(canonical).hexdigest(),
        "records": records,
    }


def write_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def manifest_samples(manifest: Mapping[str, Any]) -> List[ManifestSample]:
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("manifest 缺少 records 数组")
    result: List[ManifestSample] = []
    for record in records:
        sample_fields = {
            key: record[key]
            for key in (
                "dataset",
                "sample_id",
                "media_id",
                "question",
                "image_path",
                "category",
                "metadata",
            )
        }
        result.append(
            ManifestSample(
                sample=MultimodalSample(**sample_fields),
                prompt=str(record["prompt"]),
                media_sha256=str(record["media_sha256"]),
            )
        )
    return result


def order_samples(
    samples: Sequence[ManifestSample], order: str, seed: int
) -> List[ManifestSample]:
    ordered = list(samples)
    if order == "grouped":
        return ordered
    if order == "shuffled":
        random.Random(seed).shuffle(ordered)
        return ordered
    raise ValueError(f"未知请求顺序: {order}")


def _flatten_input_ids(value: Any) -> List[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(
        value[0], (list, tuple)
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise TypeError("processor 返回的 input_ids 不是序列")
    return [int(token) for token in value]


def image_token_ids(processor: Any) -> set[int]:
    tokenizer = processor.tokenizer
    candidates = ("<|image_pad|>", "<image>", "<|vision_pad|>")
    result: set[int] = set()
    unknown = getattr(tokenizer, "unk_token_id", None)
    for token in candidates:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != unknown:
            result.add(int(token_id))
    configured = getattr(processor, "image_token_id", None)
    if configured is not None:
        result.add(int(configured))
    if not result:
        raise RuntimeError("无法从多模态 processor 确定视觉占位 token ID")
    return result


def prepare_request(
    item: ManifestSample, processor: Any, request_index: int
) -> PreparedMultimodalRequest:
    sample = item.sample
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample.image_path},
                {"type": "text", "text": item.prompt},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processed = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True
    )
    raw_ids = processed["input_ids"] if isinstance(processed, Mapping) else processed.input_ids
    input_ids = _flatten_input_ids(raw_ids)
    visual_ids = image_token_ids(processor)
    if not any(token in visual_ids for token in input_ids):
        raise RuntimeError(f"sample_id={sample.sample_id} 不包含视觉占位 token")
    keys: List[CacheKey] = [
        ("image", token, item.media_sha256) if token in visual_ids else ("text", token)
        for token in input_ids
    ]
    return PreparedMultimodalRequest(
        request_id=(sample.dataset, request_index),
        dataset=sample.dataset,
        sample_id=sample.sample_id,
        media_id=sample.media_id,
        question=sample.question,
        image_path=sample.image_path,
        image_sha256=item.media_sha256,
        category=sample.category,
        prompt=str(prompt),
        input_ids=tuple(input_ids),
        cache_keys=tuple(keys),
    )


def prepare_requests(
    samples: Sequence[ManifestSample], processor: Any
) -> List[PreparedMultimodalRequest]:
    return [prepare_request(item, processor, index) for index, item in enumerate(samples)]


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_latencies(values: Sequence[float]) -> Dict[str, Optional[float]]:
    return {"p50_seconds": percentile(values, 0.50), "p95_seconds": percentile(values, 0.95)}


def aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    successful = [run for run in runs if run.get("status") == "ok"]
    keys = ("elapsed_seconds", "requests_per_second", "prompt_tokens_per_second")
    medians: Dict[str, Optional[float]] = {}
    for key in keys:
        values = [float(run[key]) for run in successful if run.get(key) is not None]
        medians[key] = statistics.median(values) if values else None
    return {"num_runs": len(runs), "num_successful": len(successful), "median": medians}


def encoder_reuse_opportunity(
    requests: Sequence[PreparedMultimodalRequest],
) -> Dict[str, int]:
    unique_images = len({request.image_sha256 for request in requests})
    return {
        "image_requests": len(requests),
        "unique_images": unique_images,
        "potential_hits": max(0, len(requests) - unique_images),
    }

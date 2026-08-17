"""
CSTrie 缓存命中率对比实验
====================================
实验目的: 对比基于 CSTrie 的前缀缓存策略与推理框架原生基线在前缀缓存效率上的差异
实验组:
  1. 原生基线: 可选择 SGLang RadixCache 或 vLLM Automatic Prefix Caching
  2. CSTrie 策略: 使用 CSTrie 预先计算共享前缀, 结合启发式调度算法进行缓存预填充
控制变量:
  - 批处理大小 (BATCH_SIZE)
  - 最大输入 Token 数 (MAX_INPUT_TOKENS)
  - 上下文长度 (CONTEXT_LENGTH)
  - 模型路径
  - 数据集
依赖:
  - xxxtrie.py:   XXXTrieNode 前缀树数据结构与纵向构建算法
  - scheduler.py: schedule_heuristic 启发式调度算法
  - SGLang:       模型执行框架 (自定义前缀缓存截断参数为 custom_cache_prefix_len)
实验严谨性要求:
  1. 同类实验严格确保批大小、最大解码 Token 数等无关变量一致
  2. 实验详细内容、中间结果均记录在结构化 JSON 输出中
  3. 所有配置参数均可通过命令行参数覆盖
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.metadata
import inspect
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from experiment_results import (
    build_identity,
    content_sha256,
    make_envelope,
    normalize_cache_metrics,
    print_summary,
    result_directory,
    write_experiment,
    write_results_summary,
)

# vLLM 模式不能隐式依赖 SGLang、Trie 或调度器，因此这些模块只在
# SGLang/CSTrie 分支中加载。RequestID 是二元组，可安全地在此定义。
RequestID = Tuple[str, int]


# ============================================================
# 配置常量
# ============================================================

# 模型路径（可通过 --model-path 覆盖）
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Qwen3-8B"
)

# 数据集目录
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

# 实验输出目录
_EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = Path(_EXPERIMENT_DIR) / "results"

# 指标日志路径
_BASELINE_METRICS_LOG = os.path.join(_EXPERIMENT_DIR, "baseline_metrics.jsonl")
_TRIE_METRICS_LOG = os.path.join(_EXPERIMENT_DIR, "trie_metrics.jsonl")

# 默认参数
CONTEXT_LENGTH = 4096
MAX_INPUT_TOKENS = 1024
BATCH_SIZE = 8
GPU_MEMORY_UTILIZATION = 0.8
VLLM_MIN_VERSION = (0, 26)
VLLM_MAX_VERSION = (0, 27)
VLLM_BLOCK_SIZE = 16
SGLANG_PAGE_SIZE = 1

# 用于标明跳过 RadixCache 写入的 bootstrap_host 占位值
_FAKE_BOOTSTRAP_HOST = "2.2.2.2"

# 默认数据集列表
DEFAULT_DATASETS = ["advbench", "alpaca", "squad"]

# 默认调度器列表
DEFAULT_SCHEDULER = ["heuristic", "dfs", "bfs"]


# ============================================================
# 数据集加载
# ============================================================

def _load_instruction_format(data_path: str) -> List[str]:
    """加载 advbench / alpaca 格式的数据集: [{"instruction": "..."}, ...]"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集文件不存在: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        item["instruction"].strip()
        for item in data
        if isinstance(item, dict) and item.get("instruction", "").strip()
    ]


def _load_squad_format(data_path: str) -> List[str]:
    """加载 SQuAD 格式的数据集: {"data": [{"paragraphs": [{"context": ..., "qas": [...]}]}]}
    将每个 (context, question) 对拼接为一个 prompt:
      "Context: {context}\nQuestion: {question}\nAnswer:"
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集文件不存在: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts: List[str] = []
    for article in data.get("data", []):
        for para in article.get("paragraphs", []):
            ctx = para.get("context", "").strip()
            for qa in para.get("qas", []):
                question = qa.get("question", "").strip()
                if ctx and question:
                    prompts.append(
                        f"Context: {ctx}\nQuestion: {question}\nAnswer:"
                    )
    return prompts

# 数据集名 → 加载函数映射
_DATASET_LOADERS = {
    "advbench": ("advbench.json", _load_instruction_format),
    "alpaca": ("alpaca.json", _load_instruction_format),
    "squad": ("SQuAD_val.json", _load_squad_format),
}


def load_datasets(
    data_dir: str,
    dataset_names: List[str],
) -> Dict[str, List[str]]:
    """加载一个或多个数据集, 返回 {dataset_name: [prompt, ...]}"""
    result: Dict[str, List[str]] = {}
    for name in dataset_names:
        if name not in _DATASET_LOADERS:
            raise ValueError(
                f"未知数据集: {name}, 支持: {list(_DATASET_LOADERS.keys())}"
            )
        filename, loader = _DATASET_LOADERS[name]
        path = os.path.join(data_dir, filename)
        prompts = loader(path)
        result[name] = prompts
        print(f"[DATA] 加载 {name}: {len(prompts)} 条样本 ({path})")
    return result


# ============================================================
# Token 化
# ============================================================

def tokenize_datasets(
    tokenizer,
    dataset_prompts: Dict[str, List[str]],
    max_input_tokens: int,
) -> Tuple[Dict[str, List[List[int]]], List[Tuple[RequestID, List[int]]]]:
    """对多个数据集的 prompt 进行 token 化。
    Returns:
        request_token_seqs_map:  {dataset_name: [[token_ids, ...], ...]}
        flat_requests:           [(RequestID, token_seq), ...] 扁平列表
    """
    request_token_seqs_map: Dict[str, List[List[int]]] = {}
    flat_requests: List[Tuple[RequestID, List[int]]] = []

    for dataset_name, prompts in dataset_prompts.items():
        seqs: List[List[int]] = []
        for idx, prompt in enumerate(prompts):
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(token_ids) > max_input_tokens:
                token_ids = token_ids[:max_input_tokens]
            if len(token_ids) == 0:
                token_ids = [tokenizer.eos_token_id]
            seqs.append(token_ids)
            flat_requests.append(((dataset_name, idx), token_ids))
        request_token_seqs_map[dataset_name] = seqs

    return request_token_seqs_map, flat_requests


def compute_total_tokens(
    request_token_seqs_map: Dict[str, List[List[int]]],
) -> int:
    """统计所有输入 Token 总数"""
    return sum(
        sum(len(seq) for seq in seqs)
        for seqs in request_token_seqs_map.values()
    )


def analyze_prefix_opportunities(
    seqs: List[List[int]],
    match_unit: int,
    batch_size: int,
) -> Dict[str, Any]:
    """按严格批次时序估算指定匹配粒度下、容量无限时的最大可命中量。"""
    cached_prefixes: Set[Tuple[int, ...]] = set()
    total_tokens = sum(len(seq) for seq in seqs)
    hit_tokens = 0
    hit_requests = 0
    eligible_requests = sum(max(len(seq) - 1, 0) >= match_unit for seq in seqs)

    for start in range(0, len(seqs), batch_size):
        batch = seqs[start : start + batch_size]
        for seq in batch:
            hit = 0
            max_hit = max(len(seq) - 1, 0)
            for end in range(match_unit, max_hit + 1, match_unit):
                if tuple(seq[:end]) not in cached_prefixes:
                    break
                hit = end
            hit_tokens += hit
            hit_requests += hit > 0
        # 同一批并发请求不能使用本批才产生的缓存。
        for seq in batch:
            for end in range(match_unit, len(seq) + 1, match_unit):
                cached_prefixes.add(tuple(seq[:end]))

    return {
        "match_unit_tokens": match_unit,
        "eligible_requests": eligible_requests,
        "requests_with_opportunity": hit_requests,
        "potential_hit_tokens": hit_tokens,
        "potential_micro_hit_rate": hit_tokens / total_tokens if total_tokens else 0.0,
        "potential_micro_hit_rate_percent": (
            hit_tokens / total_tokens * 100.0 if total_tokens else 0.0
        ),
    }


def _dataset_metrics_path(path: str, dataset_name: str) -> str:
    stem, suffix = os.path.splitext(path)
    return f"{stem}.{dataset_name}{suffix or '.jsonl'}"


# ============================================================
# SGLang 请求发送
# ============================================================

def _load_sglang():
    """只在 SGLang 实验分支导入运行时依赖。"""
    try:
        return importlib.import_module("sglang")
    except ImportError as exc:
        raise RuntimeError(
            "选择了 SGLang 后端，但当前环境未安装 sglang"
        ) from exc

async def _send_requests_with_cache_policy(
    llm: Any,
    prefill_batches: List[List[List]],
    normal_batches: List[List[List]],
    rid_to_seq: Dict[RequestID, List[int]],
    tokenizer_eos_id: int,
    batch_size: int,
) -> None:
    """严格按调度器返回的批次结构, 分两阶段执行请求
    Phase A (预填充): 遍历 prefill_batches, 每个批次内的请求并发执行
      每个请求设置 custom_cache_prefix_len = depth, 使 SGLang 只缓存指定深度的前缀
    Phase B (正常执行): 遍历 normal_batches, 每个批次内的请求并发执行
      设置 bootstrap_host 占位值, 跳过 RadixCache 写入, 依赖 Phase A 写入的缓存
    """
    sem = asyncio.Semaphore(batch_size * 2)

    async def _send_prefill(rid: RequestID, seq: List[int], depth: int) -> None:
        async with sem:
            dataset_name, idx = rid
            sgl_rid = f"P:{dataset_name}:{idx:04d}"
            sp = {
                "max_new_tokens": 1,
                "temperature": 0.0,
                "stop_token_ids": [tokenizer_eos_id],
                "skip_special_tokens": True,
                "custom_params": {"custom_cache_prefix_len": depth},
            }
            try:
                await llm.async_generate(
                    input_ids=seq, sampling_params=sp, stream=False, rid=sgl_rid,
                )
            except Exception as exc:
                print(f"[PREFILL] 请求 {sgl_rid} (depth={depth}) 失败: {exc}")

    async def _send_normal(rid: RequestID, seq: List[int]) -> None:
        async with sem:
            dataset_name, idx = rid
            sgl_rid = f"N:{dataset_name}:{idx:04d}"
            sp = {
                "max_new_tokens": 1,
                "temperature": 0.0,
                "stop_token_ids": [tokenizer_eos_id],
                "skip_special_tokens": True,
            }
            try:
                await llm.async_generate(
                    input_ids=seq, sampling_params=sp, stream=False, rid=sgl_rid,
                    bootstrap_host=_FAKE_BOOTSTRAP_HOST,
                )
            except Exception as exc:
                print(f"[NORMAL] 请求 {sgl_rid} 失败: {exc}")

    # ---- Phase A: 预填充 ----
    total_prefill = sum(len(b) for b in prefill_batches)
    if total_prefill > 0:
        print(f"  [PHASE A] 预填充: {len(prefill_batches)} 批, 共 {total_prefill} 请求")
        for bi, batch in enumerate(prefill_batches):
            if not batch:
                continue
            tasks = []
            for item in batch:
                rid, depth = item[0], item[1]
                seq = rid_to_seq.get(rid)
                if seq is not None:
                    tasks.append(_send_prefill(rid, seq, depth))
                else:
                    print(f"  [PREFILL] 警告: 请求 {rid} 无对应 token 序列, 跳过")
            if tasks:
                await asyncio.gather(*tasks)
            print(f"    Batch {bi}: {len(tasks)} 请求完成")
        print(f"  [PHASE A] 预填充全部完成")

    # ---- Phase B: 正常执行 ----
    total_normal = sum(len(b) for b in normal_batches)
    if total_normal > 0:
        print(f"  [PHASE B] 正常执行: {len(normal_batches)} 批, 共 {total_normal} 请求")
        for bi, batch in enumerate(normal_batches):
            if not batch:
                continue
            tasks = []
            for item in batch:
                # normal_batches 元素为 [RequestID, 0]
                # 兼容 item 可能直接是 RequestID 的情况 (防御性)
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    rid = item[0]
                else:
                    rid = item
                seq = rid_to_seq.get(rid)
                if seq is not None:
                    tasks.append(_send_normal(rid, seq))
                else:
                    print(f"  [NORMAL] 警告: 请求 {rid} 无对应 token 序列, 跳过")
            if tasks:
                await asyncio.gather(*tasks)
            print(f"    Batch {bi}: {len(tasks)} 请求完成")
        print(f"  [PHASE B] 正常执行全部完成")


async def _send_scheduled_batches_with_cache_policy(
    llm: Any,
    scheduled_batches: List[List[Any]],
    rid_to_seq: Dict[RequestID, List[int]],
    tokenizer_eos_id: int,
    batch_size: int,
) -> None:
    """按统一时序批次交错执行预填充和普通请求。"""
    sem = asyncio.Semaphore(batch_size * 2)

    async def send(task: ScheduledRequest) -> None:
        seq = rid_to_seq.get(task.request_id)
        if seq is None:
            raise KeyError(f"请求 {task.request_id} 无对应 token 序列")

        async with sem:
            dataset_name, idx = task.request_id
            prefix = "P" if task.kind == "prefill" else "N"
            sgl_rid = f"{prefix}:{dataset_name}:{idx:04d}"
            sp = {
                "max_new_tokens": 1,
                "temperature": 0.0,
                "stop_token_ids": [tokenizer_eos_id],
                "skip_special_tokens": True,
            }
            kwargs = {}
            if task.kind == "prefill":
                sp["custom_params"] = {
                    "custom_cache_prefix_len": task.cache_prefix_len,
                }
            else:
                kwargs["bootstrap_host"] = _FAKE_BOOTSTRAP_HOST

            try:
                await llm.async_generate(
                    input_ids=seq,
                    sampling_params=sp,
                    stream=False,
                    rid=sgl_rid,
                    **kwargs,
                )
            except Exception as exc:
                print(f"[{task.kind.upper()}] 请求 {sgl_rid} 失败: {exc}")
                if task.kind == "prefill":
                    # 生产者失败后不能继续执行依赖它的后续批次。
                    raise

    print(f"  [SCHEDULE] 交错调度: {len(scheduled_batches)} 批")
    for batch_index, batch in enumerate(scheduled_batches):
        if not batch:
            continue
        await asyncio.gather(*(send(task) for task in batch))
        prefill_count = sum(task.kind == "prefill" for task in batch)
        print(
            f"    Batch {batch_index}: {len(batch)} 请求完成 "
            f"(prefill={prefill_count}, normal={len(batch) - prefill_count})"
        )
    print("  [SCHEDULE] 全部时序批次完成")


async def _send_requests_baseline(
    llm: Any,
    flat_seqs: List[Tuple[RequestID, List[int]]],
    tokenizer_eos_id: int,
    batch_size: int,
) -> None:
    """基线模式: 按原始顺序发送所有请求, 不进行任何缓存策略干预"""
    sem = asyncio.Semaphore(batch_size * 2)

    async def _run_one(rid: RequestID, seq: List[int]) -> None:
        async with sem:
            dataset_name, idx = rid
            sgl_rid = f"{dataset_name}:{idx:04d}"
            sp = {
                "max_new_tokens": 1,
                "temperature": 0.0,
                "stop_token_ids": [tokenizer_eos_id],
                "skip_special_tokens": True,
            }
            try:
                await llm.async_generate(
                    input_ids=seq,
                    sampling_params=sp,
                    stream=False,
                    rid=sgl_rid,
                )
            except Exception as exc:
                print(f"[BASELINE] 请求 {sgl_rid} 失败: {exc}")

    # 严格按 batch_size 划分批次。只有上一批全部完成后才提交下一批，
    # 避免后端内部排队深度不同导致缓存可见时序不一致。
    for start in range(0, len(flat_seqs), batch_size):
        batch = flat_seqs[start : start + batch_size]
        await asyncio.gather(*(_run_one(rid, seq) for rid, seq in batch))


async def _run_sglang_cache_self_test(
    llm: Any,
    metrics_log_path: str,
    tokenizer_eos_id: int,
) -> Dict[str, Any]:
    """验证 SGLang 原生逐 token RadixCache，并清除自检数据。"""
    probe = [tokenizer_eos_id] * 4
    sampling_params = {
        "max_new_tokens": 1,
        "temperature": 0.0,
        "stop_token_ids": [tokenizer_eos_id],
        "skip_special_tokens": True,
    }
    for index in range(2):
        await llm.async_generate(
            input_ids=probe,
            sampling_params=sampling_params,
            stream=False,
            rid=f"__sglang_cache_self_test__:{index}",
        )

    observed_hit = 0
    if os.path.exists(metrics_log_path):
        with open(metrics_log_path, "r", encoding="utf-8") as metrics_file:
            for line in metrics_file:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("rid") == "__sglang_cache_self_test__:1":
                    observed_hit = int(
                        event.get("prefix_cache_hit_token_count", 0)
                    )
    if observed_hit < SGLANG_PAGE_SIZE:
        raise RuntimeError(
            "SGLang 原生 RadixCache 自检失败: "
            f"重复请求命中 {observed_hit} tokens"
        )

    flush_result = llm.flush_cache()
    if inspect.isawaitable(flush_result):
        flush_result = await flush_result
    if flush_result is False:
        raise RuntimeError("SGLang RadixCache 自检后清空缓存失败")
    if os.path.exists(metrics_log_path):
        os.remove(metrics_log_path)
    return {
        "passed": True,
        "expected_min_hit_tokens": SGLANG_PAGE_SIZE,
        "observed_hit_tokens": observed_hit,
    }


# ============================================================
# 单次实验执行
# ============================================================

@dataclass
class ExperimentConfig:
    model_path: str
    context_length: int
    batch_size: int
    max_input_tokens: int
    metrics_log_path: str
    scheduler: str
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION


async def run_baseline_experiment(
    config: ExperimentConfig,
    flat_seqs: List[Tuple[RequestID, List[int]]],
) -> Dict[str, Any]:
    """执行 SGLang 基线实验"""
    sgl = _load_sglang()
    if os.path.exists(config.metrics_log_path):
        os.remove(config.metrics_log_path)

    os.environ["SGLANG_CUSTOM_METRICS_LOG"] = config.metrics_log_path
    os.environ["SGLANG_CACHE_EXP_MODE"] = "1"

    print(f"\n{'=' * 60}")
    print("[BASELINE] 启动 SGLang 基线实验")
    print(f"{'=' * 60}")
    print(f"  模型: {config.model_path}")
    print(f"  上下文长度: {config.context_length}")
    print(f"  批大小: {config.batch_size}")
    print(f"  请求总数: {len(flat_seqs)}")

    t0 = time.time()

    print("[BASELINE] 加载 tokenizer ...")
    tokenizer = _load_tokenizer_cls().from_pretrained(
        config.model_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id 不能为 None")

    print("[BASELINE] 启动 SGLang Engine ...")
    llm = sgl.Engine(
        model_path=config.model_path,
        tp_size=1,
        mem_fraction_static=config.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="auto",
        context_length=config.context_length,
        max_running_requests=max(config.batch_size, 4),
        page_size=SGLANG_PAGE_SIZE,
        chunked_prefill_size=256,
        disable_cuda_graph=True,
        disable_radix_cache=False,
        log_level="info",
        # 暂时禁用flashinfer
        attention_backend="triton",
    )

    try:
        print("[BASELINE] 验证 SGLang 原生 token RadixCache ...")
        cache_self_test = await _run_sglang_cache_self_test(
            llm,
            config.metrics_log_path,
            tokenizer.eos_token_id,
        )
        print("[BASELINE] 发送请求 (无缓存策略干预) ...")
        await _send_requests_baseline(
            llm=llm,
            flat_seqs=flat_seqs,
            tokenizer_eos_id=tokenizer.eos_token_id,
            batch_size=config.batch_size,
        )
    finally:
        print("[BASELINE] 关闭 Engine ...")
        llm.shutdown()
        await asyncio.sleep(2)

    elapsed = time.time() - t0
    print(f"[BASELINE] 完成, 耗时 {elapsed:.1f}s")

    return {
        "elapsed_seconds": elapsed,
        "num_requests": len(flat_seqs),
        "cache_self_test": cache_self_test,
    }


def validate_vllm_version(version: Optional[str] = None) -> str:
    """验证 vLLM 主次版本为 0.26，并返回完整版本字符串。"""
    if version is None:
        try:
            version = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "选择了 vLLM 后端，但当前环境未安装 vllm>=0.26.0,<0.27.0"
            ) from exc

    match = re.match(r"^(\d+)\.(\d+)(?:\.|$)", version)
    if match is None:
        raise RuntimeError(f"无法解析 vLLM 版本: {version!r}")
    major_minor = (int(match.group(1)), int(match.group(2)))
    if not (VLLM_MIN_VERSION <= major_minor < VLLM_MAX_VERSION):
        raise RuntimeError(
            f"不支持 vLLM {version}；需要 vllm>=0.26.0,<0.27.0"
        )
    return version


def _configure_vllm_native_runtime() -> None:
    """Configure the reproducible native sampler before importing vLLM."""
    # Fast-tokenizer worker pools created before EngineCore's process can
    # deadlock the child during pinned-memory setup.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # vLLM 0.26 defaults non-MoE models to Model Runner V2. Qwen3-VL can
    # deadlock there immediately after loading weights; V1 is the supported
    # compatibility path and uses the same native block APC implementation.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"


@dataclass(frozen=True)
class _VLLMMetricSnapshot:
    """vLLM 数值指标和 Info 指标中携带的缓存配置。"""

    values: Dict[str, float]
    cache_configs: Tuple[Dict[str, str], ...] = ()


@dataclass(frozen=True)
class _VLLMRequestMetrics:
    """单个已完成 vLLM 请求的缓存统计。"""

    request_id: str
    input_tokens: int
    cached_tokens: int
    cache_creation_tokens: int
    missing_cached_tokens: bool
    missing_cache_creation_tokens: bool


async def _run_vllm_apc_self_test(
    llm: Any,
    sampling_params_cls: Any,
    tokenizer_eos_id: int,
) -> Dict[str, Any]:
    """验证原生 APC 能命中一个完整物理 block，并在测试后清空缓存。"""
    probe = [tokenizer_eos_id] * (VLLM_BLOCK_SIZE + 2)
    first = await _send_requests_vllm(
        llm,
        sampling_params_cls,
        [(('__apc_self_test__', 0), probe)],
        tokenizer_eos_id,
        batch_size=1,
    )
    second = await _send_requests_vllm(
        llm,
        sampling_params_cls,
        [(('__apc_self_test__', 1), probe)],
        tokenizer_eos_id,
        batch_size=1,
    )
    hit_tokens = second[0].cached_tokens
    passed = hit_tokens >= VLLM_BLOCK_SIZE

    reset = getattr(llm, "reset_prefix_cache", None)
    if reset is None:
        raise RuntimeError("vLLM AsyncLLMEngine 缺少 reset_prefix_cache，无法隔离自检缓存")
    reset_result = reset()
    if inspect.isawaitable(reset_result):
        reset_result = await reset_result
    if reset_result is False:
        raise RuntimeError("vLLM prefix cache 自检后清空缓存失败")
    if not passed:
        raise RuntimeError(
            "vLLM 原生 APC 自检失败: "
            f"重复请求仅命中 {hit_tokens} tokens，期望至少 {VLLM_BLOCK_SIZE}"
        )
    return {
        "passed": True,
        "expected_min_hit_tokens": VLLM_BLOCK_SIZE,
        "observed_hit_tokens": hit_tokens,
        "first_request_cached_tokens": first[0].cached_tokens,
    }


def _read_vllm_metric_snapshot() -> _VLLMMetricSnapshot:
    """读取 vLLM 进程内 Prometheus 指标，保留 cache_config_info 标签。"""
    try:
        reader = importlib.import_module("vllm.v1.metrics.reader")
        metrics: Iterable[Any] = reader.get_metrics_snapshot()
    except (ImportError, AttributeError, RuntimeError):
        return _VLLMMetricSnapshot(values={})

    result: Dict[str, float] = {}
    cache_configs: List[Dict[str, str]] = []
    for metric in metrics:
        name = str(getattr(metric, "name", ""))
        labels = getattr(metric, "labels", {})
        if name.replace(":", "_").endswith("cache_config_info"):
            if isinstance(labels, dict):
                cache_configs.append({str(k): str(v) for k, v in labels.items()})
            continue
        value = getattr(metric, "value", None)
        if not name or not isinstance(value, (int, float)):
            continue
        # Gauge 的 KV 使用率不能跨 worker 求和；其余目标指标都是 Counter。
        if name.endswith("kv_cache_usage_perc"):
            result[name] = max(result.get(name, 0.0), float(value))
        else:
            result[name] = result.get(name, 0.0) + float(value)
    return _VLLMMetricSnapshot(
        values=result,
        cache_configs=tuple(cache_configs),
    )


def _metric_value(snapshot: _VLLMMetricSnapshot, suffix: str) -> float:
    """兼容 Prometheus 冒号名和 Python reader 的下划线名。"""
    return sum(
        value
        for name, value in snapshot.values.items()
        if name.replace(":", "_").endswith(suffix.replace(":", "_"))
    )


def _metric_value_any(
    snapshot: _VLLMMetricSnapshot,
    suffixes: Tuple[str, ...],
) -> float:
    """读取在不同 Prometheus client 版本中可能带 `_total` 的 Counter。"""
    for suffix in suffixes:
        matching_names = {
            name: value
            for name, value in snapshot.values.items()
            if name.replace(":", "_").endswith(suffix.replace(":", "_"))
        }
        if matching_names:
            return sum(matching_names.values())
    return 0.0


def _vllm_cache_config_int(
    snapshots: Tuple[_VLLMMetricSnapshot, ...],
    key: str,
) -> Optional[int]:
    """从 cache_config_info 标签读取正整数配置；重复 worker 取最大值。"""
    values: List[int] = []
    for snapshot in snapshots:
        for config in snapshot.cache_configs:
            raw_value = config.get(key)
            if raw_value is None or raw_value.strip().lower() in {"", "none"}:
                continue
            try:
                value = int(raw_value)
            except ValueError:
                continue
            if value > 0:
                values.append(value)
    return max(values) if values else None


def _load_tokenizer_cls():
    try:
        transformers = importlib.import_module("transformers")
        return transformers.AutoTokenizer
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "运行实验需要安装 transformers，并提供 AutoTokenizer"
        ) from exc


async def _send_requests_vllm(
    llm: Any,
    sampling_params_cls: Any,
    flat_seqs: List[Tuple[RequestID, List[int]]],
    tokenizer_eos_id: int,
    batch_size: int,
) -> List[_VLLMRequestMetrics]:
    """使用 vLLM 异步引擎提交原生 APC 基线请求。"""
    sem = asyncio.Semaphore(batch_size * 2)
    sampling_params = sampling_params_cls(
        max_tokens=1,
        temperature=0.0,
        stop_token_ids=[tokenizer_eos_id],
        skip_special_tokens=True,
    )

    async def run_one(
        rid: RequestID,
        seq: List[int],
    ) -> _VLLMRequestMetrics:
        async with sem:
            dataset_name, idx = rid
            request_id = f"{dataset_name}:{idx:04d}"
            # 使用 token prompt，避免 vLLM 再次分词导致两个基线输入不一致。
            prompt = {"prompt_token_ids": seq}
            final_output: Any = None
            async for output in llm.generate(
                prompt,
                sampling_params,
                request_id=request_id,
            ):
                if bool(getattr(output, "finished", False)):
                    final_output = output

            if final_output is None:
                raise RuntimeError(f"vLLM 请求 {request_id} 未返回 finished 输出")

            cached_value = getattr(final_output, "num_cached_tokens", None)
            creation_value = getattr(
                final_output,
                "num_cache_creation_tokens",
                None,
            )
            return _VLLMRequestMetrics(
                request_id=request_id,
                input_tokens=len(seq),
                cached_tokens=max(0, int(cached_value or 0)),
                cache_creation_tokens=max(0, int(creation_value or 0)),
                missing_cached_tokens=cached_value is None,
                missing_cache_creation_tokens=creation_value is None,
            )

    results: List[_VLLMRequestMetrics] = []
    for start in range(0, len(flat_seqs), batch_size):
        batch = flat_seqs[start : start + batch_size]
        results.extend(
            await asyncio.gather(*(run_one(rid, seq) for rid, seq in batch))
        )
    return results


async def run_vllm_baseline_experiment(
    config: ExperimentConfig,
    flat_seqs: List[Tuple[RequestID, List[int]]],
) -> Dict[str, Any]:
    """执行 vLLM 0.26.x 原生 Automatic Prefix Caching 基线。"""
    # FlashInfer 0.6.14 的 sampling JIT 与其打包的 CCCL/CUB 3.x 不兼容，
    # 会在 EngineCore warmup 阶段因 FlagHeads API 缺失而编译失败。缓存实验只需
    # 确定性 greedy sampling，因此固定使用 vLLM 官方的原生 sampler 路径。
    # 必须在导入 vLLM 和创建 EngineCore 子进程前设置，子进程才能继承该配置。
    _configure_vllm_native_runtime()
    version = validate_vllm_version()
    try:
        vllm = importlib.import_module("vllm")
        async_engine_args_cls = vllm.AsyncEngineArgs
        async_engine_cls = vllm.AsyncLLMEngine
        sampling_params_cls = vllm.SamplingParams
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "vLLM 0.26.x 缺少 AsyncLLMEngine/AsyncEngineArgs/SamplingParams API"
        ) from exc

    print(f"\n{'=' * 60}")
    print(f"[BASELINE] 启动 vLLM {version} 原生 APC 基线")
    print(f"{'=' * 60}")
    print(f"  模型: {config.model_path}")
    print(f"  上下文长度: {config.context_length}")
    print(f"  批大小: {config.batch_size}")
    print(f"  KV cache physical block size: {VLLM_BLOCK_SIZE} tokens")
    print(f"  Prefix match mode: native block ({VLLM_BLOCK_SIZE} tokens)")
    print("  Sampler backend: vLLM native (FlashInfer sampler disabled)")
    print(f"  请求总数: {len(flat_seqs)}")

    t0 = time.time()
    print("[BASELINE] 加载 tokenizer ...")
    tokenizer = _load_tokenizer_cls().from_pretrained(
        config.model_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id 不能为 None")

    engine_args = async_engine_args_cls(
        model=config.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.context_length,
        max_num_seqs=config.batch_size,
        async_scheduling=False,
        block_size=VLLM_BLOCK_SIZE,
        enable_prefix_caching=True,
        trust_remote_code=True,
        dtype="auto",
        enforce_eager=True,
        disable_log_stats=False,
    )
    print("[BASELINE] 启动 vLLM AsyncLLMEngine ...")
    llm = async_engine_cls.from_engine_args(engine_args)
    print("[BASELINE] 验证 vLLM 原生 block APC ...")
    try:
        cache_self_test = await _run_vllm_apc_self_test(
            llm,
            sampling_params_cls,
            tokenizer.eos_token_id,
        )
    except Exception:
        shutdown_result = llm.shutdown()
        if inspect.isawaitable(shutdown_result):
            await shutdown_result
        raise
    before = _read_vllm_metric_snapshot()
    peak_kv_usage = _metric_value(before, "kv_cache_usage_perc")
    stop_sampling = asyncio.Event()

    async def sample_kv_usage() -> None:
        nonlocal peak_kv_usage
        while not stop_sampling.is_set():
            snapshot = _read_vllm_metric_snapshot()
            peak_kv_usage = max(
                peak_kv_usage,
                _metric_value(snapshot, "kv_cache_usage_perc"),
            )
            try:
                await asyncio.wait_for(stop_sampling.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass

    sampler_task = asyncio.create_task(sample_kv_usage())
    try:
        print("[BASELINE] 发送请求 (vLLM 原生 APC) ...")
        request_metrics = await _send_requests_vllm(
            llm=llm,
            sampling_params_cls=sampling_params_cls,
            flat_seqs=flat_seqs,
            tokenizer_eos_id=tokenizer.eos_token_id,
            batch_size=config.batch_size,
        )
    finally:
        stop_sampling.set()
        await sampler_task
        after = _read_vllm_metric_snapshot()
        peak_kv_usage = max(
            peak_kv_usage,
            _metric_value(after, "kv_cache_usage_perc"),
        )
        print("[BASELINE] 关闭 vLLM Engine ...")
        shutdown_result = llm.shutdown()
        if inspect.isawaitable(shutdown_result):
            await shutdown_result

    elapsed = time.time() - t0
    total_input_tokens = sum(len(seq) for _, seq in flat_seqs)
    prompt_tokens = max(
        0.0,
        _metric_value_any(after, ("prompt_tokens", "prompt_tokens_total"))
        - _metric_value_any(before, ("prompt_tokens", "prompt_tokens_total")),
    )
    cache_queries = max(
        0.0,
        _metric_value(after, "prefix_cache_queries")
        - _metric_value(before, "prefix_cache_queries"),
    )
    cache_hits = max(
        0.0,
        _metric_value(after, "prefix_cache_hits")
        - _metric_value(before, "prefix_cache_hits"),
    )
    native_hit_rate = cache_hits / cache_queries if cache_queries else 0.0
    request_input_tokens = sum(item.input_tokens for item in request_metrics)
    request_hit_tokens = sum(item.cached_tokens for item in request_metrics)
    cache_creation_tokens = sum(
        item.cache_creation_tokens for item in request_metrics
    )
    micro_hit_rate = (
        request_hit_tokens / request_input_tokens if request_input_tokens else 0.0
    )
    per_request_hit_rates = [
        item.cached_tokens / item.input_tokens
        for item in request_metrics
        if item.input_tokens
    ]
    macro_hit_rate = (
        sum(per_request_hit_rates) / len(per_request_hit_rates)
        if per_request_hit_rates
        else 0.0
    )
    missing_cached_tokens = sum(
        item.missing_cached_tokens for item in request_metrics
    )
    missing_cache_creation_tokens = sum(
        item.missing_cache_creation_tokens for item in request_metrics
    )
    if missing_cached_tokens or missing_cache_creation_tokens:
        print(
            "[BASELINE] 警告: vLLM 请求输出缺少缓存统计字段，已按 0 处理 "
            f"(num_cached_tokens={missing_cached_tokens}, "
            f"num_cache_creation_tokens={missing_cache_creation_tokens})"
        )

    cache_capacity_tokens = _vllm_cache_config_int(
        (after, before),
        "kv_cache_size_tokens",
    )
    cache_capacity_bytes = _vllm_cache_config_int(
        (after, before),
        "kv_cache_memory_bytes",
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
    peak_cache_kib = (
        peak_cache_bytes / 1024.0 if peak_cache_bytes is not None else None
    )
    peak_cache_mib = (
        peak_cache_bytes / (1024.0 * 1024.0)
        if peak_cache_bytes is not None
        else None
    )
    measured_input_tokens = request_input_tokens or (
        int(prompt_tokens) if prompt_tokens else total_input_tokens
    )
    metrics = {
        "prefill_hit": None,
        "peak_full_tokens": peak_cache_tokens,
        "peak_radix_bytes": None,
        "peak_radix_kib": None,
        "peak_radix_mib": None,
        "cache_capacity_tokens": cache_capacity_tokens,
        "peak_cache_tokens": peak_cache_tokens,
        "cache_capacity_bytes": cache_capacity_bytes,
        "peak_cache_bytes": peak_cache_bytes,
        "peak_cache_kib": peak_cache_kib,
        "peak_cache_mib": peak_cache_mib,
        "cache_bytes_available": cache_capacity_bytes is not None,
        "cache_creation_tokens": cache_creation_tokens,
        "cache_hit_tokens": request_hit_tokens,
        "aggregate_hit_rate_micro": micro_hit_rate,
        "aggregate_hit_rate_micro_percent": micro_hit_rate * 100.0,
        "aggregate_hit_rate_macro": macro_hit_rate,
        "aggregate_hit_rate_macro_percent": macro_hit_rate * 100.0,
        "total_input_tokens_measured": measured_input_tokens,
        "total_hit_tokens_measured": request_hit_tokens,
        "backend_metrics": {
            "prefix_cache_query_tokens": int(cache_queries),
            "prefix_cache_hit_tokens": int(cache_hits),
            "native_micro_hit_rate": native_hit_rate,
            "native_micro_hit_rate_percent": native_hit_rate * 100.0,
            "peak_kv_cache_usage": peak_kv_usage,
            "peak_kv_cache_usage_percent": peak_kv_usage * 100.0,
            "missing_num_cached_tokens": missing_cached_tokens,
            "missing_num_cache_creation_tokens": (
                missing_cache_creation_tokens
            ),
        },
    }
    print(f"[BASELINE] 完成, 耗时 {elapsed:.1f}s")
    return {
        "backend": "vllm",
        "backend_version": version,
        "cache_policy": "native",
        "cache_match_mode": "block",
        "block_size": VLLM_BLOCK_SIZE,
        "cache_granularity_tokens": VLLM_BLOCK_SIZE,
        "sampler_backend": "native",
        "flashinfer_sampler_enabled": False,
        "model_runner": "v1",
        "async_scheduling": False,
        "tokenizers_parallelism": False,
        "cache_self_test": cache_self_test,
        "elapsed_seconds": elapsed,
        "num_requests": len(flat_seqs),
        "requests_per_second": len(flat_seqs) / elapsed if elapsed else 0.0,
        "input_tokens_per_second": total_input_tokens / elapsed if elapsed else 0.0,
        "metrics": metrics,
    }


def aggregate_vllm_baselines(
    per_dataset: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """汇总逐数据集冷启动的 vLLM 结果，同时保留每次原始结果。"""
    if not per_dataset:
        raise ValueError("至少需要一个 vLLM 数据集结果")
    runs = list(per_dataset.values())
    metrics_list = [run["metrics"] for run in runs]
    num_requests = sum(run["num_requests"] for run in runs)
    elapsed = sum(run["elapsed_seconds"] for run in runs)
    total_input = sum(m["total_input_tokens_measured"] for m in metrics_list)
    total_hits = sum(m["total_hit_tokens_measured"] for m in metrics_list)
    cache_queries = sum(
        m["backend_metrics"]["prefix_cache_query_tokens"] for m in metrics_list
    )
    cache_hits = sum(
        m["backend_metrics"]["prefix_cache_hit_tokens"] for m in metrics_list
    )

    def max_optional(key: str) -> Optional[float]:
        values = [m[key] for m in metrics_list if m.get(key) is not None]
        return max(values) if values else None

    macro = (
        sum(
            m["aggregate_hit_rate_macro"] * run["num_requests"]
            for run, m in zip(runs, metrics_list)
        )
        / num_requests
        if num_requests
        else 0.0
    )
    micro = total_hits / total_input if total_input else 0.0
    native = cache_hits / cache_queries if cache_queries else 0.0
    first = runs[0]
    metrics = {
        **metrics_list[0],
        "peak_full_tokens": max_optional("peak_full_tokens"),
        "cache_capacity_tokens": max_optional("cache_capacity_tokens"),
        "peak_cache_tokens": max_optional("peak_cache_tokens"),
        "cache_capacity_bytes": max_optional("cache_capacity_bytes"),
        "peak_cache_bytes": max_optional("peak_cache_bytes"),
        "peak_cache_kib": max_optional("peak_cache_kib"),
        "peak_cache_mib": max_optional("peak_cache_mib"),
        "cache_bytes_available": all(
            m["cache_bytes_available"] for m in metrics_list
        ),
        "cache_creation_tokens": sum(m["cache_creation_tokens"] for m in metrics_list),
        "cache_hit_tokens": total_hits,
        "aggregate_hit_rate_micro": micro,
        "aggregate_hit_rate_micro_percent": micro * 100.0,
        "aggregate_hit_rate_macro": macro,
        "aggregate_hit_rate_macro_percent": macro * 100.0,
        "total_input_tokens_measured": total_input,
        "total_hit_tokens_measured": total_hits,
        "backend_metrics": {
            **metrics_list[0]["backend_metrics"],
            "prefix_cache_query_tokens": cache_queries,
            "prefix_cache_hit_tokens": cache_hits,
            "native_micro_hit_rate": native,
            "native_micro_hit_rate_percent": native * 100.0,
            "peak_kv_cache_usage": max(
                m["backend_metrics"]["peak_kv_cache_usage"] for m in metrics_list
            ),
            "peak_kv_cache_usage_percent": max(
                m["backend_metrics"]["peak_kv_cache_usage_percent"]
                for m in metrics_list
            ),
            "missing_num_cached_tokens": sum(
                m["backend_metrics"]["missing_num_cached_tokens"]
                for m in metrics_list
            ),
            "missing_num_cache_creation_tokens": sum(
                m["backend_metrics"]["missing_num_cache_creation_tokens"]
                for m in metrics_list
            ),
        },
    }
    return {
        **first,
        "elapsed_seconds": elapsed,
        "num_requests": num_requests,
        "requests_per_second": num_requests / elapsed if elapsed else 0.0,
        "input_tokens_per_second": total_input / elapsed if elapsed else 0.0,
        "metrics": metrics,
        "per_dataset": per_dataset,
    }


async def run_trie_experiment(
    config: ExperimentConfig,
    request_token_seqs_map: Dict[str, List[List[int]]],
    flat_seqs: List[Tuple[RequestID, List[int]]],
    dataset_names: List[str],
) -> Dict[str, Any]:
    """执行 CSTrie 前缀缓存预填充实验"""
    sgl = _load_sglang()
    from xxxtrie import XXXTrieNode
    from scheduler import (
        schedule_bfs,
        schedule_dfs,
        schedule_heuristic,
        simulate_heuristic_prefix,
        simulate_schedule_bfs,
        simulate_schedule_dfs,
    )
    if os.path.exists(config.metrics_log_path):
        os.remove(config.metrics_log_path)

    os.environ["SGLANG_CUSTOM_METRICS_LOG"] = config.metrics_log_path
    os.environ["SGLANG_CACHE_EXP_MODE"] = "1"

    print(f"\n{'=' * 60}")
    print("[TRIE] 启动 CSTrie 实验")
    print(f"{'=' * 60}")
    print(f"  模型: {config.model_path}")
    print(f"  上下文长度: {config.context_length}")
    print(f"  批大小: {config.batch_size}")
    print(f"  数据集: {dataset_names}")
    print(f"  请求总数: {len(flat_seqs)}")

    t0 = time.time()

    # ---- Phase 0: 加载 tokenizer ----
    print("[TRIE] Phase 0: 加载 tokenizer ...")
    tokenizer = _load_tokenizer_cls().from_pretrained(
        config.model_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id 不能为 None")

    # ---- Phase 1: 构建 XXXTrie ----
    print("[TRIE] Phase 1: 纵向构建 XXXTrie ...")
    t_build_start = time.time()
    root = XXXTrieNode.build_vertical(request_token_seqs_map)
    root.print_tree()
    t_build = time.time() - t_build_start
    total_reqs_in_trie = root.total_request_count()
    num_leaves = len(root.collect_leaves())
    print(f"  构建耗时: {t_build:.2f}s")
    print(f"  Trie 请求总数: {total_reqs_in_trie}")
    print(f"  叶子节点数: {num_leaves}")

    # ---- Phase 2: 构建 rid → seq 查找表 ----
    # 构建 rid → seq 快速查找
    rid_to_seq: Dict[RequestID, List[int]] = {rid: seq for rid, seq in flat_seqs}

    # ---- Phase 3: 调度 ----
    scheduled_batches: Optional[List[List[ScheduledRequest]]] = None
    prefill_batches: List[List[List]] = []
    normal_batches: List[List[List]] = []
    if config.scheduler == "bfs":
        print("[TRIE] Phase 2: 执行 BFS 调度 ...")
        t_sched_start = time.time()
        prefill_batches, normal_batches = schedule_bfs(root, config.batch_size)
        t_sched = time.time() - t_sched_start
        # 模拟 BFS 情况下缓存命中情况
        prefill_prefix, execute_prefix = simulate_schedule_bfs(root, config.batch_size, rid_to_seq)
        print(f"预填充阶段 BFS 命中: {prefill_prefix}")
        print(f"后续执行阶段 BFS 命中: {execute_prefix}")
    elif config.scheduler == "dfs":
        print("[TRIE] Phase 2: 执行 DFS 调度 ...")
        t_sched_start = time.time()
        prefill_batches, normal_batches = schedule_dfs(root, config.batch_size)
        t_sched = time.time() - t_sched_start
        # 模拟 DFS 情况下缓存命中情况
        prefill_prefix, execute_prefix = simulate_schedule_dfs(root, config.batch_size, rid_to_seq)
        print(f"预填充阶段 DFS 命中: {prefill_prefix}")
        print(f"后续执行阶段 DFS 命中: {execute_prefix}")
    else:
        print("[TRIE] Phase 2: 执行启发式调度 ...")
        t_sched_start = time.time()
        scheduled_batches = schedule_heuristic(root, config.batch_size)
        t_sched = time.time() - t_sched_start
        # 模拟缓存命中情况
        prefill_prefix, execute_prefix = simulate_heuristic_prefix(
            root,
            config.batch_size,
            rid_to_seq,
            scheduled_batches=scheduled_batches,
        )
        print(f"启发式 - 预填充阶段模拟命中: {prefill_prefix}")
        print(f"启发式 - 后续执行阶段模拟命中: {execute_prefix}")

    print(f"  调度耗时: {t_sched:.2f}s")
    if scheduled_batches is not None:
        all_tasks = [task for batch in scheduled_batches for task in batch]
        total_prefill = sum(task.kind == "prefill" for task in all_tasks)
        total_normal = len(all_tasks) - total_prefill
        prefill_batch_count = sum(
            any(task.kind == "prefill" for task in batch)
            for batch in scheduled_batches
        )
        normal_batch_count = sum(
            any(task.kind == "normal" for task in batch)
            for batch in scheduled_batches
        )
        all_prefill_depths = [
            task.cache_prefix_len
            for task in all_tasks
            if task.kind == "prefill" and task.cache_prefix_len is not None
        ]
        print(f"  统一时序批次数: {len(scheduled_batches)}")
        print(f"  预填充请求: {total_prefill}, 正常请求: {total_normal}")
        for i, batch in enumerate(scheduled_batches):
            prefill_count = sum(task.kind == "prefill" for task in batch)
            print(
                f"    Scheduled Batch {i:02d}: {len(batch)} 请求, "
                f"prefill={prefill_count}, normal={len(batch) - prefill_count}"
            )
    else:
        total_prefill = sum(len(b) for b in prefill_batches)
        total_normal = sum(len(b) for b in normal_batches)
        prefill_batch_count = len(prefill_batches)
        normal_batch_count = len(normal_batches)
        print(f"  预填充批次数: {prefill_batch_count}, 共 {total_prefill} 请求")
        print(f"  正常执行批次数: {normal_batch_count}, 共 {total_normal} 请求")
        for i, batch in enumerate(prefill_batches):
            print(batch)
            depths = [item[1] for item in batch]
            print(f"    Prefill Batch {i:02d}: {len(batch)} 请求, "
                  f"深度范围 [{min(depths)}, {max(depths)}]")
        all_prefill_depths = [item[1] for b in prefill_batches for item in b]

    # 统计预填充深度分布
    if all_prefill_depths:
        print(f"  预填充深度: min={min(all_prefill_depths)}, "
              f"max={max(all_prefill_depths)}, "
              f"avg={sum(all_prefill_depths)/len(all_prefill_depths):.1f}")

    # 验证调度覆盖完整性
    scheduled_rids: Set[RequestID] = set()
    if scheduled_batches is not None:
        scheduled_rids = {
            task.request_id for batch in scheduled_batches for task in batch
        }
    else:
        for batch in prefill_batches:
            for item in batch:
                scheduled_rids.add(item[0])
        for batch in normal_batches:
            for item in batch:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    scheduled_rids.add(item[0])
                else:
                    scheduled_rids.add(item)

    all_rids = {rid for rid, _ in flat_seqs}
    uncovered = all_rids - scheduled_rids
    if uncovered and scheduled_batches is not None:
        raise AssertionError(f"启发式调度器遗漏 {len(uncovered)} 个请求: {sorted(uncovered)}")
    if uncovered:
        print(f"  [WARN] {len(uncovered)} 个请求未被调度器覆盖, "
              f"将追加到 normal_batches 末尾")
        normal_batches.append([[rid, 0] for rid in uncovered])

    assert len(scheduled_rids | uncovered) == len(all_rids), (
        f"调度覆盖不一致: 已调度 {len(scheduled_rids)}, "
        f"未覆盖 {len(uncovered)}, 总计应有 {len(all_rids)}"
    )

    # ---- Phase 4: 启动 Engine 并执行 ----
    print("[TRIE] Phase 4: 启动 SGLang Engine 并执行请求 ...")
    llm = sgl.Engine(
        model_path=config.model_path,
        tp_size=1,  # 张量并行大小 (单张 GPU)
        mem_fraction_static=config.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="auto",
        context_length=config.context_length,
        max_running_requests=max(config.batch_size, 4),
        page_size=SGLANG_PAGE_SIZE,
        chunked_prefill_size=256,
        disable_cuda_graph=True,
        disable_radix_cache=False,
        log_level="info",
        attention_backend="triton",
    )

    try:
        if scheduled_batches is not None:
            await _send_scheduled_batches_with_cache_policy(
                llm=llm,
                scheduled_batches=scheduled_batches,
                rid_to_seq=rid_to_seq,
                tokenizer_eos_id=tokenizer.eos_token_id,
                batch_size=config.batch_size,
            )
        else:
            await _send_requests_with_cache_policy(
                llm=llm,
                prefill_batches=prefill_batches,
                normal_batches=normal_batches,
                rid_to_seq=rid_to_seq,
                tokenizer_eos_id=tokenizer.eos_token_id,
                batch_size=config.batch_size,
            )
        print("[TRIE] 所有请求执行完成")
    finally:
        print("[TRIE] 关闭 Engine ...")
        llm.shutdown()
        await asyncio.sleep(2)

    elapsed = time.time() - t0
    print(f"[TRIE] 完成, 总耗时 {elapsed:.1f}s")

    if scheduled_batches is not None:
        batches_detail = [
            {
                "phase": "mixed",
                "batch_idx": i,
                "size": len(batch),
                "prefill_count": sum(task.kind == "prefill" for task in batch),
                "normal_count": sum(task.kind == "normal" for task in batch),
            }
            for i, batch in enumerate(scheduled_batches)
        ]
        num_scheduled_batches = len(scheduled_batches)
    else:
        batches_detail = [
            {"phase": "prefill", "batch_idx": i, "size": len(batch)}
            for i, batch in enumerate(prefill_batches)
        ] + [
            {"phase": "normal", "batch_idx": i, "size": len(batch)}
            for i, batch in enumerate(normal_batches)
        ]
        num_scheduled_batches = len(prefill_batches) + len(normal_batches)

    return {
        "elapsed_seconds": elapsed,
        "num_requests": len(flat_seqs),
        "trie_stats": {
            "total_requests": total_reqs_in_trie,
            "num_leaves": num_leaves,
            "num_scheduled_batches": num_scheduled_batches,
            # 兼容字段：在交错调度中，同一批可能同时计入两者。
            "num_prefill_batches": prefill_batch_count,
            "num_normal_batches": normal_batch_count,
            "num_prefill_requests": total_prefill,
            "num_normal_requests": total_normal,
            "build_time_seconds": t_build,
            "schedule_time_seconds": t_sched,
            "depth_statistics": {
                "min_prefill_depth": min(all_prefill_depths) if all_prefill_depths else 0,
                "max_prefill_depth": max(all_prefill_depths) if all_prefill_depths else 0,
                "avg_prefill_depth": (
                    sum(all_prefill_depths) / len(all_prefill_depths)
                    if all_prefill_depths else 0.0
                ),
            },
        },
        "batches_detail": batches_detail,
    }


# ============================================================
# 指标解析
# ============================================================

def parse_metrics_log(
    metrics_log_path: str,
    rid_prefix: str = "",
) -> Dict[str, Any]:
    """解析 SGLang 自定义指标日志. 日志格式 (JSONL): 每行一个 JSON 对象, event 类型包括:
      - radix_peak:    RadixCache 峰值快照
      - request_cache: 单个请求的缓存命中信息
      - summary:       汇总信息
      
    返回的指标:
      - peak_full_tokens:              峰值缓存 Token 数
      - peak_radix_bytes / kib / mib:  峰值缓存大小
      - aggregate_hit_rate_micro:      微观命中率 (总命中 / 总输入)
      - aggregate_hit_rate_macro:      宏观命中率 (每请求命中率均值)
      - total_input_tokens_measured:   指标日志统计的输入 Token 总数
      - total_hit_tokens_measured:     指标日志统计的命中 Token 总数
    """
    if not os.path.exists(metrics_log_path):
        return {"error": f"日志文件不存在: {metrics_log_path}"}

    radix_peak_events: List[Dict] = []
    request_cache_events: List[Dict] = []
    summary_events: List[Dict] = []
    total_events = 0

    with open(metrics_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                total_events += 1
                event_type = record.get("event", "")
                if event_type == "radix_peak":
                    radix_peak_events.append(record)
                elif event_type == "request_cache":
                    request_cache_events.append(record)
                elif event_type == "summary":
                    summary_events.append(record)
            except json.JSONDecodeError:
                continue

    # 峰值缓存 Token / 字节
    peak_full_tokens = 0
    peak_radix_bytes = 0.0
    for event in radix_peak_events:
        tokens = int(event.get("full_tokens", 0))
        radix_bytes = float(event.get("total_radix_bytes", 0))
        if radix_bytes > peak_radix_bytes:
            peak_radix_bytes = radix_bytes
            peak_full_tokens = tokens

    if peak_radix_bytes == 0 and summary_events:
        last_summary = summary_events[-1]
        peak_full_tokens = int(last_summary.get("max_full_tokens", 0))
        peak_radix_bytes = float(last_summary.get("max_radix_bytes", 0))

    # 按前缀过滤 request_cache 事件
    if rid_prefix:
        filtered_events = [ e
            for e in request_cache_events
            if str(e.get("rid", "")).startswith(rid_prefix)
        ]
    else:
        filtered_events = request_cache_events

    # 预填充阶段缓存命中结算
    prefill_hit = 0
    for event in filtered_events:
        if event.get("rid").startswith("P:"):
            prefill_hit += event.get("prefix_cache_hit_token_count")

    # 命中率统计
    total_input_tokens = 0
    total_hit_tokens = 0
    hit_rates: List[float] = []
    for event in filtered_events:
        input_count = int(event.get("input_token_count", 0))
        hit_count = int(event.get("prefix_cache_hit_token_count", 0))
        total_input_tokens += input_count
        total_hit_tokens += hit_count
        if input_count > 0:
            hit_rates.append(hit_count / input_count)

    micro_average = (
        total_hit_tokens / total_input_tokens if total_input_tokens > 0 else 0.0
    )
    macro_average = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0

    return {
        "prefill_hit": prefill_hit,
        "peak_full_tokens": peak_full_tokens,
        "peak_radix_bytes": peak_radix_bytes,
        "peak_radix_kib": peak_radix_bytes / 1024.0,
        "peak_radix_mib": peak_radix_bytes / (1024 * 1024),
        "total_request_cache_events": len(filtered_events),
        "aggregate_hit_rate_micro": micro_average,
        "aggregate_hit_rate_micro_percent": micro_average * 100.0,
        "aggregate_hit_rate_macro": macro_average,
        "aggregate_hit_rate_macro_percent": macro_average * 100.0,
        "total_input_tokens_measured": total_input_tokens,
        "total_hit_tokens_measured": total_hit_tokens,
        "all_request_cache_events": len(request_cache_events),
        "summary": summary_events[-1] if summary_events else None,
        "raw_events_count": total_events,
    }


def aggregate_sglang_baselines(
    per_dataset: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """汇总逐数据集冷启动的 SGLang 原生基线。"""
    if not per_dataset:
        raise ValueError("至少需要一个 SGLang 数据集结果")
    runs = list(per_dataset.values())
    metrics_list = [run["metrics"] for run in runs]
    num_requests = sum(run["num_requests"] for run in runs)
    total_input = sum(m["total_input_tokens_measured"] for m in metrics_list)
    total_hits = sum(m["total_hit_tokens_measured"] for m in metrics_list)
    micro = total_hits / total_input if total_input else 0.0
    total_events = sum(m["total_request_cache_events"] for m in metrics_list)
    macro = (
        sum(
            m["aggregate_hit_rate_macro"] * m["total_request_cache_events"]
            for m in metrics_list
        )
        / total_events
        if total_events
        else 0.0
    )
    metrics = {
        **metrics_list[0],
        "prefill_hit": sum(m["prefill_hit"] for m in metrics_list),
        "peak_full_tokens": max(m["peak_full_tokens"] for m in metrics_list),
        "peak_radix_bytes": max(m["peak_radix_bytes"] for m in metrics_list),
        "peak_radix_kib": max(m["peak_radix_kib"] for m in metrics_list),
        "peak_radix_mib": max(m["peak_radix_mib"] for m in metrics_list),
        "total_request_cache_events": sum(
            m["total_request_cache_events"] for m in metrics_list
        ),
        "aggregate_hit_rate_micro": micro,
        "aggregate_hit_rate_micro_percent": micro * 100.0,
        "aggregate_hit_rate_macro": macro,
        "aggregate_hit_rate_macro_percent": macro * 100.0,
        "total_input_tokens_measured": total_input,
        "total_hit_tokens_measured": total_hits,
        "all_request_cache_events": sum(
            m["all_request_cache_events"] for m in metrics_list
        ),
        "raw_events_count": sum(m["raw_events_count"] for m in metrics_list),
        "summary": None,
    }
    first = runs[0]
    return {
        **first,
        "backend": "sglang",
        "cache_policy": "native",
        "cache_match_mode": "token",
        "page_size": SGLANG_PAGE_SIZE,
        "cache_granularity_tokens": SGLANG_PAGE_SIZE,
        "elapsed_seconds": sum(run["elapsed_seconds"] for run in runs),
        "num_requests": num_requests,
        "metrics": metrics,
        "per_dataset": per_dataset,
    }


# ============================================================
# 主流程
# ============================================================

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSTrie 缓存命中率对比实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog="""
示例:
  # 全部实验
  python run_experiment.py

  # 仅基线
  python run_experiment.py --skip-trie

  # 仅 CSTrie
  python run_experiment.py --skip-baseline

  # 自定义参数
  python run_experiment.py --batch-size 16 --max-input-tokens 4096
  python run_experiment.py --datasets advbench alpaca

  # 独立运行 vLLM 0.26.x 原生 APC 基线（不会加载 SGLang/CSTrie）
  python run_experiment.py --backend vllm
        """,
    )
    parser.add_argument(
        "--backend",
        choices=("sglang", "vllm"),
        default="sglang",
        help="实验后端；vllm 模式只运行原生 APC 基线 (默认: sglang)",
    )
    parser.add_argument(
        "--model-path",
        default=_MODEL_PATH,
        help=f"模型路径 (默认: {_MODEL_PATH})",
    )
    parser.add_argument(
        "--data-dir",
        default=_DATA_DIR,
        help=f"数据集目录 (默认: {_DATA_DIR})",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"数据集名称列表 (默认: {DEFAULT_DATASETS})",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=CONTEXT_LENGTH,
        help=f"上下文长度 (默认: {CONTEXT_LENGTH})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"批大小 (默认: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=GPU_MEMORY_UTILIZATION,
        help=f"推理引擎可使用的 GPU 显存比例 (默认: {GPU_MEMORY_UTILIZATION})",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=MAX_INPUT_TOKENS,
        help=f"单请求最大输入 Token 数 (默认: {MAX_INPUT_TOKENS})",
    )
    parser.add_argument(
        "--baseline-metrics",
        default=None,
        help="基线指标日志路径 (默认: 当前实验目录的 artifacts/)",
    )
    parser.add_argument(
        "--trie-metrics",
        default=None,
        help="Trie 指标日志路径 (默认: 当前实验目录的 artifacts/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"实验结果根目录 (默认: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="跳过基线实验",
    )
    parser.add_argument(
        "--skip-trie",
        action="store_true",
        help="跳过 CSTrie 实验",
    )
    parser.add_argument(
        "--scheduler",
        default=DEFAULT_SCHEDULER,
        help=f"调度器列表 (默认: {DEFAULT_SCHEDULER})",
    )
    args = parser.parse_args()
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization 必须在 (0, 1] 范围内")
    if args.backend == "vllm" and args.skip_baseline:
        parser.error("--backend vllm 只运行基线，不能同时指定 --skip-baseline")

    # ============================================================
    # Step 1: 加载数据集并 Token 化
    # ============================================================
    print("=" * 60)
    print("Step 1: 加载数据集并 Token 化")
    print("=" * 60)
    print(f"  数据集: {args.datasets}")
    print(f"  数据目录: {args.data_dir}")

    dataset_prompts = load_datasets(args.data_dir, args.datasets)
    total_prompts = sum(len(v) for v in dataset_prompts.values())
    print(f"  总样本数: {total_prompts}")

    print(f"\n  加载 tokenizer: {args.model_path}")
    tokenizer = _load_tokenizer_cls().from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id 不能为 None")

    request_token_seqs_map, flat_seqs = tokenize_datasets(
        tokenizer, dataset_prompts, args.max_input_tokens
    )
    total_tokens = compute_total_tokens(request_token_seqs_map)
    avg_tokens = total_tokens / len(flat_seqs) if flat_seqs else 0
    print(f"  Token 总数: {total_tokens}")
    print(f"  平均每请求 Token: {avg_tokens:.1f}")
    for name, seqs in request_token_seqs_map.items():
        seq_lens = [len(s) for s in seqs]
        print(
            f"    {name}: {len(seqs)} 请求, "
            f"Token 范围 [{min(seq_lens)}, {max(seq_lens)}], "
            f"均值 {sum(seq_lens)/len(seq_lens):.1f}"
        )

    dataset_hashes = {
        name: content_sha256(seqs) for name, seqs in request_token_seqs_map.items()
    }
    identity_parameters = {
        "experiment_kind": "text",
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "backend": args.backend,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "scheduler": args.scheduler,
        "run_baseline": not args.skip_baseline,
        "run_cstrie": args.backend == "sglang" and not args.skip_trie,
        "vllm_block_size": VLLM_BLOCK_SIZE if args.backend == "vllm" else None,
        "sglang_page_size": SGLANG_PAGE_SIZE if args.backend == "sglang" else None,
    }
    identity = build_identity(dataset_hashes, identity_parameters)
    experiment_dir = result_directory(args.output_dir, identity)
    artifacts_dir = experiment_dir / "artifacts"
    if args.baseline_metrics is None:
        args.baseline_metrics = str(artifacts_dir / "baseline_metrics.jsonl")
    if args.trie_metrics is None:
        args.trie_metrics = str(artifacts_dir / "trie_metrics.jsonl")

    # ============================================================
    # Step 2: 初始化结果容器
    # ============================================================
    results: Dict[str, Any] = {
        "experiment_meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": os.path.basename(__file__),
        },
        "config": {
            "backend": args.backend,
            "model_path": args.model_path,
            "datasets": args.datasets,
            "data_dir": args.data_dir,
            "context_length": args.context_length,
            "batch_size": args.batch_size,
            "max_input_tokens": args.max_input_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "dataset_summary": {
            "num_samples": len(flat_seqs),
            "total_input_tokens": total_tokens,
            "avg_tokens_per_sample": avg_tokens,
            "per_dataset": {
                name: {
                    "num_samples": len(seqs),
                    "total_tokens": sum(len(s) for s in seqs),
                    "prefix_opportunities": {
                        "sglang_native_1_token": analyze_prefix_opportunities(
                            seqs, SGLANG_PAGE_SIZE, args.batch_size
                        ),
                        "vllm_native_16_token": analyze_prefix_opportunities(
                            seqs, VLLM_BLOCK_SIZE, args.batch_size
                        ),
                    },
                }
                for name, seqs in request_token_seqs_map.items()
            },
        },
        "baseline": None,
        "trie": None,
        "comparison": None,
    }
    prepared_envelope = make_envelope(
        experiment_kind="text",
        script=os.path.basename(__file__),
        identity=identity,
        config={**results["config"], **identity_parameters},
        datasets={
            name: {
                **results["dataset_summary"]["per_dataset"][name],
                "content_sha256": dataset_hashes[name],
            }
            for name in args.datasets
        },
        runs=[],
        status="prepared",
    )
    write_experiment(prepared_envelope, experiment_dir)
    write_results_summary(args.output_dir)

    # ============================================================
    # Step 3: 基线实验
    # ============================================================
    if not args.skip_baseline:
        per_dataset_baselines: Dict[str, Dict[str, Any]] = {}
        for dataset_name in args.datasets:
            dataset_flat_seqs = [
                ((dataset_name, idx), seq)
                for idx, seq in enumerate(request_token_seqs_map[dataset_name])
            ]
            dataset_metrics_path = _dataset_metrics_path(
                args.baseline_metrics, dataset_name
            )
            baseline_config = ExperimentConfig(
                model_path=args.model_path,
                context_length=args.context_length,
                batch_size=args.batch_size,
                max_input_tokens=args.max_input_tokens,
                metrics_log_path=dataset_metrics_path,
                scheduler=args.scheduler,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            print(f"\n[BASELINE] 数据集 {dataset_name}: 冷启动独立引擎")
            if args.backend == "vllm":
                per_dataset_baselines[dataset_name] = (
                    await run_vllm_baseline_experiment(
                        config=baseline_config,
                        flat_seqs=dataset_flat_seqs,
                    )
                )
            else:
                baseline_info = await run_baseline_experiment(
                    config=baseline_config,
                    flat_seqs=dataset_flat_seqs,
                )
                baseline_metrics = parse_metrics_log(
                    dataset_metrics_path, rid_prefix=""
                )
                per_dataset_baselines[dataset_name] = {
                    **baseline_info,
                    "backend": "sglang",
                    "cache_policy": "native",
                    "cache_match_mode": "token",
                    "page_size": SGLANG_PAGE_SIZE,
                    "cache_granularity_tokens": SGLANG_PAGE_SIZE,
                    "metrics": baseline_metrics,
                }

        if args.backend == "vllm":
            results["baseline"] = aggregate_vllm_baselines(per_dataset_baselines)
            baseline_metrics = results["baseline"]["metrics"]
            backend_metrics = baseline_metrics["backend_metrics"]
            print("\n[ANALYSIS] vLLM 原生指标")
            print(
                "  Prefix cache 命中: "
                f"{backend_metrics['prefix_cache_hit_tokens']} / "
                f"{backend_metrics['prefix_cache_query_tokens']} tokens"
            )
            print(
                "  Micro 命中率 (完整 Prompt): "
                f"{baseline_metrics['aggregate_hit_rate_micro_percent']:.2f}%"
            )
            print(
                "  Macro 命中率 (完整 Prompt): "
                f"{baseline_metrics['aggregate_hit_rate_macro_percent']:.2f}%"
            )
            print(
                "  Micro 命中率 (vLLM Query):  "
                f"{backend_metrics['native_micro_hit_rate_percent']:.2f}%"
            )
            print(
                "  累计缓存写入 Token:         "
                f"{baseline_metrics['cache_creation_tokens']}"
            )
            print(
                "  峰值缓存 Token / 总容量:    "
                f"{baseline_metrics['peak_cache_tokens']} / "
                f"{baseline_metrics['cache_capacity_tokens']}"
            )
            if baseline_metrics["cache_bytes_available"]:
                print(
                    "  峰值缓存大小 / 总容量:      "
                    f"{baseline_metrics['peak_cache_mib']:.2f} / "
                    f"{baseline_metrics['cache_capacity_bytes'] / (1024 ** 2):.2f} MiB"
                )
            else:
                print("  峰值缓存大小:               unavailable")
            print(
                "  峰值 KV 使用率:             "
                f"{backend_metrics['peak_kv_cache_usage_percent']:.2f}%"
            )
        else:
            results["baseline"] = aggregate_sglang_baselines(
                per_dataset_baselines
            )
            baseline_metrics = results["baseline"]["metrics"]
            print("\n[ANALYSIS] SGLang 逐数据集冷启动汇总")
            print(f"  峰值缓存 Token:  {baseline_metrics['peak_full_tokens']}")
            print(f"  峰值缓存 (MiB):  {baseline_metrics['peak_radix_mib']:.2f}")
            print(f"  Micro 命中率:    {baseline_metrics['aggregate_hit_rate_micro_percent']:.2f}%")
            print(f"  Macro 命中率:    {baseline_metrics['aggregate_hit_rate_macro_percent']:.2f}%")

    # ============================================================
    # Step 4: CSTrie 实验
    # ============================================================
    if args.backend == "sglang" and not args.skip_trie:
        per_dataset_tries: Dict[str, Dict[str, Any]] = {}
        for dataset_name in args.datasets:
            dataset_map = {dataset_name: request_token_seqs_map[dataset_name]}
            dataset_flat_seqs = [
                ((dataset_name, idx), seq)
                for idx, seq in enumerate(request_token_seqs_map[dataset_name])
            ]
            dataset_metrics_path = _dataset_metrics_path(
                args.trie_metrics, dataset_name
            )
            trie_config = ExperimentConfig(
                model_path=args.model_path,
                context_length=args.context_length,
                batch_size=args.batch_size,
                max_input_tokens=args.max_input_tokens,
                metrics_log_path=dataset_metrics_path,
                scheduler=args.scheduler,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            print(f"\n[TRIE] 数据集 {dataset_name}: 冷启动独立引擎")
            trie_info = await run_trie_experiment(
                config=trie_config,
                request_token_seqs_map=dataset_map,
                flat_seqs=dataset_flat_seqs,
                dataset_names=[dataset_name],
            )
            per_dataset_tries[dataset_name] = {
                **trie_info,
                "metrics": parse_metrics_log(dataset_metrics_path, rid_prefix=""),
            }

        aggregate_trie = aggregate_sglang_baselines(per_dataset_tries)
        trie_stats_list = [run["trie_stats"] for run in per_dataset_tries.values()]
        aggregate_trie.update({
            "backend": "cstrie",
            "cache_policy": "cstrie",
            "batches_detail": {
                name: run["batches_detail"] for name, run in per_dataset_tries.items()
            },
            "trie_stats": {
                "total_requests": sum(s["total_requests"] for s in trie_stats_list),
                "num_leaves": sum(s["num_leaves"] for s in trie_stats_list),
                "num_scheduled_batches": sum(
                    s["num_scheduled_batches"] for s in trie_stats_list
                ),
                "num_prefill_batches": sum(
                    s["num_prefill_batches"] for s in trie_stats_list
                ),
                "num_normal_batches": sum(
                    s["num_normal_batches"] for s in trie_stats_list
                ),
                "num_prefill_requests": sum(
                    s["num_prefill_requests"] for s in trie_stats_list
                ),
                "num_normal_requests": sum(
                    s["num_normal_requests"] for s in trie_stats_list
                ),
                "build_time_seconds": sum(
                    s["build_time_seconds"] for s in trie_stats_list
                ),
                "schedule_time_seconds": sum(
                    s["schedule_time_seconds"] for s in trie_stats_list
                ),
            },
        })
        results["trie"] = aggregate_trie
        trie_metrics = aggregate_trie["metrics"]
        ts = aggregate_trie.get("trie_stats", {})
        
        print(f"  Trie 叶子数:     {ts.get('num_leaves', 'N/A')}")
        print(f"  预填充请求数:    {ts.get('num_prefill_requests', 'N/A')}")
        print(f"  预填充阶段缓存命中: {trie_metrics['prefill_hit']}")
        print(f"  峰值缓存 Token:  {trie_metrics['peak_full_tokens']}")
        print(f"  峰值缓存 (MiB):  {trie_metrics['peak_radix_mib']:.2f}")
        print(f"  Micro 命中率:    {trie_metrics['aggregate_hit_rate_micro_percent']:.2f}%")
        print(f"  Macro 命中率:    {trie_metrics['aggregate_hit_rate_macro_percent']:.2f}%")

    # ============================================================
    # Step 4: 对比分析
    # ============================================================
    if args.backend == "sglang" and not args.skip_baseline and not args.skip_trie:
        print("\n" + "=" * 60)
        print("Step 4: 对比分析")
        print("=" * 60)
        bm = results["baseline"]["metrics"]
        tm = results["trie"]["metrics"]

        comparison_rows = [
            ("峰值缓存 Token 数", "peak_full_tokens", "d"),
            ("峰值缓存 (KiB)", "peak_radix_kib", ".2f"),
            ("峰值缓存 (MiB)", "peak_radix_mib", ".2f"),
            ("Micro 命中率 (%)", "aggregate_hit_rate_micro_percent", ".2f"),
            ("Macro 命中率 (%)", "aggregate_hit_rate_macro_percent", ".2f"),
        ]

        # 表头
        print(f"{'指标':<30} {'SGLang 基线':>15} {'CSTrie':>15} {'差异':>15} {'变化率':>15}")
        print("-" * 90)

        comp: Dict[str, Any] = {}
        for label, key, fmt in comparison_rows:
            bv = bm[key]
            tv = tm[key]
            diff = tv - bv
            if isinstance(bv, (int, float)) and bv != 0:
                change_pct = (diff / bv) * 100.0
            else:
                change_pct = float("nan") if bv == 0 else 0.0

            if fmt == "d":
                b_str, t_str, d_str = str(int(bv)), str(int(tv)), str(int(diff))
            else:
                b_str = f"{bv:{fmt}}"
                t_str = f"{tv:{fmt}}"
                d_str = f"{diff:{fmt}}"

            pct_str = f"{change_pct:+.1f}%" if not (
                isinstance(change_pct, float) and (
                    change_pct != change_pct  # NaN check
                )
            ) else "N/A"

            print(f"{label:<30} {b_str:>15} {t_str:>15} {d_str:>15} {pct_str:>15}")

            comp[f"{key}_diff"] = diff
            comp[f"{key}_change_pct"] = (
                change_pct
                if not (isinstance(change_pct, float) and change_pct != change_pct)
                else None
            )

        results["comparison"] = comp

        # 额外对比: 耗时
        b_elapsed = results["baseline"].get("elapsed_seconds", 0)
        t_elapsed = results["trie"].get("elapsed_seconds", 0)
        if b_elapsed and t_elapsed:
            print(f"\n{'执行耗时':<30} {b_elapsed:>15.1f}s {t_elapsed:>15.1f}s "
                  f"{t_elapsed - b_elapsed:>+15.1f}s "
                  f"{((t_elapsed - b_elapsed) / b_elapsed * 100):>+14.1f}%")
            comp["baseline_elapsed_seconds"] = b_elapsed
            comp["trie_elapsed_seconds"] = t_elapsed
            comp["elapsed_diff_seconds"] = t_elapsed - b_elapsed

    # ============================================================
    # 统一保存结果
    # ============================================================
    standard_runs: List[Dict[str, Any]] = []
    for field, default_backend, cache_policy in (
        ("baseline", args.backend, "native"),
        ("trie", "cstrie", "cstrie"),
    ):
        aggregate = results.get(field)
        if not isinstance(aggregate, dict):
            continue
        per_dataset = aggregate.get("per_dataset")
        if not isinstance(per_dataset, dict) or not per_dataset:
            combined_name = "+".join(sorted(args.datasets))
            per_dataset = {combined_name: aggregate}
        for dataset_name, item in per_dataset.items():
            metrics_source = item.get("metrics", {})
            dataset_total = (
                results["dataset_summary"]["per_dataset"].get(dataset_name, {}).get("total_tokens")
                if dataset_name in results["dataset_summary"]["per_dataset"]
                else results["dataset_summary"]["total_input_tokens"]
            )
            request_count = (
                len(request_token_seqs_map[dataset_name])
                if dataset_name in request_token_seqs_map
                else len(flat_seqs)
            )
            standard_runs.append({
                "dataset": dataset_name,
                "backend": item.get("backend", aggregate.get("backend", default_backend)),
                "cache_policy": item.get("cache_policy", cache_policy),
                "order": "default",
                "repetition": 1,
                "status": "ok",
                "metrics": normalize_cache_metrics(
                    metrics_source,
                    fallback_total_tokens=dataset_total,
                    num_requests=item.get("num_requests", request_count),
                ),
                "performance": {"elapsed_seconds": item.get("elapsed_seconds")},
                "details": item,
            })
    envelope = make_envelope(
        experiment_kind="text",
        script=os.path.basename(__file__),
        identity=identity,
        config={**results["config"], **identity_parameters},
        datasets={
            name: {
                **results["dataset_summary"]["per_dataset"][name],
                "content_sha256": dataset_hashes[name],
            }
            for name in args.datasets
        },
        runs=standard_runs,
    )
    envelope["details"] = {"comparison": results.get("comparison")}
    write_experiment(envelope, experiment_dir)
    warnings = write_results_summary(args.output_dir)
    for warning in warnings:
        print(f"[WARN] 跳过无法解析的历史结果: {warning}")
    print_summary(envelope["summary"]["rows"])
    print(f"\n结果已保存至: {experiment_dir / 'result.json'}")
    print(f"汇总已更新: {args.output_dir / 'results_report.md'}")
    print("实验完成。")


if __name__ == "__main__":
    asyncio.run(main())

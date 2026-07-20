"""
CSTrie 缓存命中率对比实验
====================================
实验目的: 对比基于 CSTrie 的前缀缓存策略与 SGLang 原生基线在前缀缓存效率上的差异
实验组:
  1. SGLang 基线: 不进行任何前缀缓存预填充优化, 依赖 SGLang 内置 RadixCache
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
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import sglang as sgl
from transformers import AutoTokenizer

# ---- 已有实现（不可修改） ----
from xxxtrie import XXXTrieNode, RequestID
from scheduler import *


# ============================================================
# 配置常量
# ============================================================

# 模型路径（可通过 --model-path 覆盖）
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Qwen3-8B"
)

# 数据集目录
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 实验输出目录
_EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 指标日志路径
_BASELINE_METRICS_LOG = os.path.join(_EXPERIMENT_DIR, "baseline_metrics.jsonl")
_TRIE_METRICS_LOG = os.path.join(_EXPERIMENT_DIR, "trie_metrics.jsonl")

# 默认参数
CONTEXT_LENGTH = 4096
MAX_INPUT_TOKENS = 1024
BATCH_SIZE = 8

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


# ============================================================
# SGLang 请求发送
# ============================================================

async def _send_requests_with_cache_policy(
    llm: sgl.Engine,
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


async def _send_requests_baseline(
    llm: sgl.Engine,
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

    tasks = [_run_one(rid, seq) for rid, seq in flat_seqs]
    await asyncio.gather(*tasks)


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


async def run_baseline_experiment(
    config: ExperimentConfig,
    flat_seqs: List[Tuple[RequestID, List[int]]],
) -> Dict[str, Any]:
    """执行 SGLang 基线实验"""
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
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id 不能为 None")

    print("[BASELINE] 启动 SGLang Engine ...")
    llm = sgl.Engine(
        model_path=config.model_path,
        tp_size=1,
        mem_fraction_static=0.8,
        trust_remote_code=True,
        dtype="auto",
        context_length=config.context_length,
        max_running_requests=max(config.batch_size, 4),
        chunked_prefill_size=256,
        disable_cuda_graph=True,
        disable_radix_cache=False,
        log_level="info",
        # 暂时禁用flashinfer
        attention_backend="triton",
    )

    try:
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
    }


async def run_trie_experiment(
    config: ExperimentConfig,
    request_token_seqs_map: Dict[str, List[List[int]]],
    flat_seqs: List[Tuple[RequestID, List[int]]],
    dataset_names: List[str],
) -> Dict[str, Any]:
    """执行 CSTrie 前缀缓存预填充实验"""
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
    tokenizer = AutoTokenizer.from_pretrained(
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
        prefill_batches, normal_batches = schedule_heuristic(root, config.batch_size)
        t_sched = time.time() - t_sched_start
        # 模拟缓存命中情况
        prefill_prefix, execute_prefix = simulate_heuristic_prefix(root, config.batch_size, rid_to_seq)
        print(f"启发式 - 预填充阶段模拟命中: {prefill_prefix}")
        print(f"启发式 - 后续执行阶段模拟命中: {execute_prefix}")

    total_prefill = sum(len(b) for b in prefill_batches)
    total_normal = sum(len(b) for b in normal_batches)
    print(f"  调度耗时: {t_sched:.2f}s")
    print(f"  预填充批次数: {len(prefill_batches)}, 共 {total_prefill} 请求")
    print(f"  正常执行批次数: {len(normal_batches)}, 共 {total_normal} 请求")

    # 预填充批次详情
    for i, batch in enumerate(prefill_batches):
        print(batch)
        depths = [item[1] for item in batch]
        print(f"    Prefill Batch {i:02d}: {len(batch)} 请求, "
              f"深度范围 [{min(depths)}, {max(depths)}]")

    # 统计预填充深度分布
    all_prefill_depths = [item[1] for b in prefill_batches for item in b]
    if all_prefill_depths:
        print(f"  预填充深度: min={min(all_prefill_depths)}, "
              f"max={max(all_prefill_depths)}, "
              f"avg={sum(all_prefill_depths)/len(all_prefill_depths):.1f}")

    # 验证调度覆盖完整性
    scheduled_rids: Set[RequestID] = set()
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
        mem_fraction_static=0.8,
        trust_remote_code=True,
        dtype="auto",
        context_length=config.context_length,
        max_running_requests=max(config.batch_size, 4),
        chunked_prefill_size=256,
        disable_cuda_graph=True,
        disable_radix_cache=False,
        log_level="info",
        attention_backend="triton",
    )

    try:
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

    return {
        "elapsed_seconds": elapsed,
        "num_requests": len(flat_seqs),
        "trie_stats": {
            "total_requests": total_reqs_in_trie,
            "num_leaves": num_leaves,
            "num_prefill_batches": len(prefill_batches),
            "num_normal_batches": len(normal_batches),
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
        "batches_detail": [
            {"phase": "prefill", "batch_idx": i, "size": len(b)}
            for i, b in enumerate(prefill_batches)
        ] + [
            {"phase": "normal", "batch_idx": i, "size": len(b)}
            for i, b in enumerate(normal_batches)
        ],
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


# ============================================================
# 主流程
# ============================================================

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSTrie 缓存命中率对比实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        """,
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
        "--max-input-tokens",
        type=int,
        default=MAX_INPUT_TOKENS,
        help=f"单请求最大输入 Token 数 (默认: {MAX_INPUT_TOKENS})",
    )
    parser.add_argument(
        "--baseline-metrics",
        default=_BASELINE_METRICS_LOG,
        help=f"基线指标日志路径 (默认: {_BASELINE_METRICS_LOG})",
    )
    parser.add_argument(
        "--trie-metrics",
        default=_TRIE_METRICS_LOG,
        help=f"Trie 指标日志路径 (默认: {_TRIE_METRICS_LOG})",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(_EXPERIMENT_DIR, "results_formal.json"),
        help="实验结果输出路径",
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
    tokenizer = AutoTokenizer.from_pretrained(
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

    # ============================================================
    # Step 2: 初始化结果容器
    # ============================================================
    results: Dict[str, Any] = {
        "experiment_meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": os.path.basename(__file__),
        },
        "config": {
            "model_path": args.model_path,
            "datasets": args.datasets,
            "data_dir": args.data_dir,
            "context_length": args.context_length,
            "batch_size": args.batch_size,
            "max_input_tokens": args.max_input_tokens,
        },
        "dataset_summary": {
            "num_samples": len(flat_seqs),
            "total_input_tokens": total_tokens,
            "avg_tokens_per_sample": avg_tokens,
            "per_dataset": {
                name: {"num_samples": len(seqs), "total_tokens": sum(len(s) for s in seqs)}
                for name, seqs in request_token_seqs_map.items()
            },
        },
        "baseline": None,
        "trie": None,
        "comparison": None,
    }

    # ============================================================
    # Step 3: 基线实验
    # ============================================================
    if not args.skip_baseline:
        baseline_config = ExperimentConfig(
            model_path=args.model_path,
            context_length=args.context_length,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            metrics_log_path=args.baseline_metrics,
            scheduler=args.scheduler,
        )
        baseline_info = await run_baseline_experiment(
            config=baseline_config,
            flat_seqs=flat_seqs,
        )
        print("\n[ANALYSIS] 解析基线指标日志 ...")
        baseline_metrics = parse_metrics_log(
            args.baseline_metrics, rid_prefix=""
        )
        results["baseline"] = {
            **baseline_info,
            "metrics": baseline_metrics,
        }
        print(f"  峰值缓存 Token:  {baseline_metrics['peak_full_tokens']}")
        print(f"  峰值缓存 (MiB):  {baseline_metrics['peak_radix_mib']:.2f}")
        print(f"  Micro 命中率:    {baseline_metrics['aggregate_hit_rate_micro_percent']:.2f}%")
        print(f"  Macro 命中率:    {baseline_metrics['aggregate_hit_rate_macro_percent']:.2f}%")

    # ============================================================
    # Step 4: CSTrie 实验
    # ============================================================
    if not args.skip_trie:
        trie_config = ExperimentConfig(
            model_path=args.model_path,
            context_length=args.context_length,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            metrics_log_path=args.trie_metrics,
            scheduler=args.scheduler,
        )
        trie_info = await run_trie_experiment(
            config=trie_config,
            request_token_seqs_map=request_token_seqs_map,
            flat_seqs=flat_seqs,
            dataset_names=args.datasets,
        )
        print("\n[ANALYSIS] 解析 Trie 指标日志 ...")
        trie_metrics = parse_metrics_log(args.trie_metrics, rid_prefix="")
        results["trie"] = {
            **trie_info,
            "metrics": trie_metrics,
        }
        ts = trie_info.get("trie_stats", {})
        
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
    if not args.skip_baseline and not args.skip_trie:
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
    # 保存结果
    # ============================================================
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至: {args.output}")
    print("实验完成。")


if __name__ == "__main__":
    asyncio.run(main())
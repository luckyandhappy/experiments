from __future__ import annotations

import copy
from itertools import islice
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Literal, Optional, Set, Tuple
from xxxtrie import XXXTrieNode

RequestID = Tuple[str, int]


@dataclass(frozen=True)
class ScheduledRequest:
    """统一时序调度中的单个任务。"""
    request_id: RequestID
    kind: Literal["prefill", "normal"]
    cache_prefix_len: Optional[int] = None

    def __post_init__(self):
        if self.kind not in ("prefill", "normal"):
            raise ValueError(f"未知调度任务类型: {self.kind}")
        if self.kind == "prefill" and self.cache_prefix_len is None:
            raise ValueError("prefill 任务必须指定 cache_prefix_len")
        if self.kind == "normal" and self.cache_prefix_len is not None:
            raise ValueError("normal 任务不应指定 cache_prefix_len")


@dataclass(frozen=True)
class _SchedulingCandidate:
    task: ScheduledRequest
    node: XXXTrieNode
    path: Tuple[Hashable, ...]
    dfs_order: int
    ready_batch: int = 0


def _ordered_children(node: XXXTrieNode) -> List[XXXTrieNode]:
    """按 token 稳定返回子节点。"""
    return [child for _, child in sorted(node.children.items())]


def _common_ancestor_depth(
    left_path: Tuple[Hashable, ...],
    right_path: Tuple[Hashable, ...],
) -> int:
    """返回两个节点路径的最近公共祖先相对深度。"""
    depth = 0
    for left_token, right_token in zip(left_path, right_path):
        if left_token != right_token:
            break
        depth += 1
    return depth

# ============================================================
# 启发式调度算法及其模拟器
# ============================================================

def schedule_heuristic(
    root: XXXTrieNode,
    batch_size: int,
) -> List[List[ScheduledRequest]]:
    """生成依赖感知的预填充/普通请求交错调度批次。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    parent: Dict[XXXTrieNode, Optional[XXXTrieNode]] = {root: None}
    candidate_paths: Dict[XXXTrieNode, Tuple[Hashable, ...]] = {}
    candidate_order: Dict[XXXTrieNode, int] = {}
    producers: List[_SchedulingCandidate] = []
    locked_requests: Dict[XXXTrieNode, List[RequestID]] = {}
    ready_candidates: List[_SchedulingCandidate] = []

    def node_path(node: XXXTrieNode) -> Tuple[Hashable, ...]:
        tokens: List[Hashable] = []
        while node is not root:
            tokens.append(node.token)
            node = parent[node]
        tokens.reverse()
        return tuple(tokens)

    def make_normal_candidate(
        node: XXXTrieNode,
        rid: RequestID,
        ready_batch: int,
    ) -> _SchedulingCandidate:
        return _SchedulingCandidate(
            task=ScheduledRequest(rid, "normal"),
            node=node,
            path=candidate_paths[node],
            dfs_order=candidate_order[node],
            ready_batch=ready_batch,
        )

    # Explicit DFS avoids Python's recursion limit for long visual prefixes.
    # Only request-bearing nodes need to retain candidate paths.
    stack = [root]
    visit_order = 0
    while stack:
        node = stack.pop()
        request_ids = sorted(node.request_ids)
        if request_ids:
            candidate_paths[node] = node_path(node)
            candidate_order[node] = visit_order

        if node is root:
            ready_candidates.extend(
                make_normal_candidate(node, rid, 0) for rid in request_ids
            )
        elif node.is_leaf() and request_ids:
            producers.append(_SchedulingCandidate(
                task=ScheduledRequest(request_ids[0], "prefill", node.depth),
                node=node,
                path=candidate_paths[node],
                dfs_order=candidate_order[node],
            ))
            if len(request_ids) > 1:
                locked_requests[node] = request_ids[1:]
        elif request_ids:
            locked_requests[node] = request_ids

        children = _ordered_children(node)
        for child in children:
            parent[child] = node
        for child in reversed(children):
            stack.append(child)
        visit_order += 1

    available: List[_SchedulingCandidate] = producers + ready_candidates
    batches: List[List[ScheduledRequest]] = []

    def first_candidate_key(candidate: _SchedulingCandidate):
        return (
            0 if candidate.task.kind == "normal" else 1,
            candidate.ready_batch,
            candidate.dfs_order,
            candidate.task.request_id,
        )

    def adjacent_candidate_key(
        candidate: _SchedulingCandidate,
        previous: _SchedulingCandidate,
    ):
        return (
            _common_ancestor_depth(candidate.path, previous.path),
            0 if candidate.task.kind == "normal" else 1,
            candidate.ready_batch,
            candidate.dfs_order,
            candidate.task.request_id,
        )

    while available:
        selected: List[_SchedulingCandidate] = []
        while available and len(selected) < batch_size:
            if selected:
                index = min(
                    range(len(available)),
                    key=lambda i: adjacent_candidate_key(available[i], selected[-1]),
                )
            else:
                index = min(
                    range(len(available)),
                    key=lambda i: first_candidate_key(available[i]),
                )
            selected.append(available.pop(index))

        batches.append([candidate.task for candidate in selected])

        # 同批并发，只在整批完成后解锁生产者路径上的普通请求。
        next_batch = len(batches)
        for candidate in selected:
            if candidate.task.kind != "prefill":
                continue
            node: Optional[XXXTrieNode] = candidate.node
            while node is not None:
                for rid in locked_requests.pop(node, []):
                    available.append(make_normal_candidate(node, rid, next_batch))
                node = parent[node]

    if locked_requests:
        missing = sum(len(request_ids) for request_ids in locked_requests.values())
        raise ValueError(f"Trie 中有 {missing} 个请求无可用的后代预填充任务")

    return batches

def simulate_heuristic_prefix_old_version(root: XXXTrieNode, batch_size: int, rid_to_seq: Dict[RequestID, List[int]]):
    """
    针对 XXXTrie 的启发式调度算法的缓存命中率模拟 (理想情况, 排除了同一分支同批执行的情况)
    (牢版本, 但我舍不得删掉)
    """
    prefill_prefix = 0
    execute_prefix = 0
    scheduler_queue: deque[tuple[XXXTrieNode, int]] = deque()
    for child in root.children.values():
        scheduler_queue.append([child, 1])

    def split_child(node: XXXTrieNode, start_depth):
        """从根向叶的主动裂解"""
        nonlocal scheduler_queue, execute_prefix
        if len(node.children) >= 2:
            execute_prefix += node.depth * len(node.request_ids)
            is_first = True
            for child in node.children.values():
                if is_first:
                    scheduler_queue.append([child, start_depth])
                    is_first = False
                else:
                    scheduler_queue.append([child, start_depth + 1])
        elif len(node.children) == 1:
            execute_prefix += node.depth * len(node.request_ids)
            split_child(list(node.children.values())[0], start_depth)
        else:
            scheduler_queue.append([node, start_depth])

    def split_branch(node: XXXTrieNode, start_depth: int):
        """DFS 找出分支的第一个叶节点, 并在沿途进行分支裂解"""
        nonlocal scheduler_queue, execute_prefix, prefill_prefix
        if len(node.children) == 0:
            # 叶子节点, 第一个请求的命中长度等于起始深度-1, 其他请求命中长度等于当前深度
            execute_prefix += node.depth * (len(node.request_ids) - 1)
            prefill_prefix += start_depth - 1
        else:
            # 非叶子节点上的每个请求, 命中长度等于当前节点深度
            execute_prefix += node.depth * len(node.request_ids)
            for value in islice(node.children.values(), 1, None):
                scheduler_queue.append([value, node.depth + 1])
            split_branch(next(iter(node.children.values())), start_depth)

    # 同调度第一阶段的执行方式
    while len(scheduler_queue) > 0:
        if len(scheduler_queue) < batch_size:
            # 先尝试主动裂解调度队列中的节点
            before_size = len(scheduler_queue)
            for i in range(before_size):
                if (len(scheduler_queue) < batch_size):
                    split_child(*scheduler_queue.popleft())
                else:
                    break
            # 剩下的节点不足裂解以填充一个完整的批, 清理并退出
            if before_size == len(scheduler_queue):
                while scheduler_queue:
                    split_branch(*scheduler_queue.popleft())
                break
        else:
            # 从每个调度节点中选择一条 "根-叶子" 路径上的请求填充批
            for i in range(batch_size):
                split_branch(*scheduler_queue.popleft())

    return prefill_prefix, execute_prefix

def simulate_heuristic_prefix(
    root: XXXTrieNode,
    batch_size: int,
    rid_to_seq: Dict[RequestID, List[int]],
    scheduled_batches: Optional[List[List[ScheduledRequest]]] = None,
):
    """按实际交错调度时序模拟预填充和普通请求的缓存命中。"""
    cache_root = radixCacheSimulator()
    prefill_prefix = 0
    execute_prefix = 0

    if scheduled_batches is None:
        scheduled_batches = schedule_heuristic(root, batch_size)

    for batch in scheduled_batches:
        # 同批请求不能相互使用本批新生成的缓存。
        for task in batch:
            tokens = rid_to_seq.get(task.request_id)
            if tokens is None:
                raise KeyError(f"请求 {task.request_id} 缺少 token 序列")
            hit = cache_root.accessing_cache(tokens, 0)
            if task.kind == "prefill":
                prefill_prefix += hit
            else:
                execute_prefix += hit

        for task in batch:
            if task.kind != "prefill":
                continue
            cache_root.insert_cache(
                rid_to_seq[task.request_id],
                0,
                task.cache_prefix_len,
            )

    return prefill_prefix, execute_prefix

# ============================================================
# 基础 DFS 调度的方法及模拟器
# ============================================================

def schedule_dfs(root: XXXTrieNode, batch_size: int):
    """
    针对 XXXTrie 的深度优先调度算法, 用于和启发式进行对比实验. 两阶段调度:
    1. 第一阶段: 从每个 "根-叶子" 路径中取出第一条请求执行 (进行缓存预填充)
    2. 第二阶段: 收集所有未在第一阶段执行的请求，依次执行
    Args:
        root: XXXTrie 根节点
        batch_size: 批大小
    Returns:
        tuple[list[list[tuple[RequestID, int]]], list[list[tuple[RequestID, int]]]]
        填充阶段调度批次列表 + 执行阶段调度批次列表，每个批次包含 [RequestID, 前缀深度] 列表
    """
    prefill_batches: List[List[tuple[RequestID, int]]] = []
    normal_batches: List[List[tuple[RequestID, int]]] = []
    wait_requests: List[RequestID] = list(root.request_ids)
    requests = []

    def deep_first_search(node: XXXTrieNode):
        """深度优先搜索"""
        nonlocal prefill_batches, wait_requests, requests
        for child in node.children.values():
            deep_first_search(child)
        if not node.children:
            # 叶子节点, 将第一个请求放入 prefill_batch, 将其他请求放入等待集合
            wait_requests.extend(islice(node.request_ids, 1, None))
            requests.append([next(iter(node.request_ids)), node.depth])
            if len(requests) == batch_size:
                prefill_batches.append(requests)
                requests = []
        else:
            wait_requests.extend(node.request_ids)

    deep_first_search(root)
    if requests:
        prefill_batches.append(requests)
    # 之后将剩余请求放进执行批
    requests = []
    for item in wait_requests:
        requests.append([item, 0])
        if len(requests) == batch_size:
            normal_batches.append(requests)
            requests = []
    if requests:
        normal_batches.append(requests)
    return prefill_batches, normal_batches

def simulate_schedule_dfs(root: XXXTrieNode, batch_size: int, rid_to_seq: Dict[RequestID, List[int]]):
    """
    针对 XXXTrie 的 DFS 调度算法的缓存命中率模拟
    """
    execute_prefix = 0
    prefill_batches: List[List[tuple[RequestID, int]]] = []
    requests = []

    def deep_first_search(node: XXXTrieNode):
        """深度优先搜索"""
        nonlocal prefill_batches, requests, execute_prefix
        for child in node.children.values():
            deep_first_search(child)
        if not node.children:
            # 叶子节点, 将第一个请求放入 prefill_batch, 将其他请求计算基于当前深度的前缀
            execute_prefix += node.depth * (len(node.request_ids) - 1)
            requests.append([next(iter(node.request_ids)), node.depth])
            if len(requests) == batch_size:
                prefill_batches.append(requests)
                requests = []
        else:
            # 非叶子节点, 直接计算基于当前深度的前缀
            execute_prefix += node.depth * len(node.request_ids)

    deep_first_search(root)
    prefill_batches.append(requests)

    return calculate_prefill_prefix(prefill_batches, rid_to_seq), execute_prefix

# ============================================================
# 基础 BFS 调度的方法及模拟器
# ============================================================

def schedule_bfs(root: XXXTrieNode, batch_size: int):
    """
    针对 XXXTrie 的广度优先调度算法, 用于和启发式进行对比实验. 两阶段调度:
    1. 第一阶段: 从每个 "根-叶子" 路径中取出第一条请求执行 (进行缓存预填充)
    2. 第二阶段: 收集所有未在第一阶段执行的请求，依次执行
    Args:
        root: XXXTrie 根节点
        batch_size: 批大小
    Returns:
        tuple[list[list[tuple[RequestID, int]]], list[list[tuple[RequestID, int]]]]
        填充阶段调度批次列表 + 执行阶段调度批次列表，每个批次包含 [RequestID, 前缀深度] 列表
    """
    prefill_batches: List[List[tuple[RequestID, int]]] = []
    normal_batches: List[List[tuple[RequestID, int]]] = []
    wait_requests: List[RequestID] = list(root.request_ids)
    scheduler_queue = deque(root.children.values())
    requests = []

    # 广度优先搜索
    while scheduler_queue:
        node = scheduler_queue.popleft()
        if node.children:
            # 非叶子节点, 全部请求放入等待集合
            wait_requests.extend(node.request_ids)
            for child in node.children.values():
                scheduler_queue.append(child)
        else:
            # 叶子节点, 将第一个请求放入 prefill_batch, 将其他请求放入等待集合
            wait_requests.extend(islice(node.request_ids, 1, None))
            requests.append([next(iter(node.request_ids)), node.depth])
            if len(requests) == batch_size:
                prefill_batches.append(requests)
                requests = []

    if requests:
        prefill_batches.append(requests)
    # 之后将剩余请求放进执行批
    requests = []
    for item in wait_requests:
        requests.append([item, 0])
        if len(requests) == batch_size:
            normal_batches.append(requests)
            requests = []
    if requests:
        normal_batches.append(requests)
    return prefill_batches, normal_batches

def simulate_schedule_bfs(root: XXXTrieNode, batch_size: int, rid_to_seq: Dict[RequestID, List[int]]):
    """
    针对 XXXTrie 的 BFS 调度算法的缓存命中率模拟
    """
    execute_prefix = 0
    prefill_batches: List[List[tuple[RequestID, int]]] = []
    scheduler_queue = deque(root.children.values())
    requests = []

    # 广度优先搜索
    while scheduler_queue:
        node = scheduler_queue.popleft()
        if node.children:
            # 非叶子节点, 直接计算基于当前深度的前缀
            execute_prefix += node.depth * len(node.request_ids)
            for child in node.children.values():
                scheduler_queue.append(child)
        else:
            # 叶子节点, 将第一个请求放入 prefill_batch, 将其他请求计算基于当前深度的前缀
            execute_prefix += node.depth * (len(node.request_ids) - 1)
            requests.append([next(iter(node.request_ids)), node.depth])
            if len(requests) == batch_size:
                prefill_batches.append(requests)
                requests = []

    if requests:
        prefill_batches.append(requests)
    return calculate_prefill_prefix(prefill_batches, rid_to_seq), execute_prefix

# ============================================================
# 其他工具
# ============================================================

def calculate_prefill_prefix(
    prefill_batches: List[List[tuple[RequestID, int]]],
    rid_to_seq: Dict[RequestID, List[int]]
) -> int:
    """基于预填充批计算该阶段缓存命中"""
    cache_root = radixCacheSimulator()
    prefill_prefix = 0
    for batch in prefill_batches:
        # 批内请求无法共享前缀
        for rid in batch:
            tokens = rid_to_seq.get(rid[0])
            prefill_prefix += cache_root.accessing_cache(tokens, 0)
        # 批执行后统一写入前缀
        for rid in batch:
            tokens = rid_to_seq.get(rid[0])
            cache_root.insert_cache(tokens, 0)
    return prefill_prefix

class radixCacheSimulator:
    """
    模拟 RadixCache 行为的模拟树
    """
    def __init__(self, token: Optional[int] = None):
        self.token = token
        self.children: Dict[int, "radixCacheSimulator"] = {}
    
    def insert_cache(
        self,
        tokens: List[int],
        index: int,
        end_index: Optional[int] = None,
    ):
        if index >= len(tokens) or (end_index is not None and index >= end_index):
            return
        if tokens[index] in self.children:
            self.children[tokens[index]].insert_cache(tokens, index + 1, end_index)
        else:
            child = radixCacheSimulator(token=tokens[index])
            self.children[tokens[index]] = child
            child.insert_cache(tokens, index + 1, end_index)
    
    def accessing_cache(self, tokens: List[int], index: int) -> int:
        if index >= len(tokens):
            return 0
        child = self.children.get(tokens[index])
        if child:
            return child.accessing_cache(tokens, index + 1) + 1
        else:
            return 0


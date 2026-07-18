from __future__ import annotations

import copy
from itertools import islice
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from xxxtrie import XXXTrieNode

RequestID = Tuple[str, int]

# ============================================================
# 启发式调度算法及其模拟器
# ============================================================

def schedule_heuristic(root: XXXTrieNode, batch_size: int):
    """
    针对 XXXTrie 的启发式调度算法. 两阶段调度:
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
    scheduler_queue = deque(root.children.values())
    wait_requests: List[RequestID] = list(root.request_ids)
    
    def split_child(node: XXXTrieNode):
        """从根向叶的主动裂解"""
        nonlocal scheduler_queue, wait_requests
        if len(node.children) >= 2:
            wait_requests.extend(node.request_ids)
            for child in node.children.values():
                scheduler_queue.append(child)
        elif len(node.children) == 1:
            wait_requests.extend(node.request_ids)
            split_child(list(node.children.values())[0])
        else:
            scheduler_queue.append(node)

    def split_branch(node: XXXTrieNode) -> tuple[RequestID, depth]:
        """DFS 找出分支的第一个叶节点, 并在沿途进行分支裂解"""
        nonlocal scheduler_queue, wait_requests
        if len(node.children) == 0:
            # 叶子节点, 将第一个请求返回, 将其他请求放入等待集合
            wait_requests.extend(islice(node.request_ids, 1, None))
            return [next(iter(node.request_ids)), node.depth]
        else:
            wait_requests.extend(node.request_ids)
            # 将其他分支对应的节点加入调度队列后, 进入第一个分支
            for value in islice(node.children.values(), 1, None):
                scheduler_queue.append(value)
            return split_branch(next(iter(node.children.values())))

    # 第一阶段
    while len(scheduler_queue) > 0:
        if len(scheduler_queue) < batch_size:
            # 先尝试主动裂解调度队列中的节点
            before_size = len(scheduler_queue)
            for i in range(before_size):
                if (len(scheduler_queue) < batch_size):
                    split_child(scheduler_queue.popleft())
                else:
                    break
            # 剩下的节点不足裂解以填充一个完整的批, 进入二阶段
            if before_size == len(scheduler_queue):
                break
        else:
            # 从每个调度节点中选择一条 "根-叶子" 路径上的请求填充批
            requests = []
            for i in range(batch_size):
                requests.append(split_branch(scheduler_queue.popleft()))
            prefill_batches.append(requests)

    # 第二阶段 - 首先将调度队列中的残余请求放进预填充批中
    if scheduler_queue:
        requests = []
        for item in scheduler_queue:
            requests.append(split_branch(item))
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

def simulate_heuristic_prefix(root: XXXTrieNode, batch_size: int, rid_to_seq: Dict[RequestID, List[int]]):
    """
    针对 XXXTrie 的启发式调度算法的缓存命中率模拟
    """
    execute_prefix = 0
    prefill_batches: List[List[tuple[RequestID, int]]] = []
    scheduler_queue = deque(root.children.values())
    
    def split_child(node: XXXTrieNode):
        """从根向叶的主动裂解"""
        nonlocal scheduler_queue, execute_prefix
        if len(node.children) >= 2:
            execute_prefix += node.depth * len(node.request_ids)
            for child in node.children.values():
                scheduler_queue.append(child)
        elif len(node.children) == 1:
            execute_prefix += node.depth * len(node.request_ids)
            split_child(list(node.children.values())[0])
        else:
            scheduler_queue.append(node)

    def split_branch(node: XXXTrieNode) -> tuple[RequestID, depth]:
        """DFS 找出分支的第一个叶节点, 并在沿途进行分支裂解"""
        nonlocal scheduler_queue, execute_prefix
        if len(node.children) == 0:
            # 叶子节点, 将第一个请求返回, 将其他请求计算命中长度
            execute_prefix += node.depth * (len(node.request_ids) - 1)
            return [next(iter(node.request_ids)), node.depth]
        else:
            # 非叶子节点上的每个请求, 命中长度等于当前节点深度
            execute_prefix += node.depth * len(node.request_ids)
            # 将其他分支对应的节点加入调度队列后, 进入第一个分支
            for value in islice(node.children.values(), 1, None):
                scheduler_queue.append(value)
            return split_branch(next(iter(node.children.values())))

    # 第一阶段
    while len(scheduler_queue) > 0:
        if len(scheduler_queue) < batch_size:
            # 先尝试主动裂解调度队列中的节点
            before_size = len(scheduler_queue)
            for i in range(before_size):
                if (len(scheduler_queue) < batch_size):
                    split_child(scheduler_queue.popleft())
                else:
                    break
            # 剩下的节点不足裂解以填充一个完整的批, 进入二阶段
            if before_size == len(scheduler_queue):
                break
        else:
            # 从每个调度节点中选择一条 "根-叶子" 路径上的请求填充批
            requests = []
            for i in range(batch_size):
                requests.append(split_branch(scheduler_queue.popleft()))
            prefill_batches.append(requests)

    # 第二阶段 - 首先将调度队列中的残余请求放进预填充批中
    if scheduler_queue:
        requests = []
        for item in scheduler_queue:
            requests.append(split_branch(item))
        prefill_batches.append(requests)
    
    return calculate_prefill_prefix(prefill_batches, rid_to_seq), execute_prefix

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
    
    def insert_cache(self, tokens: List[int], index: int):
        if index >= len(tokens):
            return
        if tokens[index] in self.children:
            self.children[tokens[index]].insert_cache(tokens, index + 1)
        else:
            child = radixCacheSimulator(token=tokens[index])
            self.children[tokens[index]] = child
            child.insert_cache(tokens, index + 1)
    
    def accessing_cache(self, tokens: List[int], index: int) -> int:
        child = self.children.get(tokens[index])
        if child:
            return child.accessing_cache(tokens, index + 1) + 1
        else:
            return 0
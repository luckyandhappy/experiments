"""
XXXTrie (Cache Scheduler Trie): 基于请求编号的纵向共享前缀树 (用于模型评估场景下的高效缓存复用)
"""

from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

RequestID = Tuple[str, int]

class XXXTrieNode:
    """
    Attributes:
        token: 从父节点到本节点的边所代表的 Token (根为 None)
        depth: 本节点对应前缀的总长度 (根 depth=0)
        request_ids: 共享从根到本节点路径为最长公共前缀的请求 ID 集合
        children: 子节点字典, key 为该节点的下一 Token
    """

    def __init__(self, token: Optional[int] = None, depth: int = 0, request_ids: Optional[Set[RequestID]] = None):
        self.token = token
        self.depth = depth
        self.request_ids: Set[RequestID] = request_ids if request_ids is not None else set()
        self.children: Dict[int, "XXXTrieNode"] = {}

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def has_requests(self) -> bool:
        return len(self.request_ids) > 0

    def total_request_count(self) -> int:
        return len(self.request_ids) + sum(
            c.total_request_count() for c in self.children.values()
        )

    def collect_all_request_ids(self) -> Set[RequestID]:
        ids = set(self.request_ids)
        for c in self.children.values():
            ids |= c.collect_all_request_ids()
        return ids

    def collect_leaves(self) -> List["XXXTrieNode"]:
        """收集所有"有请求停留"的节点 (即 request_ids 非空的节点)"""
        leaves: List["XXXTrieNode"] = []
        stack = [self]
        while stack:
            node = stack.pop()
            if node.request_ids:
                leaves.append(node)
            stack.extend(node.children.values())
        return leaves

    @classmethod
    def build_vertical(
        cls,
        request_token_seqs_map: Dict[str, List[List[int]]],
        req_indices: Optional[Set[RequestID]] = None,
        start_depth: int = 0,
    ) -> "XXXTrieNode":
        """纵向构建: 加载数据集中所有样本, 从第一个 Token 位置开始逐列扫描共享 Token 并创建 XXXTrie
        纵向构建算法示例 (请求 [1,2,3,4,5], [1,2,4,5,6], [1,3,5,7,9], [2,3,4,5,6]):
            初始时: 根节点的 request_ids = {0,1,2,3} (四个请求). 发现 Token 1 被请求 0,1,2 共享 
                → 在当前节点下创建代表 Toke 1 的子节点, 并将请求 {0,1,2} 记录移动到该节点中; 请求 3 留在此节点
            第二步: 进入子节点 (Token=1), request_ids = {0,1,2}. 发现 Token 2 被请求 0,1 共享
                → 在当前节点下创建代表 Token 2 的子节点, 并将请求 {0,1} 记录移动到该节点中; 请求 2 留在此节点
            第三步: 进入子节点 (Token=2), request_ids = {0,1}. 发现请求 0,1 无更多共享 Token → 停留在此节点
            结果: ROOT{request:[3]} → 1 {requests:[2]} → 2 {requests:[0,1]}
        Args:
            request_token_seqs_map: 来自多个数据集的请求 Token 序列
            dataset_name: 当前参与构建的数据集名称
            req_indices: 参与构建的请求 ID 集合 (None 表示全部)
            start_depth: 开始扫描的位置
        """
        if req_indices is None:
            req_indices = set()
            for dataset_name, seqs in request_token_seqs_map.items():
                for i in range(len(seqs)):
                    req_indices.add((dataset_name, i))

        node = XXXTrieNode(depth=start_depth)

        active: Set[RequestID] = set(req_indices)
        depth = start_depth

        # 统计 start_depth 处 Token 与请求之间的计数关系 (即每个 Token 被哪些请求所持有)
        groups: Dict[int, Set[RequestID]] = {}
        for rid in active:
            name, idx = rid
            seq = request_token_seqs_map[name][idx]
            if depth < len(seq):
                t = seq[depth]
                groups.setdefault(t, set()).add(rid)

        # 找到 start_depth 处被 ≥2 请求共享的 Token
        shared = {t: idxs for t, idxs in groups.items() if len(idxs) >= 2}

        for token, ids in shared.items():
            child = cls.build_vertical(
                request_token_seqs_map,
                ids,
                depth + 1
            )
            child.token = token
            node.children[token] = child
            active -= ids

        # 未进入任何子节点的请求停留在此
        node.request_ids = active
        return node

    def branch_extension(
        self,
        request_token_seqs: List[List[int]],
    ) -> "XXXTrieNode":
        """
        Trie 合并后, 从已有 Trie 节点开始继续扩展共享前缀
        只按需加载对应分支上的请求
        Args:
            request_token_seqs: 请求 Token 序列列表 (每个请求的 index 和 req_indices 一一对应)
            req_indices: 当前节点对应的所有请求
            start_depth: 当前节点对应的深度, 即继续寻找共享 Token 的位置
        Returns: 进行正确延展后的分支根节点
        """
        # 统计当前 depth 位置, 各 Token 对应的 activate 请求集合
        groups: Dict[int, Set[RequestID]] = {}

        for idx, rid in enumerate(self.request_ids):
            name = rid[0]
            seq = request_token_seqs[idx]
            if self.depth < len(seq):
                t = seq[self.depth]
                groups.setdefault(t, set()).add(rid)

        # 找到 start_depth 处被 ≥2 请求共享的 Token
        shared = {t: idxs for t, idxs in groups.items() if len(idxs) >= 2}

        for token, ids in shared.items():
            child = XXXTrieNode(token=token, depth=self.depth + 1, request_ids=ids)
            self.children[token] = child
            self.request_ids -= ids
            child.branch_extension(request_token_seqs)


    def split_by_ratio(
        self,
        ratio: float,
    ) -> "XXXTrieNode":
        """ 按比例对 Trie 进行切分
            - 纵向切分
            - 优先保持完整共享前缀
            - 若最后一条路径超出 target, 则从叶节点 request 开始进行拆分
        """
        ratio = min(ratio, 1.0 - ratio)
        total = self.total_request_count()
        if total <= 1:
            warnings.warn("Trie contains <=1 request, cannot split.")
            return self, None
        target = max(1, round(total * ratio))
        count = 0
        # 新树根节点
        new_tree = XXXTrieNode(depth=0)

        # 遍历旧树路径并同步在新树中复制, 直到达到 Target
        def deep_first_split(new_node, old_node):
            nonlocal count
            child_to_remove = []
            # 先从根到叶在新树中构建路径
            for child in old_node.children.values():
                node = XXXTrieNode(depth=old_node.depth + 1, token=child.token)
                new_node.children[child.token] = node
                deep_first_split(node, child)
                if count <= target:
                    child_to_remove.append(child.token)
                else:
                    break
            # 再从叶到根移动请求
            for token in child_to_remove:
                old_node.children.pop(token)
            if count > target:
                return
            count += len(old_node.request_ids)
            if count <= target:
                new_node.request_ids = old_node.request_ids
            else:
                trans = len(old_node.request_ids) - count + target
                sorted_ids = sorted(old_node.request_ids)
                new_node.request_ids = set(sorted_ids[:trans])
                old_node.request_ids = set(sorted_ids[trans:])
            
        deep_first_split(new_tree, self)
        return new_tree

    def _remove_leaf_requests(self, leaves: List["XXXTrieNode"]):
        """从树中移除指定叶节点的 request_ids。"""
        for leaf in leaves:
            leaf.request_ids.clear()

    @classmethod
    def merge(
        cls,
        tree_a: "XXXTrieNode",
        tree_b: "XXXTrieNode",
        dataloader: RequestDataLoader,
    ) -> "XXXTrieNode":
        """ 合并两棵 Trie. 流程:
        1. 合并树结构并记录 request 数量发生变化的节点
        2. 对每个 request 数量发生变化的节点进行 branch_extension
        """
        merged = cls._deep_copy(tree_a)
        b_paths = []
        changed_nodes = []
        cls._collect_leaf_paths(tree_b, [], b_paths)

        for path_tokens in b_paths:
            a_node = merged
            b_node = tree_b
            for token in path_tokens:
                if token not in a_node.children:
                    a_node.children[token] = XXXTrieNode(
                        token=token,
                        depth=a_node.depth + 1,
                    )
                old_size = len(a_node.request_ids)
                a_node.request_ids |= b_node.request_ids
                b_node.request_ids.clear()  # 避免重复合并
                if len(a_node.request_ids) > old_size and len(a_node.request_ids) >= 2:
                    changed_nodes.append(a_node)
                a_node = a_node.children[token]
                b_node = b_node.children[token]
            old_size = len(a_node.request_ids)
            a_node.request_ids |= b_node.request_ids
            if len(a_node.request_ids) > old_size and len(a_node.request_ids) >= 2:
                changed_nodes.append(a_node)

        # 合并后请求数量发生增加的节点可能会 "发芽", 因此需要对其进行延展
        for node in changed_nodes:
            node.branch_extension(dataloader.load_requests(node.request_ids))
        return merged


    @classmethod
    def _collect_leaf_paths(
        cls,
        node: "XXXTrieNode",
        path: List[int],
        result: List[List[int]],
    ):
        """收集 Trie 中所有 Root -> Leaf 路径"""
        if node.is_leaf():
            result.append(path)
            return
        for token, child in node.children.items():
            cls._collect_leaf_paths(child, path + [token], result)

    @classmethod
    def _deep_copy(cls, node: "XXXTrieNode") -> "XXXTrieNode":
        new = XXXTrieNode(token=node.token, depth=node.depth)
        new.request_ids = set(node.request_ids)
        new.children = {t: cls._deep_copy(c) for t, c in node.children.items()}
        return new

    # ---- 调度辅助 ----

    def branches_for_scheduling(self) -> List[Tuple[int, Set[int]]]:
        """
        返回叶节点分支列表 (用于预填充调度)

        只收集 is_leaf=True 且 request_ids 非空的节点。
        非叶节点的请求不参与预填充——因为其前缀已被更深层的叶节点路径覆盖。

        Returns:
            [(depth, {rid, ...}), ...], 按 depth 降序
        """
        branches: List[Tuple[int, Set[int]]] = []
        stack = [self]
        while stack:
            node = stack.pop()
            if node.request_ids and not node.children:
                branches.append((node.depth, node.request_ids))
            stack.extend(node.children.values())
        branches.sort(key=lambda x: (-x[0], -len(x[1])))
        return branches

    def collect_non_leaf_requests(self) -> Set[int]:
        """收集所有非叶节点上的请求 ID（预填充阶段不处理）。"""
        ids: Set[int] = set()
        stack = [self]
        while stack:
            node = stack.pop()
            if node.request_ids and node.children:
                ids |= node.request_ids
            stack.extend(node.children.values())
        return ids

    def print_tree(self, indent: int = 0):
        """打印树结构"""
        prefix = "  " * indent
        reqs = f"reqs={sorted(self.request_ids)}" if self.request_ids else ""
        token_str = f"token={self.token}" if self.token is not None else "ROOT"
        print(f"{prefix}[{token_str} depth={self.depth} {reqs}]")
        for t, child in sorted(self.children.items()):
            child.print_tree(indent + 1)

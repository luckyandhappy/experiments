from typing import Dict, List, Tuple

RequestID = Tuple[str, int]


class RequestDataLoader:
    """
    根据 (dataset_name, request_idx) 按需加载请求 Token。

    dataset_map 可以是真实数据集，也可以是 mmap、LMDB、Arrow 等，
    这里只要求实现 get_request() 即可。
    """

    def __init__(self):
        _REQUEST_MAP: Dict[RequestID, List[int]] = {
            ("A", 0): [1, 2, 3, 4, 5],
            ("A", 1): [1, 2, 4, 5, 6],
            ("A", 2): [1, 3, 5, 7, 9],
            ("A", 3): [2, 3, 4, 5, 6],

            ("B", 0): [1, 2, 3, 4, 8],
            ("B", 1): [1, 2, 4, 5, 9],
            ("B", 2): [1, 3, 5, 7, 10],
            ("B", 3): [2, 3, 4, 5, 11],
        }
        self._REQUEST_MAP = _REQUEST_MAP
    
    def load_requests(self, request_ids: List[RequestID]) -> List[List[int]]:
        result: List[List[int]] = []

        for rid in request_ids:
            if rid not in self._REQUEST_MAP:
                raise KeyError(f"Unknown RequestID: {rid}")

            result.append(self._REQUEST_MAP[rid])

        return result
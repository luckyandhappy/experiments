from __future__ import annotations

import unittest

from scheduler import ScheduledRequest, schedule_heuristic
from xxxtrie import XXXTrieNode


def _task_rows(batches):
    return [
        [(task.request_id, task.kind, task.cache_prefix_len) for task in batch]
        for batch in batches
    ]


class LongPrefixTrieTests(unittest.TestCase):
    def test_two_requests_share_long_token_prefix(self):
        for prefix_len in (2000, 4096):
            with self.subTest(prefix_len=prefix_len):
                prefix = list(range(prefix_len))
                sequences = {"mme": [prefix + [1], prefix + [2]]}

                root = XXXTrieNode.build_vertical(sequences)
                batches = schedule_heuristic(root, batch_size=1)

                self.assertEqual(root.total_request_count(), 2)
                self.assertEqual(
                    root.collect_all_request_ids(), {("mme", 0), ("mme", 1)}
                )
                self.assertEqual(
                    batches[0],
                    [ScheduledRequest(("mme", 0), "prefill", prefix_len)],
                )
                self.assertEqual(
                    batches[1], [ScheduledRequest(("mme", 1), "normal")]
                )

    def test_short_sequence_scheduling_is_unchanged(self):
        cases = [
            (
                [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6], [1, 7], [8]],
                [
                    [(("d", 4), "normal", None), (("d", 0), "prefill", 3)],
                    [(("d", 3), "normal", None), (("d", 2), "normal", None)],
                    [(("d", 1), "normal", None)],
                ],
            ),
            (
                [[1, 2, 9], [1, 2, 8], [1, 3, 7], [1, 3, 6], [4, 5]],
                [
                    [(("d", 4), "normal", None), (("d", 0), "prefill", 2)],
                    [(("d", 1), "normal", None), (("d", 2), "prefill", 2)],
                    [(("d", 3), "normal", None)],
                ],
            ),
            (
                [[1, 2], [1, 2, 3], [9]],
                [
                    [(("d", 2), "normal", None), (("d", 0), "prefill", 2)],
                    [(("d", 1), "normal", None)],
                ],
            ),
            (
                [[1], [2], [3]],
                [
                    [(("d", 0), "normal", None), (("d", 1), "normal", None)],
                    [(("d", 2), "normal", None)],
                ],
            ),
        ]

        for sequences, expected in cases:
            with self.subTest(sequences=sequences):
                root = XXXTrieNode.build_vertical({"d": sequences})
                self.assertEqual(_task_rows(schedule_heuristic(root, 2)), expected)

    def test_multiple_long_branches_keep_independent_prefix_depths(self):
        left = [("left", index) for index in range(1500)]
        right = [("right", index) for index in range(1800)]
        root = XXXTrieNode.build_vertical(
            {"mme": [left + [0], left + [1], right + [0], right + [1]]}
        )

        tasks = [task for batch in schedule_heuristic(root, 2) for task in batch]
        producer_depths = sorted(
            task.cache_prefix_len for task in tasks if task.kind == "prefill"
        )

        self.assertEqual(producer_depths, [1500, 1800])
        self.assertEqual({task.request_id for task in tasks}, {
            ("mme", 0),
            ("mme", 1),
            ("mme", 2),
            ("mme", 3),
        })


if __name__ == "__main__":
    unittest.main()

from xxxtrie import XXXTrieNode
from scheduler import schedule_heuristic, simulate_heuristic_prefix
from dataloader import RequestDataLoader

req1 = [1,2,3,4,5]
req2 = [1,2,4,5,6]
req3 = [1,3,5,7,9]
req4 = [2,3,4,5,6]
reqs_a = [req1, req2, req3, req4]

req5 = [1,2,3,4,8]
req6 = [1,2,4,5,9]
req7 = [1,3,5,7,10]
req8 = [2,3,4,5,11]
reqs_b = [req5, req6, req7, req8]

request_token_seqs_map = {
    "A": reqs_a,
}

trie_root_a = XXXTrieNode.build_vertical(request_token_seqs_map)

request_token_seqs_map = {
    "B": reqs_b,
}

trie_root_b = XXXTrieNode.build_vertical(request_token_seqs_map)

merge = XXXTrieNode.merge(trie_root_a, trie_root_b, RequestDataLoader())

a, b = schedule_heuristic(merge, 3)
print(a)
print(b)

a, b = simulate_heuristic_prefix(merge, 3)
print(a)
print(b)


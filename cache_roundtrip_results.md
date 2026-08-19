# CSTrie cache export/restart/import results

Experiment date: 2026-08-17 (Asia/Shanghai)

## Protocol

Each dataset was exercised with the same end-to-end sequence:

1. Run the CSTrie backend and export a new persistent HiCache bundle.
2. Let the experiment and SGLang processes exit completely.
3. Start a new process, validate the bundle manifest/inventory, and mount the
   bundle read-only.
4. Run the same workload once and record both total cache hits and persistent
   storage hits.

All ten runs completed with exit code 0 and `status: ok`. Text datasets used
`run_experiment.py`; ChartQA and VQAv2 used
`run_multimodal_experiment.py`. The multimodal manifests explicitly record
`visual_encoder_cache_persisted: false`, so their results measure decoder KV
cache persistence only.

`elapsed_seconds` is the measured request-processing phase. Process startup,
dataset preparation, model loading, and the import inventory preflight are not
included. Total hit rate includes both in-memory radix hits and persistent
storage hits; storage hits are therefore a component of, not an amount to add
to, total hit tokens.

## Results

| Dataset | Requests | Prompt tokens | Export hit rate | Import hit rate | Hit-rate delta | Added hit tokens | Persistent storage hits | Requests with storage hits | Request time export -> import | Time change | Bundle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| advbench | 520 | 6,668 | 32.4385% | 32.9034% | **+0.4649 pp** | **+31** | 483 | 121 (23.27%) | 38.16s -> 21.06s | **44.82% faster** | 485 pages / 0.067 GiB |
| alpaca | 31,323 | 381,309 | 29.9542% | 29.9088% | -0.0454 pp | -173 | 12,536 | 5,758 (18.38%) | 308.19s -> 259.95s | **15.65% faster** | 12,543 pages / 1.723 GiB |
| squad | 10,570 | 1,972,761 | 72.5525% | 92.6043% | **+20.0518 pp** | **+395,574** | 348,400 | 2,664 (25.20%) | 138.91s -> 225.42s | 62.28% slower | 348,435 pages / 47.850 GiB |
| chartqa | 2,500 | 1,180,286 | 37.5603% | 66.9234% | **+29.3631 pp** | **+346,568** | 346,148 | 801 (32.04%) | 164.55s -> 228.54s | 38.89% slower | 346,678 pages / 47.609 GiB |
| vqav2 | 52,818 | 15,100,222 | 75.4905% | 96.0283% | **+20.5379 pp** | **+3,101,265** | 2,713,210 | 14,632 (27.70%) | 1,248.76s -> 1,909.49s | 52.91% slower | 2,713,219 pages / 372.604 GiB |
| **Weighted/total** | **97,731** | **18,641,246** | **71.8311%** | **92.4481%** | **+20.6170 pp** | **+3,843,265** | **3,420,777** | **23,976 (24.53%)** | 1,898.56s -> 2,644.46s | 39.29% slower | **3,421,360 pages / 469.852 GiB** |

## Improvements to highlight

- Across all five workloads, weighted hit rate increased from **71.8311% to
  92.4481%**, a **20.6170 percentage-point** absolute gain (28.70% relative)
  and **3,843,265 additional hit tokens**.
- ChartQA produced the largest hit-rate gain: **+29.3631 pp**. VQAv2 and SQuAD
  also gained **+20.5379 pp** and **+20.0518 pp**, respectively.
- VQAv2 reached **96.0283%** total hit rate after restart, with **2,713,210
  persistent storage-hit tokens**. This is direct evidence that the newly
  started process consumed the exported cache rather than merely rebuilding an
  in-memory cache.
- Every dataset recorded non-zero persistent storage hits after restart.
  Across the suite, **23,976 requests** (24.53%) consumed at least one token
  from the imported bundle.
- The two smaller text bundles improved request-processing time: advbench was
  **44.82% faster** and alpaca was **15.65% faster**.

Alpaca's total hit-rate change (-0.0454 pp, 173 tokens) is negligible relative
to 381,309 input tokens, while the import run still recorded 12,536 persistent
hits and completed faster. The small difference is consistent with run-level
scheduling/cache-overlap variation rather than a failed import.

## I/O trade-off exposed by the full runs

SQuAD, ChartQA, and VQAv2 improved cache hit rates substantially but took longer
in the request phase. Their bundles store one 144 KiB KV page per file. The
largest run, VQAv2, performs lookups across 2.71 million page files and reads
from a 372.6 GiB bundle; small-file metadata and random-read latency outweigh
the saved GPU prefill work on this storage layout.

The next performance optimization should therefore target the persistence
format/read path rather than cache correctness: pack many KV pages into larger
container files, maintain a compact page index, and batch or asynchronously
prefetch adjacent pages. The round-trip implementation itself is functioning:
bundle identity/inventory validation passed, all imports mounted read-only, and
all five restarted runs reported real storage hits.

## Reproducibility records

| Dataset | Bundle ID | Export result | Import result |
|---|---|---|---|
| advbench | `61d39105bb1d...` | `/tmp/cstrie-advbench-cache-e2e.OJpRlo/export-results/advbench/9f1d9758c1a2/result.json` | `/tmp/cstrie-advbench-cache-e2e.OJpRlo/import-results-fixed/advbench/dbec202dbe90/result.json` |
| alpaca | `af358550dc45...` | `/tmp/cstrie-all-cache-e2e.ngL9Zt/alpaca-export/alpaca/941683758ff2/result.json` | `/tmp/cstrie-all-cache-e2e.ngL9Zt/alpaca-import/alpaca/f43cb8857b9b/result.json` |
| squad | `41254bdb62cc...` | `/tmp/cstrie-all-cache-e2e.ngL9Zt/squad-export/squad/3caf3225f682/result.json` | `/tmp/cstrie-all-cache-e2e.ngL9Zt/squad-import/squad/d7813e1c4ac9/result.json` |
| chartqa | `b3605aec646a...` | `/tmp/cstrie-chartqa-cache-e2e.kNuoIM/export-results/chartqa/7822913e0c9b/result.json` | `/tmp/cstrie-chartqa-cache-e2e.kNuoIM/import-results/chartqa/221375f6baf3/result.json` |
| vqav2 | `e10c11c95541...` | `/tmp/cstrie-all-cache-e2e.ngL9Zt/vqav2-export/vqav2/da0a0107f44e/result.json` | `/tmp/cstrie-all-cache-e2e.ngL9Zt/vqav2-import/vqav2/6a1d81606417/result.json` |


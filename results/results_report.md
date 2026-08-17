# CSTrie Experiment Results

| Dataset | Run ID | Backend | Policy | Status | Runs | Total Tokens | Hit Tokens | Peak Cache Tokens | Peak Cache Size | Micro | Macro | Result |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| advbench | `2727a283e3c1` | cstrie | cstrie | ok | 1/1 | 6,668 | 2,163 | 485 | 68.21 MiB | 32.44% | 34.16% | [advbench/2727a283e3c1/result.json](advbench/2727a283e3c1/result.json) |
| advbench | `2727a283e3c1` | sglang | native | ok | 1/1 | 6,668 | 2,221 | 4,943 | 695.14 MiB | 33.31% | 35.19% | [advbench/2727a283e3c1/result.json](advbench/2727a283e3c1/result.json) |
| advbench | `6a002a1a14a6` | vllm | native | ok | 1/1 | 6,668 | 0 | 0 | N/A | 0.00% | 0.00% | [advbench/6a002a1a14a6/result.json](advbench/6a002a1a14a6/result.json) |
| alpaca | `15eff71de972` | vllm | native | ok | 1/1 | 381,309 | 16 | 0 | N/A | 0.00% | 0.00% | [alpaca/15eff71de972/result.json](alpaca/15eff71de972/result.json) |
| alpaca | `241f7637c83f` | cstrie | cstrie | ok | 1/1 | 381,309 | 114,002 | 12,543 | 1763.94 MiB | 29.90% | 32.24% | [alpaca/241f7637c83f/result.json](alpaca/241f7637c83f/result.json) |
| alpaca | `241f7637c83f` | sglang | native | ok | 1/1 | 381,309 | 92,640 | 22,912 | 3222.14 MiB | 24.30% | 26.40% | [alpaca/241f7637c83f/result.json](alpaca/241f7637c83f/result.json) |
| chartqa | `185f821140b9` | cstrie | cstrie | ok | 1/1 | 1,180,286 | 442,453 | 13,856 | 1948.64 MiB | 37.49% | 38.23% | [chartqa/185f821140b9/result.json](chartqa/185f821140b9/result.json) |
| chartqa | `b0f17056f618` | sglang | native | ok | 3/3 | 3,540,858 | 1,078,557 | 13,865 | 1949.91 MiB | 30.46% | 29.56% | [chartqa/b0f17056f618/result.json](chartqa/b0f17056f618/result.json) |
| chartqa | `ce4d534b6c02` | vllm | native | ok | 1/1 | 1,180,286 | 429,952 | 1,201 | N/A | 36.43% | 37.01% | [chartqa/ce4d534b6c02/result.json](chartqa/ce4d534b6c02/result.json) |
| squad | `5a09f986e081` | vllm | native | ok | 1/1 | 1,972,761 | 1,408,688 | 0 | N/A | 71.41% | 69.94% | [squad/5a09f986e081/result.json](squad/5a09f986e081/result.json) |
| squad | `ad3f99bf0c74` | cstrie | cstrie | ok | 1/1 | 1,972,761 | 1,431,050 | 22,901 | 3220.59 MiB | 72.54% | 71.76% | [squad/ad3f99bf0c74/result.json](squad/ad3f99bf0c74/result.json) |
| squad | `ad3f99bf0c74` | sglang | native | ok | 1/1 | 1,972,761 | 1,126,234 | 22,912 | 3222.14 MiB | 57.09% | 54.16% | [squad/ad3f99bf0c74/result.json](squad/ad3f99bf0c74/result.json) |
| vqav2 | `a0cc12f84d3b` | cstrie | cstrie | ok | 1/1 | 15,100,222 | 11,394,814 | 13,855 | 1948.50 MiB | 75.46% | 75.33% | [vqav2/a0cc12f84d3b/result.json](vqav2/a0cc12f84d3b/result.json) |
| vqav2 | `d5d4a9c12a8a` | vllm | native | ok | 1/1 | 15,100,222 | 11,489,216 | 0 | N/A | 76.09% | 75.82% | [vqav2/d5d4a9c12a8a/result.json](vqav2/d5d4a9c12a8a/result.json) |
| vqav2 | `f2306ef0d982` | sglang | native | ok | 1/1 | 15,100,222 | 10,395,846 | 13,866 | 1950.05 MiB | 68.85% | 67.97% | [vqav2/f2306ef0d982/result.json](vqav2/f2306ef0d982/result.json) |

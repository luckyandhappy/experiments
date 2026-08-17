# CSTrie Experiment Results

| Dataset | Run ID | Backend | Policy | Status | Runs | Total Tokens | Hit Tokens | Peak Cache Tokens | Peak Cache Size | Micro | Macro | Result |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| advbench | `2727a283e3c1` | cstrie | cstrie | ok | 1/1 | 6,668 | 2,163 | 485 | 68.21 MiB | 32.44% | 34.16% | [advbench/2727a283e3c1/result.json](advbench/2727a283e3c1/result.json) |
| advbench | `2727a283e3c1` | sglang | native | ok | 1/1 | 6,668 | 2,221 | 4,943 | 695.14 MiB | 33.31% | 35.19% | [advbench/2727a283e3c1/result.json](advbench/2727a283e3c1/result.json) |
| chartqa | `b0f17056f618` | sglang | native | ok | 3/3 | 3,540,858 | 1,078,557 | 13,865 | 1949.91 MiB | 30.46% | 29.56% | [chartqa/b0f17056f618/result.json](chartqa/b0f17056f618/result.json) |
| vqav2 | `legacy:results` | vllm | native | ok | 1/1 | 15,100,222 | 11,489,216 | 0 | N/A | 76.09% | 75.82% | [multimodal/vqav2/results.json](multimodal/vqav2/results.json) |
| squad | `ad3f99bf0c74` | cstrie | cstrie | ok | 1/1 | 1,972,761 | 1,431,050 | 22,901 | 3220.59 MiB | 72.54% | 71.76% | [squad/ad3f99bf0c74/result.json](squad/ad3f99bf0c74/result.json) |
| squad | `ad3f99bf0c74` | sglang | native | ok | 1/1 | 1,972,761 | 1,126,234 | 22,912 | 3222.14 MiB | 57.09% | 54.16% | [squad/ad3f99bf0c74/result.json](squad/ad3f99bf0c74/result.json) |
| vqav2 | `5c695c84b60b` | sglang | native | ok | 1/1 | 15,100,222 | N/A | N/A | N/A | N/A | N/A | [vqav2/5c695c84b60b/result.json](vqav2/5c695c84b60b/result.json) |

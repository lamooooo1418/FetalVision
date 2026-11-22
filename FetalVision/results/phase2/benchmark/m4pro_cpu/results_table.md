# Benchmark m4pro_cpu

Conditions: Apple M4 Pro, 14 cores, 24.0 GB RAM, GPU None, Darwin 24.3.0, Python 3.12.6, onnxruntime 1.31.1, torch 2.15.0, tensorflow 2.21.0. Images: 32 x [540, 800] from /workspace/data/hc18/raw/parity_samples, batch 1, warm-up 20, N=200, ORT threads default, force_head=True, detector conf override=None. Stand-in models: [].

| backend | stage | input | p50 ms | p95 ms | p99 ms | img/s | peak RSS MB |
|---|---|---|---|---|---|---|---|
| ort_cpu | classify | 224x224 | 1.5 | 1.6 | 1.8 | 667.1 | 1418 |
| ort_cpu | detect | 640x640 | 13.6 | 14.8 | 16.7 | 72.8 | 1418 |
| ort_cpu | segment | 352x512 | 26.5 | 34.6 | 39.9 | 37.1 | 1418 |
| ort_cpu | measure | 540x800 | 0.7 | 1.1 | 1.2 | 1372.8 | 1418 |
| ort_cpu | **full pipeline** | 540x800 | 46.2 | 52.0 | 55.1 | 21.4 | 1418 |
| ort_cpu | classify batch_8 | 224x224 | - | - | - | 1225.2 | |
| ort_cpu | classify batch_32 | 224x224 | - | - | - | 1432.8 | |
| ort_coreml | classify | 224x224 | 3.3 | 4.1 | 4.9 | 293.2 | 1418 |
| ort_coreml | detect | 640x640 | 7.3 | 8.8 | 9.5 | 134.2 | 1418 |
| ort_coreml | segment | 352x512 | 5.8 | 6.5 | 6.8 | 170.6 | 1418 |
| ort_coreml | measure | 540x800 | 0.8 | 1.3 | 1.5 | 1194.9 | 1418 |
| ort_coreml | **full pipeline** | 540x800 | 18.7 | 21.4 | 26.1 | 52.3 | 1418 |
| ort_coreml | classify batch_8 | 224x224 | - | - | - | 694.1 | |
| ort_coreml | classify batch_32 | 224x224 | - | - | - | 1022.6 | |
| native | classify | 224x224 | 44.3 | 53.8 | 56.9 | 21.9 | 1567 |
| native | detect | 640x640 | 37.4 | 48.2 | 58.1 | 25.6 | 1567 |
| native | segment | 352x512 | 121.4 | 134.8 | 143.7 | 8.2 | 1567 |

HTTP service process RSS: 650 MB (uvicorn + ORT only). Note: the per-backend peak RSS above is the benchmark process, which also imports TensorFlow/PyTorch for the native comparison.

| HTTP /predict | p50 ms | p95 ms | p99 ms | req/s | errors |
|---|---|---|---|---|---|
| concurrency_1 | 57.8 | 67.5 | 72.8 | 16.96 | 0 |
| concurrency_4 | 222.2 | 291.5 | 398.3 | 16.73 | 0 |


Acceptance:
- ort_cpu: pipeline_p95_le_1500ms -> PASS (actual 51.99 <= 1500.0)
- ort_coreml: pipeline_p95_le_1500ms -> PASS (actual 21.39 <= 1500.0)

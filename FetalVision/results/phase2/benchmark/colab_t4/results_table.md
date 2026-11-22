# Benchmark colab_t4

Conditions: Intel(R) Xeon(R) CPU @ 2.00GHz, 2 cores, 12.7 GB RAM, GPU Tesla T4, 15360 MiB, Linux 6.6.159+, Python 3.13.16, onnxruntime None, torch 2.14.1+cu130, tensorflow 2.21.0. Images: 32 x [540, 800] from /content/data/hc18/raw/parity_samples, batch 1, warm-up 20, N=200, ORT threads default, force_head=True, detector conf override=None. Stand-in models: [].

| backend | stage | input | p50 ms | p95 ms | p99 ms | img/s | peak RSS MB |
|---|---|---|---|---|---|---|---|
| ort_cuda | classify | 224x224 | 3.3 | 4.1 | 5.3 | 292.2 | 2461 |
| ort_cuda | detect | 640x640 | 14.7 | 17.2 | 19.4 | 70.6 | 2461 |
| ort_cuda | segment | 352x512 | 13.2 | 14.2 | 15.5 | 75.0 | 2461 |
| ort_cuda | measure | 540x800 | 3.5 | 4.0 | 5.7 | 276.4 | 2461 |
| ort_cuda | **full pipeline** | 540x800 | 32.7 | 42.4 | 45.2 | 28.6 | 2461 |
| ort_cuda | classify batch_8 | 224x224 | - | - | - | 2313.3 | |
| ort_cuda | classify batch_32 | 224x224 | - | - | - | 2371.7 | |
| ort_cpu | classify | 224x224 | 4.4 | 5.6 | 6.4 | 220.6 | 2820 |
| ort_cpu | detect | 640x640 | 95.5 | 151.8 | 225.5 | 9.1 | 2820 |
| ort_cpu | segment | 352x512 | 203.2 | 302.8 | 315.5 | 4.5 | 2820 |
| ort_cpu | measure | 540x800 | 3.6 | 4.2 | 5.0 | 269.9 | 2820 |
| ort_cpu | **full pipeline** | 540x800 | 312.2 | 448.5 | 463.3 | 3.0 | 2820 |
| ort_cpu | classify batch_8 | 224x224 | - | - | - | 186.5 | |
| ort_cpu | classify batch_32 | 224x224 | - | - | - | 216.7 | |

HTTP service process RSS: 442 MB (uvicorn + ORT only). Note: the per-backend peak RSS above is the benchmark process, which also imports TensorFlow/PyTorch for the native comparison.

| HTTP /predict | p50 ms | p95 ms | p99 ms | req/s | errors |
|---|---|---|---|---|---|
| concurrency_1 | 333.1 | 471.3 | 493.6 | 2.73 | 0 |
| concurrency_4 | 1295.2 | 2102.8 | 2463.0 | 2.81 | 0 |


Acceptance:
- ort_cuda: pipeline_p95_le_200ms -> PASS (actual 42.40 <= 200.0), throughput_ge_5 -> PASS (actual 28.56 >= 5.0)
- ort_cpu: pipeline_p95_le_1500ms -> PASS (actual 448.51 <= 1500.0)

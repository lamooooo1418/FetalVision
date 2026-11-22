## Parity: segmenter

| item | value |
|---|---|
| ONNX | head_segmenter.onnx |
| native | head_segmenter.pt (segmentation_models_pytorch / torch 2.15.0) |
| images | 48 (32 real HC18 + noise) |
| stand-in | False  |
| providers | CPUExecutionProvider |
| prob max |Δ| | 9.388e-06 |
| Dice min / mean | 1.000000 / 1.000000 (criterion ≥ 0.999) |
| latency native p50 | 279.7 ms |
| latency ONNX p50 | 24.4 ms |
| PASSED | True |

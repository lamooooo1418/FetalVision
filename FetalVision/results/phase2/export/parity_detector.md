## Parity: detector

| item | value |
|---|---|
| ONNX | head_detector.onnx |
| native | head_detector.pt (ultralytics 8.4.208 / torch 2.15.0) |
| images | 48 (32 real HC18 + noise) |
| stand-in | False  |
| providers | CPUExecutionProvider |
| conf / iou | 0.85 / 0.3 |
| raw output max |Δ| | 2.258e-03 |
| images with boxes (native/onnx) | 32 / 32 |
| best-box IoU min / mean | 0.9988560080528259 / 0.9999640230089426 (criterion ≥ 0.99) |
| best-box |Δscore| max | 1.7881393432617188e-07 (criterion < 0.001) |
| all boxes IoU min / |Δscore| max | 0.9988560080528259 / 1.7881393432617188e-07 |
| latency native predict p50 | 27.4 ms |
| latency ONNX+decode p50 | 15.6 ms |
| PASSED | True |

## Parity: classifier

| item | value |
|---|---|
| ONNX | plane_classifier.onnx |
| native | p1v2-mnv3s-partial-ft-e40.keras (keras 3.15.1) |
| images | 48 (32 real HC18 + noise) |
| stand-in | False  |
| providers | CPUExecutionProvider |
| max |Δprob| | 2.325e-06 (criterion < 1e-04) |
| mean |Δprob| | 2.487e-07 |
| argmax agreement | 100.0 % |
| latency native p50 | 40.8 ms |
| latency ONNX p50 | 1.2 ms |
| PASSED | True |

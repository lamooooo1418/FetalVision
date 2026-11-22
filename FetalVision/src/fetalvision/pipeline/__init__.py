"""Packaged inference pipeline: ONNX Runtime sessions for the plane classifier, head detector and head segmenter,
numpy post-processing (YOLO decode + NMS, mask -> ellipse -> HC) and per-stage timings."""
from .pipeline import FetalVisionPipeline  # noqa: F401

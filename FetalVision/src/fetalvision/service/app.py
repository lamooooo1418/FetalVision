"""FastAPI service: GET /health, GET /manifest, GET /metrics, POST /predict (JSON), POST /predict/annotated (PNG).

    FETALVISION_SERVICE_CONFIG=configs/service.yaml uvicorn fetalvision.service.app:app --host 0.0.0.0 --port 8000 --workers 1
    python -m fetalvision.service.app --config configs/service.yaml --port 8000

The pipeline is created once at startup (warm-up: one dummy inference per stage). Requests are processed one at a
time per worker (the ONNX sessions are thread-safe but a lock keeps latency measurements clean).
"""
from __future__ import annotations
import argparse, collections, io, logging, os, sys, threading, time
from pathlib import Path
from typing import Optional
import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

import fetalvision
from fetalvision.common.config import load_config
from fetalvision.pipeline.pipeline import FetalVisionPipeline, strip_arrays
from fetalvision.pipeline.utils import package_versions, setup_logging

log = logging.getLogger("fetalvision.service")


class Metrics:
    def __init__(self, window: int = 1000):
        self.lock = threading.Lock()
        self.requests_total = 0
        self.errors_total = 0
        self.by_endpoint: dict[str, int] = collections.Counter()
        self.by_plane: dict[str, int] = collections.Counter()
        self.by_status: dict[str, int] = collections.Counter()
        self.latency = collections.deque(maxlen=window)
        self.stage_latency = {k: collections.deque(maxlen=window) for k in ("classify", "detect", "segment", "measure")}

    def record(self, endpoint: str, total_ms: float, result: dict | None, error: bool = False):
        with self.lock:
            self.requests_total += 1
            self.by_endpoint[endpoint] += 1
            if error:
                self.errors_total += 1
                return
            self.latency.append(total_ms)
            if result:
                self.by_plane[result["plane"]["label"]] += 1
                self.by_status[result["head_measurement"]] += 1
                for k, dq in self.stage_latency.items():
                    v = result["timings_ms"].get(k, 0.0)
                    if v > 0:
                        dq.append(v)

    @staticmethod
    def _pct(dq):
        if not dq:
            return {"n": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "mean_ms": None}
        a = np.asarray(dq, dtype=float)
        return {"n": int(a.size), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
                "p99_ms": float(np.percentile(a, 99)), "mean_ms": float(a.mean())}

    def snapshot(self) -> dict:
        with self.lock:
            return {"requests_total": self.requests_total, "errors_total": self.errors_total,
                    "requests_by_endpoint": dict(self.by_endpoint), "predictions_by_plane": dict(self.by_plane),
                    "predictions_by_head_measurement": dict(self.by_status),
                    "latency_request_ms": self._pct(self.latency),
                    "latency_stage_ms": {k: self._pct(dq) for k, dq in self.stage_latency.items()}}


class ServiceState:
    def __init__(self):
        self.pipeline: FetalVisionPipeline | None = None
        self.cfg: dict = {}
        self.started = time.time()
        self.warmup_ms: dict = {}
        self.metrics = Metrics()
        self.lock = threading.Lock()
        self.max_upload_bytes = 25 * 2**20

    def load(self, cfg_path: str | Path | None = None, cfg: dict | None = None):
        cfg = cfg if cfg is not None else load_config(cfg_path or os.environ.get("FETALVISION_SERVICE_CONFIG", "configs/service.yaml"))
        self.cfg = cfg
        svc = cfg.get("service", {})
        self.max_upload_bytes = int(float(svc.get("max_upload_mb", 25)) * 2**20)
        self.metrics = Metrics(int(svc.get("latency_window", 1000)))
        self.pipeline = FetalVisionPipeline.from_config(cfg)
        if svc.get("warmup", True):
            self.warmup_ms = self.pipeline.warmup()
        self.started = time.time()


state = ServiceState()


def create_app(cfg_path: str | Path | None = None, cfg: dict | None = None, pipeline: FetalVisionPipeline | None = None) -> FastAPI:
    app = FastAPI(title="FetalVision service", version=fetalvision.__version__,
                  description="Fetal plane classification + fetal-head detection/segmentation/HC measurement (research prototype).")

    @app.on_event("startup")
    def _startup():
        if pipeline is not None:
            state.pipeline, state.cfg = pipeline, cfg or {}
            state.warmup_ms = pipeline.warmup()
            return
        if state.pipeline is None:
            setup_logging()
            state.load(cfg_path, cfg)

    def _pipe() -> FetalVisionPipeline:
        if state.pipeline is None:
            raise HTTPException(503, "pipeline not loaded")
        return state.pipeline

    def _decode(upload: UploadFile, data: bytes) -> np.ndarray:
        import cv2
        if len(data) > state.max_upload_bytes:
            raise HTTPException(413, f"file too large (> {state.max_upload_bytes} bytes)")
        if not data:
            raise HTTPException(400, "empty file")
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(415, f"could not decode image {upload.filename!r}")
        return img

    def _run(endpoint: str, upload: UploadFile, data: bytes, pixel_size_mm, force_head, return_mask=False):
        pipe = _pipe()
        t0 = time.perf_counter()
        try:
            img = _decode(upload, data)
            with state.lock:
                res = pipe.run(img, pixel_size_mm, bool(force_head), return_mask=return_mask)
        except HTTPException:
            state.metrics.record(endpoint, 0.0, None, error=True)
            raise
        except Exception as e:
            state.metrics.record(endpoint, 0.0, None, error=True)
            log.exception("inference failed")
            raise HTTPException(500, f"inference failed: {e}")
        total = (time.perf_counter() - t0) * 1000
        res["timings_ms"]["request"] = round(total, 3)
        res["filename"] = upload.filename
        state.metrics.record(endpoint, total, res)
        return img, res

    @app.get("/health")
    def health():
        p = state.pipeline
        return {"status": "ok" if p is not None and p.classifier is not None else "degraded",
                "models_loaded": {"plane_classifier": bool(p and p.classifier), "head_detector": bool(p and p.detector),
                                  "head_segmenter": bool(p and p.segmenter)},
                "stand_in_models": p.manifest()["stand_in_models"] if p else [],
                "manifest_id": p.manifest_id if p else None,
                "versions": {k: v for k, v in package_versions().items() if k in ("fetalvision", "onnxruntime", "numpy", "cv2", "fastapi")},
                "providers": p.providers if p else None, "warmup_ms": {k: round(v, 1) for k, v in state.warmup_ms.items()},
                "uptime_s": round(time.time() - state.started, 1)}

    @app.get("/manifest")
    def manifest():
        return JSONResponse(_pipe().manifest())

    @app.get("/metrics")
    def metrics():
        return state.metrics.snapshot()

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), pixel_size_mm: Optional[float] = Form(None),
                      force_head: Optional[bool] = Form(False)):
        data = await file.read()
        _, res = _run("/predict", file, data, pixel_size_mm, force_head)
        return JSONResponse(strip_arrays(res))

    @app.post("/predict/annotated")
    async def predict_annotated(file: UploadFile = File(...), pixel_size_mm: Optional[float] = Form(None),
                                force_head: Optional[bool] = Form(False)):
        import cv2
        data = await file.read()
        img, res = _run("/predict/annotated", file, data, pixel_size_mm, force_head, return_mask=True)
        png = cv2.imencode(".png", _pipe().annotate(img, res))[1].tobytes()
        headers = {"X-FetalVision-Plane": res["plane"]["label"], "X-FetalVision-Probability": f"{res['plane']['probability']:.4f}",
                   "X-FetalVision-Head-Measurement": res["head_measurement"],
                   "X-FetalVision-HC-px": "" if not res.get("hc") or res["hc"]["px"] is None else f"{res['hc']['px']:.2f}",
                   "X-FetalVision-HC-mm": "" if not res.get("hc") or res["hc"]["mm"] is None else f"{res['hc']['mm']:.2f}",
                   "X-FetalVision-Total-ms": f"{res['timings_ms']['total']:.1f}"}
        return Response(content=png, media_type="image/png", headers=headers)

    return app


app = create_app()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/service.yaml")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    a = ap.parse_args(argv)
    import uvicorn
    os.environ["FETALVISION_SERVICE_CONFIG"] = a.config
    cfg = load_config(a.config).get("service", {})
    uvicorn.run("fetalvision.service.app:app", host=a.host or cfg.get("host", "0.0.0.0"), port=int(a.port or cfg.get("port", 8000)),
                workers=1, log_level="info")


if __name__ == "__main__":
    sys.exit(main())

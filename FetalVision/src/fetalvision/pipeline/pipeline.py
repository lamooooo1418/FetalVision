"""FetalVisionPipeline: classifier -> gate -> detector -> segmenter -> ellipse/HC with ONNX Runtime sessions.

Usage:
    from fetalvision.pipeline import FetalVisionPipeline
    pipe = FetalVisionPipeline.from_config("configs/service.yaml")
    result = pipe.run(image_bgr, pixel_size_mm=0.069)
    annotated = pipe.annotate(image_bgr, result)
"""
from __future__ import annotations
import hashlib, json, logging, time
from pathlib import Path
from typing import Any
import numpy as np

import fetalvision
from fetalvision.common.constants import CLASS_NAMES, CLASS_TO_IDX
from fetalvision.common.config import load_config
from fetalvision.common.metrics import apply_temperature, probs_to_logits
from .utils import resolve_path, load_json_if_exists, file_record, package_versions, hardware_description, write_json, jsonable
from . import preprocess as pre
from . import postprocess as post

log = logging.getLogger(__name__)

DEFAULT_CONF, DEFAULT_IOU, DEFAULT_T = 0.25, 0.5, 1.0
_GRAPH_OPT = {"all": "ORT_ENABLE_ALL", "extended": "ORT_ENABLE_EXTENDED", "basic": "ORT_ENABLE_BASIC", "disable": "ORT_DISABLE_ALL"}


class OrtModel:
    """Thin wrapper around an onnxruntime.InferenceSession with provider fallback and timing."""

    def __init__(self, path: Path, providers: list[str], intra: int = 0, inter: int = 0, opt: str = "all", name: str = "model"):
        import onnxruntime as ort
        self.path, self.name = Path(path), name
        avail = ort.get_available_providers()
        wanted = [p if isinstance(p, str) else p[0] for p in providers]
        use = [p for p in wanted if p in avail]
        missing = [p for p in wanted if p not in avail]
        if missing:
            log.warning("%s: providers %s not available (have %s)", name, missing, avail)
        if "CPUExecutionProvider" not in use:
            use.append("CPUExecutionProvider")
        so = ort.SessionOptions()
        so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, _GRAPH_OPT.get(opt, "ORT_ENABLE_ALL"))
        if intra:
            so.intra_op_num_threads = int(intra)
        if inter:
            so.inter_op_num_threads = int(inter)
        so.log_severity_level = 3
        self.session = ort.InferenceSession(str(self.path), sess_options=so, providers=use)
        self.providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.output_shapes = [o.shape for o in self.session.get_outputs()]
        log.info("%s loaded from %s (%.1f MB) providers=%s input=%s %s output=%s %s", name, self.path,
                 self.path.stat().st_size / 2**20, self.providers, self.input_name, self.input_shape,
                 self.output_names[0], self.output_shapes[0])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_names[0]], {self.input_name: x})[0]


class FetalVisionPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        rt = cfg.get("runtime", {})
        self.providers = list(rt.get("providers", ["CPUExecutionProvider"]))
        self._sess_kwargs = dict(providers=self.providers, intra=int(rt.get("intra_op_num_threads", 0) or 0),
                                 inter=int(rt.get("inter_op_num_threads", 0) or 0), opt=str(rt.get("graph_optimization_level", "all")))
        self.notes: list[str] = []
        self._load_classifier(cfg.get("classifier", {}))
        self._load_detector(cfg.get("detector", {}))
        self._load_segmenter(cfg.get("segmenter", {}))
        g = cfg.get("gating", {})
        self.head_class = g.get("head_class", "Brain")
        if self.head_class not in CLASS_TO_IDX:
            raise ValueError(f"gating.head_class={self.head_class!r} not in CLASS_NAMES {CLASS_NAMES}")
        self.head_idx = CLASS_TO_IDX[self.head_class]
        self.gate_min_prob = float(g.get("min_probability", 0.0))
        self.manifest_id = self._manifest_id()
        log.info("pipeline ready: manifest_id=%s classifier=%s detector=%s segmenter=%s", self.manifest_id,
                 self.classifier is not None, self.detector is not None, self.segmenter is not None)

    # ------------------------------------------------------------------------------------------------- construction
    @classmethod
    def from_config(cls, cfg: dict | str | Path) -> "FetalVisionPipeline":
        if not isinstance(cfg, dict):
            cfg = load_config(cfg)
        return cls(cfg)

    def _load_classifier(self, c: dict) -> None:
        self.classifier, self.classifier_meta = None, {}
        p = resolve_path(c.get("onnx"))
        if p is None or not p.exists():
            self.notes.append(f"classifier ONNX missing at {p}")
            log.error("classifier ONNX not found at %s", p)
        else:
            self.classifier = OrtModel(p, name="plane_classifier", **self._sess_kwargs)
        self.classifier_meta = load_json_if_exists(c.get("meta"), "plane_classifier.json") or {}
        self.class_names = list(self.classifier_meta.get("class_names", CLASS_NAMES))
        if self.class_names != list(CLASS_NAMES):
            log.warning("classifier meta class_names %s differ from CLASS_NAMES %s", self.class_names, CLASS_NAMES)
        self.cls_input_hw = tuple(self.classifier_meta.get("input_size", c.get("input_size", [224, 224])))[:2]
        # temperature: config override -> meta -> calibration.json (via model_selection.json) -> 1.0
        t, src = c.get("temperature"), "config"
        if t is None and "temperature" in self.classifier_meta:
            t, src = self.classifier_meta["temperature"], "plane_classifier.json"
        if t is None:
            sel = load_json_if_exists(c.get("model_selection_json"), "model_selection.json") or {}
            run = sel.get("selected_run") or sel.get("run_name")
            if run and c.get("calibration_json_template"):
                cal = load_json_if_exists(c["calibration_json_template"].format(run=run), "calibration.json") or {}
                if "temperature" in cal:
                    t, src = cal["temperature"], f"results/phase1/{run}/calibration.json"
        if t is None:
            t, src = DEFAULT_T, "default"
        self.temperature, self.temperature_source = float(t), src
        log.info("classifier temperature T=%.4f (%s)", self.temperature, src)

    def _load_detector(self, d: dict) -> None:
        self.detector, self.detector_meta = None, {}
        p = resolve_path(d.get("onnx"))
        if p is None or not p.exists():
            self.notes.append(f"detector ONNX missing at {p}")
            log.warning("detector ONNX not found at %s: head stage disabled", p)
        else:
            self.detector = OrtModel(p, name="head_detector", **self._sess_kwargs)
        self.detector_meta = load_json_if_exists(d.get("meta"), "head_detector.json") or {}
        self.imgsz = int(self.detector_meta.get("imgsz", d.get("imgsz", 640)))
        op = load_json_if_exists(d.get("operating_point_json"), "operating_point.json") or {}
        self.conf = float(d["conf"]) if d.get("conf") is not None else float(op.get("conf", DEFAULT_CONF))
        self.iou = float(d["iou"]) if d.get("iou") is not None else float(op.get("iou", DEFAULT_IOU))
        self.conf_source = "config" if d.get("conf") is not None else ("operating_point.json" if "conf" in op else "default")
        self.max_det = int(d.get("max_det", 10))
        self.class_filter = d.get("class_filter")
        log.info("detector operating point conf=%.3f iou=%.3f (%s)", self.conf, self.iou, self.conf_source)

    def _load_segmenter(self, s: dict) -> None:
        self.segmenter, self.segmenter_meta = None, {}
        p = resolve_path(s.get("onnx"))
        if p is None or not p.exists():
            self.notes.append(f"segmenter ONNX missing at {p}")
            log.warning("segmenter ONNX not found at %s: segmentation/HC disabled", p)
        else:
            self.segmenter = OrtModel(p, name="head_segmenter", **self._sess_kwargs)
        self.segmenter_meta = load_json_if_exists(s.get("meta"), "head_segmenter.json") or {}
        self.seg_input_wh = tuple(self.segmenter_meta.get("input_size", s.get("input_size", [512, 352])))[:2]
        if self.segmenter is not None:
            shp = self.segmenter.input_shape  # [1,3,H,W]
            if all(isinstance(v, int) for v in shp[2:]) and (shp[3], shp[2]) != tuple(self.seg_input_wh):
                log.warning("segmenter ONNX input %s overrides configured input_size %s", shp, self.seg_input_wh)
                self.seg_input_wh = (int(shp[3]), int(shp[2]))
        self.crop_margin = float(s.get("crop_margin", 0.10))
        self.mask_threshold = float(s.get("mask_threshold", 0.5))
        self.seg_normalize = str(self.segmenter_meta.get("normalize", s.get("normalize", "imagenet")))
        self.seg_grayscale = bool(self.segmenter_meta.get("grayscale", s.get("grayscale", False)))
        if "mask_threshold" in self.segmenter_meta and s.get("mask_threshold") is None:
            self.mask_threshold = float(self.segmenter_meta["mask_threshold"])
        self.min_mask_area = int(s.get("min_mask_area_px", 50))

    # ------------------------------------------------------------------------------------------------------ stages
    def classify(self, image_bgr: np.ndarray) -> dict:
        if self.classifier is None:
            raise RuntimeError("classifier not loaded")
        x = pre.classifier_input(image_bgr, self.cls_input_hw)
        raw = self.classifier(x)[0].astype(np.float64)
        probs = apply_temperature(probs_to_logits(raw[None]), self.temperature)[0] if self.temperature != 1.0 else raw
        k = int(np.argmax(probs))
        return {"label": self.class_names[k], "index": k, "probability": float(probs[k]),
                "probabilities": {n: float(p) for n, p in zip(self.class_names, probs)},
                "raw_probabilities": {n: float(p) for n, p in zip(self.class_names, raw)},
                "temperature": self.temperature}

    def classify_batch(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        x = np.concatenate([pre.classifier_input(im, self.cls_input_hw) for im in images_bgr], 0)
        return self.classifier(x)

    def detect(self, image_bgr: np.ndarray, conf: float | None = None, iou: float | None = None) -> dict:
        conf = self.conf if conf is None else conf
        iou = self.iou if iou is None else iou
        out = {"detected": False, "box_xyxy": None, "score": None, "class_id": None, "n_candidates": 0,
               "conf_threshold": conf, "iou_threshold": iou, "status": "ok"}
        if self.detector is None:
            out["status"] = "unavailable"
            return out
        x, params = pre.detector_input(image_bgr, self.imgsz)
        raw = self.detector(x)
        dec = post.decode_yolo(raw, conf, iou, params, self.max_det, self.class_filter)
        out["n_candidates"] = int(len(dec["scores"]))
        if len(dec["scores"]):
            i = int(np.argmax(dec["scores"]))
            out.update(detected=True, box_xyxy=[float(v) for v in dec["boxes_xyxy"][i]], score=float(dec["scores"][i]),
                       class_id=int(dec["classes"][i]))
        return out

    def segment(self, image_bgr: np.ndarray, box_xyxy) -> dict:
        h, w = image_bgr.shape[:2]
        out = {"mask": None, "crop_xyxy": None, "mask_area_px": 0, "status": "ok"}
        if self.segmenter is None:
            out["status"] = "unavailable"
            return out
        crop = pre.expand_box(box_xyxy, self.crop_margin, (h, w))
        x1, y1, x2, y2 = crop
        x = pre.segmenter_input(image_bgr[y1:y2, x1:x2], self.seg_input_wh, self.seg_normalize, self.seg_grayscale)
        logits = self.segmenter(x)[0, 0]
        prob = post.sigmoid(logits.astype(np.float32))
        mask = post.paste_mask(prob, crop, (h, w), self.mask_threshold)
        out.update(mask=mask, crop_xyxy=list(crop), mask_area_px=int(mask.sum()))
        return out

    def measure(self, mask: np.ndarray, pixel_size_mm: float | None) -> dict:
        return post.measure_hc(mask, pixel_size_mm, self.min_mask_area)

    # -------------------------------------------------------------------------------------------------------- run
    def run(self, image: np.ndarray, pixel_size_mm: float | None = None, force_head: bool = False,
            return_mask: bool = False) -> dict[str, Any]:
        t_total = time.perf_counter()
        image_bgr = pre.to_bgr3(np.ascontiguousarray(image))
        timings = {"classify": 0.0, "detect": 0.0, "segment": 0.0, "measure": 0.0, "total": 0.0}
        t0 = time.perf_counter()
        plane = self.classify(image_bgr)
        timings["classify"] = (time.perf_counter() - t0) * 1000
        is_head = plane["label"] == self.head_class and plane["probability"] >= self.gate_min_prob
        result: dict[str, Any] = {
            "plane": plane, "gate": {"head_plane": is_head, "forced": bool(force_head), "head_class": self.head_class},
            "head": None, "segmentation": None, "hc": None,
            "head_measurement": "not_applicable", "pixel_size_mm": pixel_size_mm,
            "image_hw": [int(image_bgr.shape[0]), int(image_bgr.shape[1])],
        }
        if is_head or force_head:
            t0 = time.perf_counter()
            head = self.detect(image_bgr)
            timings["detect"] = (time.perf_counter() - t0) * 1000
            result["head"] = {k: v for k, v in head.items()}
            if head["status"] == "unavailable":
                result["head_measurement"] = "detector_unavailable"
            elif not head["detected"]:
                result["head_measurement"] = "head_not_detected"
            else:
                t0 = time.perf_counter()
                seg = self.segment(image_bgr, head["box_xyxy"])
                timings["segment"] = (time.perf_counter() - t0) * 1000
                if seg["status"] == "unavailable":
                    result["head_measurement"] = "segmenter_unavailable"
                else:
                    t0 = time.perf_counter()
                    hc = self.measure(seg["mask"], pixel_size_mm)
                    timings["measure"] = (time.perf_counter() - t0) * 1000
                    result["segmentation"] = {"mask_area_px": hc["mask_area_px"], "ellipse": hc["ellipse"],
                                              "crop_xyxy": seg["crop_xyxy"], "mask_threshold": self.mask_threshold}
                    result["hc"] = {"px": hc["hc_px"], "mm": hc["hc_mm"], "implementation": hc["implementation"]}
                    result["head_measurement"] = "ok" if hc["hc_px"] is not None else "ellipse_fit_failed"
                    if return_mask:
                        result["mask"] = seg["mask"]
        timings["total"] = (time.perf_counter() - t_total) * 1000
        result["timings_ms"] = {k: round(v, 3) for k, v in timings.items()}
        result["manifest_id"] = self.manifest_id
        return result

    def annotate(self, image: np.ndarray, result: dict) -> np.ndarray:
        from .visualize import annotate
        return annotate(image, result)

    def warmup(self) -> dict:
        """One dummy inference per stage (forces kernel/graph initialisation). Returns the timings."""
        dummy = (np.random.RandomState(0).randint(0, 256, (540, 800, 3))).astype(np.uint8)
        t = {}
        t0 = time.perf_counter(); self.classify(dummy); t["classify"] = (time.perf_counter() - t0) * 1000
        if self.detector is not None:
            t0 = time.perf_counter(); self.detect(dummy); t["detect"] = (time.perf_counter() - t0) * 1000
        if self.segmenter is not None:
            t0 = time.perf_counter(); self.segment(dummy, [200, 100, 600, 440]); t["segment"] = (time.perf_counter() - t0) * 1000
        log.info("warm-up done: %s", {k: round(v, 1) for k, v in t.items()})
        return t

    # ---------------------------------------------------------------------------------------------------- manifest
    def _model_records(self) -> dict:
        recs = {}
        for name, mdl, meta in (("plane_classifier", self.classifier, self.classifier_meta),
                                ("head_detector", self.detector, self.detector_meta),
                                ("head_segmenter", self.segmenter, self.segmenter_meta)):
            rec = file_record(mdl.path) if mdl is not None else {"path": None, "exists": False, "size_bytes": None, "sha256": None}
            rec["stand_in"] = bool(meta.get("stand_in", False))
            rec["source"] = meta.get("source_keras") or meta.get("source_weights")
            rec["meta_sha256"] = meta.get("sha256")
            recs[name] = rec
        return recs

    def _manifest_id(self) -> str:
        recs = self._model_records()
        key = json.dumps({k: v["sha256"] for k, v in recs.items()}, sort_keys=True) + json.dumps(self.thresholds(), sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    def thresholds(self) -> dict:
        return {"temperature": self.temperature, "temperature_source": self.temperature_source,
                "conf": self.conf, "iou": self.iou, "conf_source": self.conf_source, "max_det": self.max_det,
                "crop_margin": self.crop_margin, "mask_threshold": self.mask_threshold, "min_mask_area_px": self.min_mask_area,
                "segmenter_normalize": self.seg_normalize, "segmenter_grayscale": self.seg_grayscale,
                "gate_min_probability": self.gate_min_prob, "head_class": self.head_class}

    def manifest(self) -> dict:
        recs = self._model_records()
        return {
            "manifest_id": self.manifest_id, "fetalvision_version": fetalvision.__version__,
            "models": recs, "stand_in_models": [k for k, v in recs.items() if v["stand_in"]],
            "class_names": self.class_names, "thresholds": self.thresholds(),
            "input_resolution": {"classifier_hw": list(self.cls_input_hw), "detector_imgsz": self.imgsz,
                                 "segmenter_wh": list(self.seg_input_wh)},
            "providers_requested": self.providers,
            "providers_active": {n: m.providers for n, m in (("plane_classifier", self.classifier), ("head_detector", self.detector),
                                                              ("head_segmenter", self.segmenter)) if m is not None},
            "versions": package_versions(), "hardware": hardware_description(), "notes": self.notes,
        }

    def save_manifest(self, path: str | Path | None = None) -> Path:
        path = path or self.cfg.get("manifest_out", "results/phase2/export/manifest.json")
        return write_json(resolve_path(path), self.manifest())


def strip_arrays(result: dict) -> dict:
    """JSON-safe copy of a run() result (drops the mask)."""
    r = {k: v for k, v in result.items() if k != "mask"}
    return jsonable(r)

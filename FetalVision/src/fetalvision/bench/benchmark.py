"""Benchmark: per-stage and full-pipeline latency/throughput/memory on real images, several backends, plus an HTTP benchmark.

    python -m fetalvision.bench.benchmark --config configs/service.yaml --out results/phase2/benchmark/m4pro_cpu \
        [--backends ort_cpu,ort_coreml,ort_cuda,native] [--n-iters 200] [--n-warmup 20] [--http] [--conf 0.001] [--no-force-head]

Stated conditions (recorded in metrics.json): hardware (CPU model, cores, RAM, GPU), OS, Python, library versions,
providers, input resolution per stage, batch size 1 (plus batched classifier at 8/32), warm-up iterations, N timed
iterations, thread settings. Images: the 32 HC18 parity samples cycled round-robin. metrics.json follows the results
contract (fetalvision.common.config.write_metrics). The Colab T4 rerun is the same command with --backends ort_cuda,ort_cpu.
"""
from __future__ import annotations
import argparse, copy, json, logging, os, socket, subprocess, sys, threading, time
from pathlib import Path
import numpy as np

import fetalvision
from fetalvision.common.config import load_config, write_metrics, save_resolved_config
from fetalvision.common.seed import set_seed
from fetalvision.pipeline.pipeline import FetalVisionPipeline
from fetalvision.pipeline import preprocess as pre, postprocess as post
from fetalvision.pipeline.utils import resolve_path, hardware_description, package_versions, write_json, setup_logging, load_json_if_exists
from fetalvision.export.onnx_utils import list_images
from .stats import latency_summary, throughput_img_per_s, time_loop, peak_rss_mb, current_rss_mb, check_criteria

log = logging.getLogger(__name__)

# Acceptance criteria fixed in advance (M8)
CRITERIA_T4 = {"pipeline_p95_le_200ms": {"metric": "full_pipeline.latency.p95_ms", "op": "<=", "value": 200.0},
               "throughput_ge_5": {"metric": "full_pipeline.throughput_img_per_s", "op": ">=", "value": 5.0}}
CRITERIA_CPU = {"pipeline_p95_le_1500ms": {"metric": "full_pipeline.latency.p95_ms", "op": "<=", "value": 1500.0}}
BACKEND_PROVIDERS = {"ort_cpu": ["CPUExecutionProvider"], "ort_coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
                     "ort_cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"], "ort_tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]}


def load_images(cfg: dict, images_dir: str | None, n: int = 32) -> tuple[list[np.ndarray], str]:
    import cv2
    cands = [images_dir, cfg.get("benchmark", {}).get("images_dir"),
             str(Path(os.environ.get("FETALVISION_DATA", "data")) / "hc18" / "raw" / "parity_samples"),
             str(Path(os.environ.get("FETALVISION_DATA", "data")) / "hc18" / "raw" / "training_set")]
    for c in cands:
        if c and Path(c).exists():
            files = list_images(c, n)
            imgs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in files]
            imgs = [i for i in imgs if i is not None]
            if imgs:
                return imgs, str(c)
    log.warning("no real images found; using %d synthetic 800x540 noise images", n)
    rng = np.random.RandomState(0)
    return [rng.randint(0, 256, (540, 800, 3), dtype=np.uint8) for _ in range(n)], "synthetic"


def gpu_mem_mb() -> float | None:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      stderr=subprocess.DEVNULL, timeout=5).decode().strip().splitlines()[0]
        return float(out)
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------------ ORT backends
def bench_ort_backend(cfg: dict, name: str, providers: list[str], imgs, n_warmup: int, n_iters: int, conf: float | None,
                      force_head: bool, batch_sizes=(8, 32)) -> dict | None:
    import onnxruntime as ort
    avail = ort.get_available_providers()
    if providers[0] not in avail:
        log.warning("backend %s skipped: %s not in %s", name, providers[0], avail)
        return None
    c = copy.deepcopy(cfg)
    c.setdefault("runtime", {})["providers"] = providers
    if conf is not None:
        c.setdefault("detector", {})["conf"] = conf
    rss0 = current_rss_mb()
    pipe = FetalVisionPipeline.from_config(c)
    res = {"backend": name, "providers_requested": providers, "providers_active": pipe.manifest()["providers_active"], "stages": {},
           "thresholds": pipe.thresholds(), "force_head": force_head}
    h, w = imgs[0].shape[:2]
    # per stage
    lat, wall = time_loop(pipe.classify, imgs, n_warmup, n_iters)
    res["stages"]["classify"] = {"latency": latency_summary(lat), "throughput_img_per_s": throughput_img_per_s(n_iters, wall),
                                 "input_resolution": list(pipe.cls_input_hw), "batch_size": 1}
    if pipe.detector is not None:
        lat, wall = time_loop(pipe.detect, imgs, n_warmup, n_iters)
        res["stages"]["detect"] = {"latency": latency_summary(lat), "throughput_img_per_s": throughput_img_per_s(n_iters, wall),
                                   "input_resolution": [pipe.imgsz, pipe.imgsz], "batch_size": 1}
    if pipe.segmenter is not None:
        box = [0.25 * w, 0.2 * h, 0.75 * w, 0.8 * h]
        lat, wall = time_loop(lambda im: pipe.segment(im, box), imgs, n_warmup, n_iters)
        res["stages"]["segment"] = {"latency": latency_summary(lat), "throughput_img_per_s": throughput_img_per_s(n_iters, wall),
                                    "input_resolution": [pipe.seg_input_wh[1], pipe.seg_input_wh[0]], "batch_size": 1,
                                    "note": "fixed central box (segmentation always runs regardless of the detector)"}
        # measure stage on a synthetic ellipse mask of image size
        import cv2
        m = np.zeros((h, w), np.uint8); cv2.ellipse(m, (w // 2, h // 2), (w // 4, h // 4), 20, 0, 360, 1, -1)
        lat, wall = time_loop(lambda _: pipe.measure(m, 0.1), imgs, min(n_warmup, 5), n_iters)
        res["stages"]["measure"] = {"latency": latency_summary(lat), "throughput_img_per_s": throughput_img_per_s(n_iters, wall),
                                    "input_resolution": [h, w], "batch_size": 1}
    # full pipeline
    lat, wall = time_loop(lambda im: pipe.run(im, 0.1, force_head), imgs, n_warmup, n_iters)
    outs = [pipe.run(im, 0.1, force_head) for im in imgs]
    stage_means = {k: float(np.mean([o["timings_ms"][k] for o in outs])) for k in ("classify", "detect", "segment", "measure")}
    res["full_pipeline"] = {"latency": latency_summary(lat), "throughput_img_per_s": throughput_img_per_s(n_iters, wall),
                            "stage_mean_ms_within_pipeline": stage_means,
                            "head_measurement_status_counts": {s: int(sum(o["head_measurement"] == s for o in outs)) for s in set(o["head_measurement"] for o in outs)},
                            "plane_counts": {s: int(sum(o["plane"]["label"] == s for o in outs)) for s in set(o["plane"]["label"] for o in outs)},
                            "input_resolution": [h, w], "batch_size": 1}
    # batched classifier throughput
    res["classifier_batched"] = {}
    for bs in batch_sizes:
        batch = [imgs[i % len(imgs)] for i in range(bs)]
        x = np.concatenate([pre.classifier_input(im, pipe.cls_input_hw) for im in batch], 0)
        try:
            for _ in range(3):
                pipe.classifier(x)
            n_b = max(5, n_iters // bs)
            t0 = time.perf_counter()
            for _ in range(n_b):
                pipe.classifier(x)
            wall = time.perf_counter() - t0
            res["classifier_batched"][f"batch_{bs}"] = {"throughput_img_per_s": throughput_img_per_s(n_b * bs, wall),
                                                        "latency_per_batch_ms": wall / n_b * 1000, "n_batches": n_b}
        except Exception as e:
            res["classifier_batched"][f"batch_{bs}"] = {"error": str(e)}
    res["memory"] = {"rss_before_load_mb": rss0, "rss_after_mb": current_rss_mb(), "peak_rss_mb": peak_rss_mb(), "gpu_mem_used_mb": gpu_mem_mb()}
    del pipe
    return res


# ---------------------------------------------------------------------------------------------------- native backends
def bench_native(cfg: dict, imgs, n_warmup: int, n_iters: int) -> dict:
    """Native Keras (classifier) / PyTorch (detector via ultralytics predict, segmenter via smp) for comparison."""
    out = {"backend": "native", "stages": {}}
    h, w = imgs[0].shape[:2]
    models_dir = resolve_path(cfg.get("models_dir", "models"))
    cls_meta = load_json_if_exists(cfg["classifier"].get("meta"), "plane_classifier.json") or {}
    src = cls_meta.get("source_keras")
    if src and Path(src).exists():
        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            import keras
            m = keras.models.load_model(src, compile=False)
            hw = tuple(cls_meta.get("input_size", [224, 224]))
            lat, wall = time_loop(lambda im: m(pre.classifier_input(im, hw), training=False), imgs, n_warmup, n_iters)
            out["stages"]["classify"] = {"latency": latency_summary(lat), "throughput_img_per_s": throughput_img_per_s(n_iters, wall),
                                         "framework": f"keras {keras.__version__} (TF CPU)", "input_resolution": list(hw), "batch_size": 1}
        except Exception as e:
            out["stages"]["classify"] = {"error": str(e)}
    det_meta = load_json_if_exists(cfg["detector"].get("meta"), "head_detector.json") or {}
    src = det_meta.get("source_weights")
    if src and Path(src).exists():
        try:
            import torch
            from ultralytics import YOLO
            y = YOLO(src)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            imgsz = int(det_meta.get("imgsz", 640))
            lat, wall = time_loop(lambda im: y.predict(im, imgsz=imgsz, conf=0.25, iou=0.5, device=dev, verbose=False, rect=False), imgs, n_warmup, n_iters)
            out["stages"]["detect"] = {"latency": latency_summary(lat), "throughput_img_per_s": throughput_img_per_s(n_iters, wall),
                                       "framework": f"ultralytics predict / torch {torch.__version__} ({dev})", "input_resolution": [imgsz, imgsz], "batch_size": 1}
        except Exception as e:
            out["stages"]["detect"] = {"error": str(e)}
    seg_meta = load_json_if_exists(cfg["segmenter"].get("meta"), "head_segmenter.json") or {}
    src = seg_meta.get("source_weights")
    if src and Path(src).exists():
        try:
            import torch
            from fetalvision.export.export_segmenter import build_unet
            u = build_unet(seg_meta.get("encoder", "mobilenet_v2"), src)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            u = u.to(dev)
            wh = tuple(seg_meta.get("input_size", [512, 352]))
            box = pre.expand_box([0.25 * w, 0.2 * h, 0.75 * w, 0.8 * h], 0.1, (h, w))

            def seg(im):
                x = torch.from_numpy(pre.segmenter_input(im[box[1]:box[3], box[0]:box[2]], wh, seg_meta.get("normalize", "imagenet"), bool(seg_meta.get("grayscale", False)))).to(dev)
                with torch.no_grad():
                    r = torch.sigmoid(u(x))
                    if dev == "cuda":
                        torch.cuda.synchronize()
                return r
            lat, wall = time_loop(seg, imgs, n_warmup, n_iters)
            out["stages"]["segment"] = {"latency": latency_summary(lat), "throughput_img_per_s": throughput_img_per_s(n_iters, wall),
                                        "framework": f"segmentation_models_pytorch / torch {torch.__version__} ({dev})", "input_resolution": [wh[1], wh[0]], "batch_size": 1}
        except Exception as e:
            out["stages"]["segment"] = {"error": str(e)}
    out["memory"] = {"peak_rss_mb": peak_rss_mb()}
    return out


# ------------------------------------------------------------------------------------------------------------- HTTP
def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def bench_http(cfg_path: str, imgs, n_requests: int = 100, concurrencies=(1, 4), force_head: bool = True, conf: float | None = None,
               timeout_s: float = 120.0) -> dict:
    """Start uvicorn in a subprocess, hit POST /predict with httpx at the given concurrencies, stop it."""
    import httpx, cv2
    port = _free_port()
    env = {**os.environ, "FETALVISION_SERVICE_CONFIG": cfg_path}
    if conf is not None:
        env["FETALVISION_BENCH_CONF"] = str(conf)
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "fetalvision.service.app:app", "--host", "127.0.0.1", "--port", str(port),
                             "--workers", "1", "--log-level", "warning"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    out = {"port": port, "n_requests": n_requests, "endpoint": "/predict", "force_head": force_head, "results": {}}
    try:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                if httpx.get(base + "/health", timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("service did not start")
        payloads = [cv2.imencode(".png", im)[1].tobytes() for im in imgs]
        for conc in concurrencies:
            lats, errors, lock = [], [0], threading.Lock()

            def worker(idx_range):
                with httpx.Client(timeout=60) as cl:
                    for i in idx_range:
                        files = {"file": (f"img_{i}.png", payloads[i % len(payloads)], "image/png")}
                        data = {"force_head": "true" if force_head else "false", "pixel_size_mm": "0.1"}
                        t = time.perf_counter()
                        try:
                            r = cl.post(base + "/predict", files=files, data=data)
                            ok = r.status_code == 200
                        except Exception:
                            ok = False
                        dt = (time.perf_counter() - t) * 1000
                        with lock:
                            if ok:
                                lats.append(dt)
                            else:
                                errors[0] += 1
            # warm-up
            worker(range(5))
            lats.clear(); errors[0] = 0
            chunks = [range(k, n_requests, conc) for k in range(conc)]
            threads = [threading.Thread(target=worker, args=(ch,)) for ch in chunks]
            t_start = time.perf_counter()
            for th in threads: th.start()
            for th in threads: th.join()
            wall = time.perf_counter() - t_start
            out["results"][f"concurrency_{conc}"] = {"latency": latency_summary(lats), "errors": errors[0],
                                                     "throughput_req_per_s": throughput_img_per_s(len(lats), wall), "wall_s": wall}
        out["server_metrics"] = httpx.get(base + "/metrics", timeout=10).json()
        out["health"] = httpx.get(base + "/health", timeout=10).json()
        try:  # RSS of the service process itself (ORT + FastAPI only; no TF/torch imports), the deployment-relevant number
            import psutil
            pr = psutil.Process(proc.pid)
            rss = pr.memory_info().rss + sum(c.memory_info().rss for c in pr.children(recursive=True))
            out["server_rss_mb"] = rss / 2**20
        except Exception as e:
            out["server_rss_mb"] = None; out["server_rss_error"] = str(e)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
    return out


# ------------------------------------------------------------------------------------------------------------ report
def _fmt(v, nd=1):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v))


def results_table(payload: dict) -> str:
    lines = ["| backend | stage | input | p50 ms | p95 ms | p99 ms | img/s | peak RSS MB |", "|---|---|---|---|---|---|---|---|"]
    for b in payload["backends"]:
        rss = (b.get("memory") or {}).get("peak_rss_mb")
        for st, v in b.get("stages", {}).items():
            if "latency" not in v:
                lines.append(f"| {b['backend']} | {st} | - | error: {v.get('error', '')[:40]} | | | | |"); continue
            L = v["latency"]
            lines.append(f"| {b['backend']} | {st} | {'x'.join(map(str, v.get('input_resolution', [])))} | {_fmt(L['p50_ms'])} | {_fmt(L['p95_ms'])} | {_fmt(L['p99_ms'])} | {_fmt(v.get('throughput_img_per_s'))} | {_fmt(rss, 0)} |")
        if "full_pipeline" in b:
            L = b["full_pipeline"]["latency"]
            lines.append(f"| {b['backend']} | **full pipeline** | {'x'.join(map(str, b['full_pipeline']['input_resolution']))} | {_fmt(L['p50_ms'])} | {_fmt(L['p95_ms'])} | {_fmt(L['p99_ms'])} | {_fmt(b['full_pipeline']['throughput_img_per_s'])} | {_fmt(rss, 0)} |")
        for k, v in b.get("classifier_batched", {}).items():
            if "throughput_img_per_s" in v:
                lines.append(f"| {b['backend']} | classify {k} | 224x224 | - | - | - | {_fmt(v['throughput_img_per_s'])} | |")
    if payload.get("http"):
        lines += ["", f"HTTP service process RSS: {_fmt(payload['http'].get('server_rss_mb'), 0)} MB (uvicorn + ORT only). Note: the per-backend peak RSS above is the benchmark process, which also imports TensorFlow/PyTorch for the native comparison.", "",
                  "| HTTP /predict | p50 ms | p95 ms | p99 ms | req/s | errors |", "|---|---|---|---|---|---|"]
        for k, v in payload["http"]["results"].items():
            L = v["latency"]
            lines.append(f"| {k} | {_fmt(L['p50_ms'])} | {_fmt(L['p95_ms'])} | {_fmt(L['p99_ms'])} | {_fmt(v['throughput_req_per_s'], 2)} | {v['errors']} |")
    return "\n".join(lines) + "\n"


def latency_figure(payload: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    backends = [b for b in payload["backends"] if b.get("stages")]
    stages = ["classify", "detect", "segment", "measure", "full"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / max(1, len(backends))
    for i, b in enumerate(backends):
        vals, errs = [], []
        for st in stages:
            v = b["full_pipeline"]["latency"] if st == "full" and "full_pipeline" in b else b["stages"].get(st, {}).get("latency")
            vals.append(v["p50_ms"] if v else 0.0); errs.append((v["p95_ms"] - v["p50_ms"]) if v else 0.0)
        x = np.arange(len(stages)) + i * width
        ax.bar(x, vals, width, yerr=[np.zeros(len(vals)), errs], capsize=3, label=b["backend"])
    ax.set_xticks(np.arange(len(stages)) + width * (len(backends) - 1) / 2); ax.set_xticklabels(stages)
    ax.set_ylabel("latency ms (bar = p50, whisker to p95)"); ax.set_yscale("log")
    hw = payload["conditions"]["hardware"]
    ax.set_title(f"{payload['tag']}: {hw.get('cpu_model')} | batch 1 | N={payload['conditions']['n_iters']} | images {Path(str(payload['conditions']['images'])).name}", fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_png, dpi=200); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/service.yaml")
    ap.add_argument("--out", required=True, help="results/phase2/benchmark/<tag>")
    ap.add_argument("--backends", default="ort_cpu,ort_coreml,ort_cuda,native")
    ap.add_argument("--images", default=None)
    ap.add_argument("--n-images", type=int, default=32)
    ap.add_argument("--n-warmup", type=int, default=20)
    ap.add_argument("--n-iters", type=int, default=200)
    ap.add_argument("--conf", type=float, default=None, help="detector conf override so stand-in detectors produce boxes (recorded)")
    ap.add_argument("--no-force-head", action="store_true", help="by default the full pipeline runs with force_head=True so all stages execute")
    ap.add_argument("--http", dest="http", action="store_true", default=True)
    ap.add_argument("--no-http", dest="http", action="store_false")
    ap.add_argument("--http-requests", type=int, default=100)
    ap.add_argument("--threads", type=int, default=0, help="onnxruntime intra-op threads (0 = default)")
    a = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(a.config)
    set_seed(int(cfg.get("seed", 42)))
    if a.threads:
        cfg.setdefault("runtime", {})["intra_op_num_threads"] = a.threads
    out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    tag = out.name
    imgs, img_src = load_images(cfg, a.images, a.n_images)
    force_head = not a.no_force_head
    backends = [b.strip() for b in a.backends.split(",") if b.strip()]
    results = []
    for b in backends:
        log.info("=== backend %s ===", b)
        if b in BACKEND_PROVIDERS:
            r = bench_ort_backend(cfg, b, BACKEND_PROVIDERS[b], imgs, a.n_warmup, a.n_iters, a.conf, force_head)
        elif b == "native":
            r = bench_native(cfg, imgs, min(a.n_warmup, 10), min(a.n_iters, 100))
        else:
            log.warning("unknown backend %s", b); r = None
        if r:
            results.append(r)
            log.info("%s done: %s", b, json.dumps({k: v["latency"]["p50_ms"] for k, v in r["stages"].items() if "latency" in v}))
    http = None
    if a.http:
        log.info("=== HTTP benchmark ===")
        http_cfg_path = a.config
        if a.conf is not None:  # write a temp config with the conf override so the service matches the pipeline settings
            import yaml
            c2 = copy.deepcopy(cfg); c2.setdefault("detector", {})["conf"] = a.conf
            c2.pop("_config_path", None)
            http_cfg_path = str(out / "service_config_http.yaml"); Path(http_cfg_path).write_text(yaml.safe_dump(c2, sort_keys=False))
        try:
            http = bench_http(http_cfg_path, imgs, a.http_requests, (1, 4), force_head, a.conf)
        except Exception as e:
            log.error("HTTP benchmark failed: %s", e); http = {"error": str(e)}
    import onnxruntime as ort
    pipe_probe = FetalVisionPipeline.from_config(cfg)
    man = pipe_probe.manifest()
    conditions = {"hardware": hardware_description(), "versions": package_versions(), "ort_available_providers": ort.get_available_providers(),
                  "images": img_src, "n_images": len(imgs), "image_resolution_hw": list(imgs[0].shape[:2]), "batch_size": 1,
                  "n_warmup": a.n_warmup, "n_iters": a.n_iters, "ort_intra_op_threads": cfg.get("runtime", {}).get("intra_op_num_threads", 0) or "default",
                  "input_resolution": man["input_resolution"], "thresholds": man["thresholds"], "detector_conf_override": a.conf,
                  "force_head": force_head, "stand_in_models": man["stand_in_models"], "models": man["models"], "manifest_id": man["manifest_id"],
                  "timer": "time.perf_counter, single process, sequential"}
    payload = {"tag": tag, "conditions": conditions, "backends": results, "http": http, "stand_in_models": man["stand_in_models"]}
    # criteria
    crit = {}
    for b in results:
        if "full_pipeline" in b:
            cset = CRITERIA_T4 if b["backend"] in ("ort_cuda", "ort_tensorrt") else CRITERIA_CPU
            crit[b["backend"]] = check_criteria(b, cset)
    payload["acceptance"] = crit
    write_json(out / "benchmark.json", payload)
    # metrics.json per results contract
    flat = {}
    for b in results:
        for st, v in b.get("stages", {}).items():
            if "latency" in v:
                for k in ("p50_ms", "p95_ms", "p99_ms"):
                    flat[f"{b['backend']}/{st}/{k}"] = v["latency"][k]
                flat[f"{b['backend']}/{st}/throughput_img_per_s"] = v["throughput_img_per_s"]
        if "full_pipeline" in b:
            for k in ("p50_ms", "p95_ms", "p99_ms"):
                flat[f"{b['backend']}/full_pipeline/{k}"] = b["full_pipeline"]["latency"][k]
            flat[f"{b['backend']}/full_pipeline/throughput_img_per_s"] = b["full_pipeline"]["throughput_img_per_s"]
            flat[f"{b['backend']}/peak_rss_mb"] = (b.get("memory") or {}).get("peak_rss_mb")
        for k, v in b.get("classifier_batched", {}).items():
            if "throughput_img_per_s" in v:
                flat[f"{b['backend']}/classify_{k}/throughput_img_per_s"] = v["throughput_img_per_s"]
    if http and "results" in http:
        flat["http/server_rss_mb"] = http.get("server_rss_mb")
        for k, v in http["results"].items():
            for q in ("p50_ms", "p95_ms", "p99_ms"):
                flat[f"http/{k}/{q}"] = v["latency"][q]
            flat[f"http/{k}/throughput_req_per_s"] = v["throughput_req_per_s"]
    write_metrics(out, tag, "phase2", "benchmark_images", conditions["image_resolution_hw"], len(imgs), int(cfg.get("seed", 42)), flat,
                  extra={"conditions": conditions, "acceptance": crit, "stand_in_models": man["stand_in_models"], "backends": backends})
    save_resolved_config(cfg, out)
    (out / "results_table.md").write_text(f"# Benchmark {tag}\n\nConditions: {conditions['hardware'].get('cpu_model')}, {conditions['hardware'].get('cpu_cores_logical')} cores, "
                                          f"{conditions['hardware'].get('ram_gb')} GB RAM, GPU {conditions['hardware'].get('gpu')}, {conditions['hardware'].get('os')}, Python {conditions['hardware'].get('python')}, "
                                          f"onnxruntime {conditions['versions'].get('onnxruntime')}, torch {conditions['versions'].get('torch')}, tensorflow {conditions['versions'].get('tensorflow')}. "
                                          f"Images: {len(imgs)} x {conditions['image_resolution_hw']} from {img_src}, batch 1, warm-up {a.n_warmup}, N={a.n_iters}, ORT threads {conditions['ort_intra_op_threads']}, "
                                          f"force_head={force_head}, detector conf override={a.conf}. Stand-in models: {man['stand_in_models']}.\n\n" + results_table(payload)
                                          + "\n\nAcceptance:\n" + "\n".join(f"- {b}: " + ", ".join(f"{n} -> {'PASS' if c['passed'] else 'FAIL' if c['passed'] is not None else 'n/a'} (actual {_fmt(c['actual'], 2)} {c['op']} {c['value']})" for n, c in cc.items()) for b, cc in crit.items()) + "\n")
    try:
        latency_figure(payload, out / "figures" / "latency_by_stage.png")
        write_json(out / "figures" / "captions.json", {"latency_by_stage.png": f"Per-stage and full-pipeline latency ({tag}); bars p50, whiskers to p95; log scale; batch 1; N={a.n_iters}; images {img_src}; stand-ins: {man['stand_in_models']}"})
    except Exception as e:
        log.warning("figure failed: %s", e)
    print((out / "results_table.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())

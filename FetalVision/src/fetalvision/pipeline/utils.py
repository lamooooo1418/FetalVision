"""Small helpers shared by the export, pipeline, service and benchmark modules (no heavy imports here)."""
from __future__ import annotations
import hashlib, json, logging, os, platform, subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def repo_root() -> Path:
    """Repository root = three levels above this file (src/fetalvision/pipeline/utils.py)."""
    return Path(__file__).resolve().parents[3]


def resolve_path(p: str | Path | None, base: Path | None = None) -> Path | None:
    """Resolve a config path: absolute as-is; relative first against the CWD, then against the repo root."""
    if p is None or p == "":
        return None
    p = Path(str(p)).expanduser()
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    root_candidate = (base or repo_root()) / p
    if root_candidate.exists():
        return root_candidate.resolve()
    return cwd_candidate.resolve()  # does not exist (yet): caller decides how to log it


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json_if_exists(path: str | Path | None, what: str = "file") -> dict | None:
    p = resolve_path(path)
    if p is None:
        return None
    if not p.exists():
        log.info("%s not found at %s (using defaults)", what, p)
        return None
    try:
        data = json.loads(p.read_text())
        log.info("%s loaded from %s", what, p)
        return data
    except Exception as e:  # pragma: no cover - corrupt file
        log.warning("%s at %s could not be parsed (%s); using defaults", what, p, e)
        return None


def file_record(path: str | Path | None) -> dict[str, Any]:
    """{path, exists, size_bytes, sha256} for the manifest."""
    p = resolve_path(path)
    if p is None or not p.exists():
        return {"path": str(path), "exists": False, "size_bytes": None, "sha256": None}
    return {"path": str(p), "exists": True, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)}


def package_versions() -> dict[str, str | None]:
    """Installed versions via importlib.metadata (no imports: keeps the service process free of TF/torch)."""
    from importlib import metadata
    dists = {"fetalvision": "fetalvision", "numpy": "numpy", "onnxruntime": "onnxruntime", "onnxruntime-gpu": "onnxruntime-gpu",
             "onnx": "onnx", "cv2": "opencv-python-headless", "torch": "torch", "tensorflow": "tensorflow", "keras": "keras",
             "ultralytics": "ultralytics", "segmentation_models_pytorch": "segmentation_models_pytorch", "fastapi": "fastapi", "uvicorn": "uvicorn"}
    out: dict[str, str | None] = {}
    for key, dist in dists.items():
        try:
            out[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[key] = None
    if out["cv2"] is None:
        try:
            out["cv2"] = metadata.version("opencv-python")
        except metadata.PackageNotFoundError:
            pass
    return out


def hardware_description() -> dict[str, Any]:
    """Hostname-free hardware description: CPU model, cores, RAM, GPU (if any), OS, Python."""
    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_model": None, "cpu_cores_logical": os.cpu_count(), "ram_gb": None, "gpu": None,
    }
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
    except Exception:
        pass
    try:
        if platform.system() == "Darwin":
            info["cpu_model"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        elif platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    if info["cpu_model"] is None:
        info["cpu_model"] = platform.processor() or None
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                      stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        if out:
            info["gpu"] = out.splitlines()[0]
    except Exception:
        pass
    return info


def jsonable(o: Any) -> Any:
    import numpy as np
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return o


def write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(jsonable(payload), indent=2))
    return p


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

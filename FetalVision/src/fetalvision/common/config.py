"""YAML config loading with ${ENV_VAR} expansion and a few helpers."""
from __future__ import annotations
import os, re, json, subprocess, datetime as _dt
from pathlib import Path
from typing import Any
import yaml

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m):
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")
        return _ENV.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def data_root() -> Path:
    return Path(os.environ.get("FETALVISION_DATA", "./data")).expanduser().resolve()


def load_config(path: str | Path) -> dict:
    os.environ.setdefault("FETALVISION_DATA", str(data_root()))
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _expand(cfg)
    cfg.setdefault("seed", 42)
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=Path(__file__).resolve().parents[3]).decode().strip()
    except Exception:  # not a git repo (e.g. Colab copy)
        return "unknown"


def results_dir(phase: str, run_name: str, root: str | Path | None = None) -> Path:
    root = Path(root) if root else Path(os.environ.get("FETALVISION_RESULTS", "results"))
    d = root / phase / run_name
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def write_metrics(out_dir: Path, run_name: str, phase: str, split: str, input_resolution, n_images: int,
                  seed: int, metrics: dict, extra: dict | None = None) -> Path:
    payload = {
        "run_name": run_name, "phase": phase, "split_evaluated": split,
        "input_resolution": list(input_resolution) if input_resolution is not None else None,
        "n_images": int(n_images), "seed": int(seed), "git_commit": git_commit(),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "metrics": _jsonable(metrics),
    }
    if extra:
        payload.update(_jsonable(extra))
    out = Path(out_dir) / "metrics.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def save_resolved_config(cfg: dict, out_dir: Path) -> None:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    (Path(out_dir) / "config_resolved.yaml").write_text(yaml.safe_dump(clean, sort_keys=False))


def _jsonable(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return o

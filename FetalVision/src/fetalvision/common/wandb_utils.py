"""Weights & Biases helpers. Offline by default when no API key is present so runs can be synced later."""
from __future__ import annotations
import os, logging
from .constants import WANDB_PROJECT

log = logging.getLogger(__name__)


def init_run(cfg: dict, phase: str, job_type: str = "train", tags: list[str] | None = None):
    try:
        import wandb
    except ImportError:
        log.warning("wandb not installed; experiment tracking disabled")
        return None
    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") is None:
        os.environ["WANDB_MODE"] = "offline"
        log.info("No WANDB_API_KEY: logging offline (sync later with `wandb sync`)")
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return wandb.init(project=os.environ.get("WANDB_PROJECT", WANDB_PROJECT), entity=os.environ.get("WANDB_ENTITY"),
                      group=phase, job_type=job_type, name=cfg.get("run_name"), config=clean,
                      tags=tags or [phase], reinit=True)


def log_metrics(run, metrics: dict, step: int | None = None):
    if run is None:
        return
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            flat[k] = v
    run.log(flat, step=step)


def log_figure(run, name: str, path):
    if run is None:
        return
    import wandb
    run.log({name: wandb.Image(str(path))})


def finish(run, summary: dict | None = None):
    if run is None:
        return
    if summary:
        for k, v in summary.items():
            if isinstance(v, (int, float, str)):
                run.summary[k] = v
    run.finish()

"""CLI: run the packaged pipeline on an image, a folder of images or an .mp4 and write annotated outputs + results.jsonl.

    python -m fetalvision.pipeline.run --config configs/service.yaml --input <img|folder|video.mp4> --out <dir> [--pixel-size 0.069] [--force-head]
"""
from __future__ import annotations
import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np
import cv2

from fetalvision.common.config import load_config
from fetalvision.common.seed import set_seed
from .pipeline import FetalVisionPipeline, strip_arrays
from .utils import setup_logging, write_json

log = logging.getLogger(__name__)
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def iter_images(inp: Path):
    if inp.is_dir():
        for p in sorted(inp.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                im = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if im is not None:
                    yield p.name, im
    elif inp.suffix.lower() in IMG_EXT:
        yield inp.name, cv2.imread(str(inp), cv2.IMREAD_COLOR)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/service.yaml")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pixel-size", type=float, default=None, help="pixel size in mm (HC in mm)")
    ap.add_argument("--force-head", action="store_true")
    ap.add_argument("--no-annotate", action="store_true")
    a = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(a.config)
    set_seed(int(cfg.get("seed", 42)))
    pipe = FetalVisionPipeline.from_config(cfg)
    pipe.warmup()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    inp = Path(a.input)
    n = 0
    with open(out / "results.jsonl", "w") as fj:
        if inp.suffix.lower() in VID_EXT:
            cap = cv2.VideoCapture(str(inp))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(out / f"{inp.stem}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                res = pipe.run(frame, a.pixel_size, a.force_head, return_mask=True)
                writer.write(pipe.annotate(frame, res))
                rec = strip_arrays(res); rec["frame"] = n
                fj.write(json.dumps(rec) + "\n"); n += 1
            cap.release(); writer.release()
        else:
            for name, im in iter_images(inp):
                res = pipe.run(im, a.pixel_size, a.force_head, return_mask=True)
                if not a.no_annotate:
                    cv2.imwrite(str(out / f"{Path(name).stem}_annotated.png"), pipe.annotate(im, res))
                rec = strip_arrays(res); rec["image"] = name
                fj.write(json.dumps(rec) + "\n"); n += 1
                log.info("%s -> %s p=%.3f %s %.1f ms", name, res["plane"]["label"], res["plane"]["probability"],
                         res["head_measurement"], res["timings_ms"]["total"])
    write_json(out / "manifest.json", pipe.manifest())
    log.info("processed %d frames/images -> %s", n, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

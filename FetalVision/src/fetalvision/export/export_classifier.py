"""Export the Keras 3 plane classifier to ONNX (opset 17, NHWC float32 [0,255], input 'image', output 'probabilities').

    python -m fetalvision.export.export_classifier --keras models/x.keras --out models/plane_classifier.onnx
    python -m fetalvision.export.export_classifier --config configs/export.yaml   # resolves the selected run / stand-in

Route 1: Keras 3 `model.export(path, format="onnx")` (tf2onnx under the hood). Route 2 (fallback): SavedModel export +
`python -m tf2onnx.convert`. Also writes `<out>.json` with class names, input size, preprocessing note, temperature
(from results/phase1/<run>/calibration.json when present, else 1.0), source path and sha256.
"""
from __future__ import annotations
import argparse, json, logging, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

from fetalvision.common.constants import CLASS_NAMES
from fetalvision.common.config import load_config
from fetalvision.pipeline.utils import resolve_path, sha256_file, load_json_if_exists, write_json, setup_logging
from .onnx_utils import rename_io, onnx_io_summary

log = logging.getLogger(__name__)
INPUT_NAME, OUTPUT_NAME = "image", "probabilities"


def resolve_keras_source(cfg_cls: dict) -> tuple[Path | None, str | None, bool]:
    """Return (keras_path, run_name, is_stand_in) following model_selection.json -> models/<run>.keras -> stand-in."""
    sel = load_json_if_exists(cfg_cls.get("model_selection_json"), "model_selection.json")
    run = None
    if sel:
        run = sel.get("selected_run") or sel.get("run_name") or sel.get("best_run")
    if run:
        p = resolve_path(cfg_cls["keras_template"].format(run=run))
        if p.exists():
            return p, run, False
        log.warning("selected run %s but %s does not exist", run, p)
    si = resolve_path(cfg_cls.get("stand_in_keras"))
    if si and si.exists():
        log.warning("using STAND-IN Keras classifier %s", si)
        return si, run, True
    return None, run, True


def resolve_temperature(cfg_cls: dict, run: str | None, override: float | None = None) -> tuple[float, str]:
    if override is not None:
        return float(override), "cli"
    if run:
        cal = load_json_if_exists(cfg_cls.get("calibration_json_template", "").format(run=run), "calibration.json")
        if cal and "temperature" in cal:
            return float(cal["temperature"]), "calibration.json"
    return 1.0, "default"


def export_keras_to_onnx(keras_path: str | Path, out_onnx: str | Path, input_size=(224, 224), opset: int = 17) -> dict:
    """Keras 3 -> ONNX. Returns a dict with route, io summary and a self-parity number on 4 random images."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import keras, tensorflow as tf
    out_onnx = Path(out_onnx)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    model = keras.models.load_model(str(keras_path), compile=False)
    h, w = int(input_size[0]), int(input_size[1])
    x = np.random.RandomState(0).uniform(0, 255, (4, h, w, 3)).astype(np.float32)
    ref = np.asarray(model(x, training=False))  # the model must be traced once before export
    route = "keras.Model.export(format='onnx')"
    try:
        model.export(str(out_onnx), format="onnx", opset_version=opset,
                     input_signature=[tf.TensorSpec([None, h, w, 3], tf.float32, name=INPUT_NAME)])
    except Exception as e:
        log.warning("Keras native ONNX export failed (%s); falling back to SavedModel + tf2onnx", e)
        route = "SavedModel + tf2onnx.convert"
        with tempfile.TemporaryDirectory() as td:
            sm = Path(td) / "saved_model"
            model.export(str(sm), format="tf_saved_model",
                         input_signature=[tf.TensorSpec([None, h, w, 3], tf.float32, name=INPUT_NAME)])
            cmd = [sys.executable, "-m", "tf2onnx.convert", "--saved-model", str(sm), "--output", str(out_onnx),
                   "--opset", str(opset)]
            subprocess.run(cmd, check=True)
    # normalise I/O names
    import onnx
    m = onnx.load(str(out_onnx))
    in_map = {m.graph.input[0].name: INPUT_NAME} if m.graph.input[0].name != INPUT_NAME else {}
    out_map = {m.graph.output[0].name: OUTPUT_NAME} if m.graph.output[0].name != OUTPUT_NAME else {}
    rename_io(out_onnx, in_map, out_map)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
    got = sess.run(None, {INPUT_NAME: x})[0]
    self_parity = float(np.abs(got - ref).max())
    log.info("exported %s via %s; self-parity max|dp|=%.2e", out_onnx, route, self_parity)
    return {"route": route, "io": onnx_io_summary(out_onnx), "self_parity_max_abs_diff": self_parity,
            "n_outputs": int(ref.shape[1])}


def write_classifier_meta(out_meta: Path, keras_path: Path, out_onnx: Path, input_size, temperature: float,
                          temperature_source: str, stand_in: bool, run: str | None, export_info: dict, opset: int) -> dict:
    meta = {
        "model": "plane_classifier", "class_names": list(CLASS_NAMES), "input_size": [int(input_size[0]), int(input_size[1])],
        "input_layout": "NHWC", "input_name": INPUT_NAME, "output_name": OUTPUT_NAME,
        "preprocessing": "raw 0-255, model includes rescaling", "color_order": "RGB",
        "temperature": float(temperature), "temperature_source": temperature_source,
        "source_keras": str(keras_path), "source_run": run, "sha256": sha256_file(out_onnx),
        "source_sha256": sha256_file(keras_path), "opset": opset, "stand_in": bool(stand_in),
        "stand_in_kind": ("phase1_trained_weights (real Phase 1 classifier until the Phase 2 selected run exists)" if stand_in else None),
        "export_route": export_info.get("route"), "onnx_io": export_info.get("io"),
        "self_parity_max_abs_diff": export_info.get("self_parity_max_abs_diff"),
    }
    write_json(out_meta, meta)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/export.yaml")
    ap.add_argument("--keras", default=None, help="override the Keras model path")
    ap.add_argument("--out", default=None, help="override the ONNX output path")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--stand-in", action="store_true", help="flag the exported model as a stand-in in the metadata")
    a = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(a.config)
    c = cfg["classifier"]
    if a.keras:
        keras_path, run, stand_in = Path(a.keras), None, a.stand_in
    else:
        keras_path, run, stand_in = resolve_keras_source(c)
        stand_in = stand_in or a.stand_in
    if keras_path is None or not Path(keras_path).exists():
        log.error("no Keras classifier found (checked model_selection.json / stand-in); nothing exported")
        return 2
    out_onnx = Path(a.out) if a.out else resolve_path(c["out_onnx"])
    out_meta = out_onnx.with_suffix(".json") if a.out else resolve_path(c["out_meta"])
    opset = int(cfg.get("opset", 17))
    info = export_keras_to_onnx(keras_path, out_onnx, c.get("input_size", [224, 224]), opset)
    if info["n_outputs"] != len(CLASS_NAMES):
        log.warning("model has %d outputs but CLASS_NAMES has %d", info["n_outputs"], len(CLASS_NAMES))
    temperature, t_src = resolve_temperature(c, run, a.temperature)
    meta = write_classifier_meta(out_meta, Path(keras_path), out_onnx, c.get("input_size", [224, 224]), temperature,
                                 t_src, stand_in, run, info, opset)
    log.info("metadata written to %s (temperature=%.4f from %s, stand_in=%s)", out_meta, temperature, t_src, stand_in)
    print(json.dumps({k: meta[k] for k in ["source_keras", "temperature", "sha256", "stand_in", "export_route"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

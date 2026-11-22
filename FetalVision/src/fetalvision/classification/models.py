"""Phase 1 model builders (Keras 3 / TensorFlow backend).

* build_scratch_cnn      – the original Phase 1 from-scratch CNN, layer for layer (110,276 parameters).
* build_mobilenetv3s     – MobileNetV3Small (ImageNet, include_preprocessing=True) + GAP + Dropout(0.3) + Dense(4).
  The backbone is always CALLED with training=False, so BatchNormalization layers use their moving statistics in
  every mode (the original Phase 1 did this for the frozen/partial stages; we keep it for the full fine-tuning run
  as well because batch 32 on a domain-shifted dataset gives noisy BN batch statistics and would otherwise destroy the
  pretrained features at the start of stage 2 – the standard Keras transfer-learning recommendation).
* set_finetune_mode      – stage-2 trainable flags:
      frozen  : backbone frozen (stage 1 only)
      partial : last 30 backbone layers trainable, every BatchNormalization layer frozen (exactly as Phase 1)
      full    : ALL backbone layers trainable (BN gamma/beta included; moving statistics still frozen, see above)
"""
from __future__ import annotations

from fetalvision.common.constants import CLASS_NAMES

INPUT_SHAPE = (224, 224, 3)
MODES = ("frozen", "partial", "full")


def build_scratch_cnn(input_shape=INPUT_SHAPE, n_classes: int = len(CLASS_NAMES), dropout: float = 0.3):
    from keras import layers, models
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(n_classes, activation="softmax"),
    ], name="cnn_scratch")


def build_mobilenetv3s(mode: str = "frozen", input_shape=INPUT_SHAPE, n_classes: int = len(CLASS_NAMES),
                       dropout: float = 0.3, weights: str | None = "imagenet", partial_last_n: int = 30):
    """Returns (model, base_model). The model is built in the frozen (stage-1) state; call set_finetune_mode()
    for stage 2. `mode` is recorded on the model (model.fv_mode) and validated here."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    from keras import layers, Model
    from keras.applications import MobileNetV3Small
    base = MobileNetV3Small(input_shape=input_shape, include_top=False, weights=weights, include_preprocessing=True)
    base.trainable = False
    inputs = layers.Input(shape=input_shape, name="image")
    x = base(inputs, training=False)  # BN in inference mode in ALL stages (see module docstring)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="probs")(x)
    model = Model(inputs, outputs, name=f"mobilenetv3s_{mode}")
    model.fv_mode = mode
    model.fv_partial_last_n = partial_last_n
    return model, base


def set_finetune_mode(base, mode: str, partial_last_n: int = 30) -> int:
    """Set stage-2 trainable flags on the backbone; returns the number of trainable backbone layers."""
    from keras import layers
    if mode not in MODES:
        raise ValueError(mode)
    # Keras only propagates `model.trainable = x` to sub-layers when the model-level flag changes, so every layer flag is
    # set explicitly here to make the result independent of the previous state.
    base.trainable = mode != "frozen"
    for layer in base.layers:
        layer.trainable = mode != "frozen"
    if mode == "partial":
        for layer in base.layers[:-partial_last_n]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    return sum(int(l.trainable) for l in base.layers)


def count_params(model) -> dict:
    import numpy as np
    trainable = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    non_trainable = int(sum(int(np.prod(w.shape)) for w in model.non_trainable_weights))
    return {"trainable": trainable, "non_trainable": non_trainable, "total": trainable + non_trainable}

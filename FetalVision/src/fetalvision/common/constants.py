"""Project-wide constants. Import from here; never retype."""
CLASS_NAMES = ["Abdomen", "Brain", "Femur", "Thorax"]  # alphabetical == image_dataset_from_directory order
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
BRAIN_IDX = CLASS_TO_IDX["Brain"]
DEFAULT_SEED = 42
WANDB_PROJECT = "FetalVision"
# FETAL_PLANES_DB raw label -> project class
PLANES_LABEL_MAP = {
    "fetal brain": "Brain", "fetal abdomen": "Abdomen", "fetal femur": "Femur", "fetal thorax": "Thorax",
    "brain": "Brain", "abdomen": "Abdomen", "femur": "Femur", "thorax": "Thorax",
}
HEAD_CLASS_NAMES = ["fetal_head"]

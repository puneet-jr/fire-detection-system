import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
ARTIFACT_DIR = ROOT_DIR / "src" / "fire_ai" / "artifacts"


def resolve_dataset_root(candidate: str | Path | None = None) -> Path:
    """Pick a usable dataset root with sensible fallbacks.

    Priority order:
    1) Explicit argument
    2) FIRE_DATASET_ROOT environment variable
    3) Original Windows path used during development
    4) Repo-local `external_datasets/` or `data/`
    """

    env_root = os.getenv("FIRE_DATASET_ROOT")
    fallbacks = [
        candidate,
        env_root,
        ROOT_DIR / "external_datasets",
        ROOT_DIR / "data",
    ]

    for path in fallbacks:
        if not path:
            continue
        resolved = Path(path).expanduser()
        if resolved.exists():
            return resolved

    # If nothing exists yet, default to repo data directory; caller will create or raise.
    return Path(candidate or env_root or ROOT_DIR / "data")


DEFAULT_EXTERNAL_DATASET_ROOT = resolve_dataset_root()

RAW_DATASET_PATH = DATA_DIR / "fire_sensor_dataset.csv"
MODEL_BUNDLE_PATH = MODEL_DIR / "hybrid_fire_model.joblib"
METRICS_PATH = MODEL_DIR / "training_metrics.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"

LABEL_MAP = {
    0: "NORMAL",
    1: "WARNING",
    2: "FIRE",
}

FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "flame_signal",
    "temp_rate",
    "humidity_rate",
    "temp_zscore",
    "humidity_zscore",
    "rolling_temp",
    "rolling_humidity",
    "dryness_index",
    "flame_persistence",
    "rule_score",
]

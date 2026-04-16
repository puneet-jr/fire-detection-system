from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .config import DEFAULT_EXTERNAL_DATASET_ROOT, RAW_DATASET_PATH, resolve_dataset_root

@dataclass
class SequenceConfig:
    warm_length: int = 30
    transition_length: int = 18
    fire_length: int = 12


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _normal_step(rng: np.random.Generator) -> tuple[float, float, int, int]:
    temp = rng.normal(30.0, 1.8)
    humidity = rng.normal(58.0, 5.0)
    flame = int(rng.random() < 0.02)
    label = 0
    return _clip(temp, 22, 40), _clip(humidity, 25, 85), flame, label


def _warning_step(
    rng: np.random.Generator,
    progress: float,
    base_temp: float,
    base_humidity: float,
) -> tuple[float, float, int, int]:
    temp = base_temp + 4.0 + progress * rng.uniform(6.0, 12.0) + rng.normal(0, 1.2)
    humidity = base_humidity - 6.0 - progress * rng.uniform(8.0, 18.0) + rng.normal(0, 2.0)
    flame = int(rng.random() < (0.08 + 0.10 * progress))
    label = 1
    return _clip(temp, 28, 58), _clip(humidity, 12, 70), flame, label


def _fire_step(
    rng: np.random.Generator,
    progress: float,
    base_temp: float,
    base_humidity: float,
) -> tuple[float, float, int, int]:
    temp = base_temp + 12.0 + progress * rng.uniform(10.0, 22.0) + rng.normal(0, 1.5)
    humidity = base_humidity - 15.0 - progress * rng.uniform(10.0, 25.0) + rng.normal(0, 2.5)
    flame = int(rng.random() < (0.72 + 0.22 * progress))
    label = 2
    return _clip(temp, 35, 80), _clip(humidity, 5, 55), flame, label


def generate_synthetic_dataset(
    num_sequences: int = 240,
    seed: int = 42,
    config: SequenceConfig | None = None,
) -> pd.DataFrame:
    """Generate a sensor dataset aligned to temperature, humidity, and flame hardware."""
    cfg = config or SequenceConfig()
    rng = np.random.default_rng(seed)
    rows: List[dict] = []

    for sequence_id in range(num_sequences):
        scenario = rng.choice(
            ["normal", "warning_then_fire", "warning_only", "noisy_spike"],
            p=[0.42, 0.28, 0.20, 0.10],
        )
        base_temp = float(rng.normal(29.5, 2.2))
        base_humidity = float(rng.normal(60.0, 6.0))

        total_steps = cfg.warm_length + cfg.transition_length + cfg.fire_length
        for step in range(total_steps):
            if scenario == "normal":
                temp, humidity, flame, label = _normal_step(rng)
            elif scenario == "warning_only":
                if step < cfg.warm_length:
                    temp, humidity, flame, label = _normal_step(rng)
                else:
                    progress = (step - cfg.warm_length) / max(cfg.transition_length - 1, 1)
                    temp, humidity, flame, label = _warning_step(rng, progress, base_temp, base_humidity)
            elif scenario == "warning_then_fire":
                if step < cfg.warm_length:
                    temp, humidity, flame, label = _normal_step(rng)
                elif step < cfg.warm_length + cfg.transition_length:
                    progress = (step - cfg.warm_length) / max(cfg.transition_length - 1, 1)
                    temp, humidity, flame, label = _warning_step(rng, progress, base_temp, base_humidity)
                else:
                    progress = (step - cfg.warm_length - cfg.transition_length) / max(cfg.fire_length - 1, 1)
                    temp, humidity, flame, label = _fire_step(rng, progress, base_temp, base_humidity)
            else:
                temp, humidity, flame, label = _normal_step(rng)
                if step > cfg.warm_length // 2 and rng.random() < 0.18:
                    temp = _clip(temp + rng.uniform(2.5, 6.0), 20, 45)
                    humidity = _clip(humidity - rng.uniform(3.0, 8.0), 20, 80)
                    label = 1 if temp > 35 or humidity < 35 else 0

            rows.append(
                {
                    "sequence_id": sequence_id,
                    "step": step,
                    "temperature": round(temp, 3),
                    "humidity": round(humidity, 3),
                    "flame_signal": flame,
                    "label": label,
                    "scenario": scenario,
                }
            )

    return pd.DataFrame(rows)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = {}
    for col in df.columns:
        key = str(col).strip().lower().replace("#", "").replace("%", "")
        key = " ".join(key.split())
        normalized[col] = key
    return df.rename(columns=normalized)


def _load_sensor_log(path: Path, start_sequence_id: int) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(path))

    humidity_col = next((c for c in df.columns if "humidity" in c), None)
    temperature_col = next((c for c in df.columns if "temperature" in c), None)
    detector_col = next((c for c in df.columns if c == "detector"), None)

    if humidity_col is None or temperature_col is None or "status" not in df.columns:
        raise ValueError(f"Missing expected sensor columns in {path}")

    temp = pd.to_numeric(df[temperature_col], errors="coerce")
    humidity = pd.to_numeric(df[humidity_col], errors="coerce")
    status = pd.to_numeric(df["status"], errors="coerce").fillna(0).astype(int).clip(0, 2)

    if detector_col is not None:
        detector = df[detector_col].astype(str).str.strip().str.upper().eq("ON").astype(int)
    else:
        detector = (status == 2).astype(int)

    flame_signal = np.where((status == 2) | (detector == 1), 1, 0)

    cleaned = pd.DataFrame(
        {
            "sequence_id": start_sequence_id,
            "step": np.arange(len(df)),
            "temperature": temp,
            "humidity": humidity,
            "flame_signal": flame_signal.astype(int),
            "label": status.astype(int),
            "scenario": path.stem,
            "source": "sensor_log",
        }
    )
    cleaned = cleaned.dropna(subset=["temperature", "humidity"]).reset_index(drop=True)
    return cleaned


def load_sensor_log_dataset(dataset_root: Path | str | None = None) -> pd.DataFrame:
    root = resolve_dataset_root(dataset_root)
    sensor_files = []
    for prefix in ("carton_", "clothing_", "electrical_"):
        sensor_files.extend(sorted(root.glob(f"{prefix}*.csv")))

    rows = []
    sequence_id = 10000
    for path in sensor_files:
        rows.append(_load_sensor_log(path, sequence_id))
        sequence_id += 1

    if not rows:
        warnings.warn(f"No sensor log CSV files found in {root}")
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def load_algerian_dataset(dataset_root: Path | str | None = None) -> pd.DataFrame:
    root = resolve_dataset_root(dataset_root)
    path = root / "Algerian_forest_fires_dataset_CLEANED.csv"
    if not path.exists():
        warnings.warn(f"Algerian dataset not found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    label = (
        df["classes"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"notfire": 0, "fire": 2, "not fire": 0})
        .fillna(0)
        .astype(int)
    )
    flame_signal = (label == 2).astype(int)
    cleaned = pd.DataFrame(
        {
            "sequence_id": 20000,
            "step": np.arange(len(df)),
            "temperature": pd.to_numeric(df["temperature"], errors="coerce"),
            "humidity": pd.to_numeric(df["rh"], errors="coerce"),
            "flame_signal": flame_signal,
            "label": label,
            "scenario": "algerian_forest_fire",
            "source": "algerian_dataset",
        }
    )
    return cleaned.dropna(subset=["temperature", "humidity"]).reset_index(drop=True)


def load_forestfires_dataset(dataset_root: Path | str | None = None) -> pd.DataFrame:
    root = resolve_dataset_root(dataset_root)
    path = root / "forestfires.csv"
    if not path.exists():
        warnings.warn(f"Forest fires dataset not found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)

    temp = pd.to_numeric(df["temp"], errors="coerce")
    humidity = pd.to_numeric(df["RH"], errors="coerce")
    area = pd.to_numeric(df["area"], errors="coerce").fillna(0.0)
    isi = pd.to_numeric(df["ISI"], errors="coerce").fillna(0.0)
    fwi_like = pd.to_numeric(df["FFMC"], errors="coerce").fillna(0.0)

    warning_mask = (temp > 30) | (humidity < 35) | (isi > 10) | (fwi_like > 92)
    fire_mask = area > 0
    label = np.where(fire_mask, 2, np.where(warning_mask, 1, 0))

    flame_signal = np.where(area > 1.0, 1, 0)
    cleaned = pd.DataFrame(
        {
            "sequence_id": 30000,
            "step": np.arange(len(df)),
            "temperature": temp,
            "humidity": humidity,
            "flame_signal": flame_signal.astype(int),
            "label": label.astype(int),
            "scenario": "forestfires",
            "source": "uci_forestfires",
        }
    )
    return cleaned.dropna(subset=["temperature", "humidity"]).reset_index(drop=True)


def build_unified_dataset(
    dataset_root: Path | str | None = None,
    include_synthetic: bool = True,
    synthetic_sequences: int = 160,
    seed: int = 42,
) -> pd.DataFrame:
    resolved_root = resolve_dataset_root(dataset_root)
    frames = [
        load_sensor_log_dataset(resolved_root),
        load_algerian_dataset(resolved_root),
        load_forestfires_dataset(resolved_root),
    ]
    frames = [f for f in frames if not f.empty]
    if include_synthetic:
        synthetic = generate_synthetic_dataset(num_sequences=synthetic_sequences, seed=seed).copy()
        synthetic["source"] = "synthetic"
        frames.append(synthetic)

    if not frames:
        raise FileNotFoundError(
            f"No data sources found in {resolved_root} and synthetic generation "
            "is disabled. Provide external datasets or enable synthetic data."
        )

    df = pd.concat(frames, ignore_index=True)
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
    df["flame_signal"] = pd.to_numeric(df["flame_signal"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).clip(0, 2)
    df = df.dropna(subset=["temperature", "humidity"]).reset_index(drop=True)
    return df


def save_unified_dataset(
    dataset_root: Path | str | None = None,
    output_path: Path | str | None = None,
    include_synthetic: bool = True,
    synthetic_sequences: int = 160,
    seed: int = 42,
) -> Path:
    target = Path(output_path or RAW_DATASET_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    df = build_unified_dataset(
        dataset_root=dataset_root,
        include_synthetic=include_synthetic,
        synthetic_sequences=synthetic_sequences,
        seed=seed,
    )
    df.to_csv(target, index=False)
    return target

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS


def _safe_std(values: Iterable[float], floor: float) -> float:
    std = float(np.std(list(values)))
    return max(std, floor)


@dataclass
class FeatureBuilder:
    window_size: int = 12
    temp_history: Deque[float] = field(default_factory=lambda: deque(maxlen=12))
    humidity_history: Deque[float] = field(default_factory=lambda: deque(maxlen=12))
    flame_history: Deque[int] = field(default_factory=lambda: deque(maxlen=12))
    prev_temp: float | None = None
    prev_humidity: float | None = None

    def reset(self) -> None:
        self.temp_history.clear()
        self.humidity_history.clear()
        self.flame_history.clear()
        self.prev_temp = None
        self.prev_humidity = None

    def transform_reading(
        self,
        temperature: float,
        humidity: float,
        flame_signal: int,
    ) -> dict:
        baseline_t = list(self.temp_history) if self.temp_history else [temperature]
        baseline_h = list(self.humidity_history) if self.humidity_history else [humidity]
        baseline_f = list(self.flame_history) if self.flame_history else [flame_signal]

        temp_rate = 0.0 if self.prev_temp is None else temperature - self.prev_temp
        humidity_rate = 0.0 if self.prev_humidity is None else humidity - self.prev_humidity

        temp_mean = float(np.mean(baseline_t))
        humidity_mean = float(np.mean(baseline_h))
        temp_std = _safe_std(baseline_t, floor=1.0)
        humidity_std = _safe_std(baseline_h, floor=2.0)

        temp_zscore = (temperature - temp_mean) / temp_std
        humidity_zscore = (humidity - humidity_mean) / humidity_std

        future_t = baseline_t + [temperature]
        future_h = baseline_h + [humidity]
        future_f = baseline_f + [flame_signal]

        rolling_temp = float(np.mean(future_t[-self.window_size :]))
        rolling_humidity = float(np.mean(future_h[-self.window_size :]))
        flame_persistence = float(np.mean(future_f[-self.window_size :]))
        dryness_index = temperature * (100.0 - humidity) / 100.0

        temp_anomaly = max(temp_zscore, 0.0) / 4.0
        humidity_drop = max(-humidity_zscore, 0.0) / 4.0
        temp_rate_norm = max(temp_rate, 0.0) / 10.0

        rule_score = float(
            np.clip(
                0.45 * flame_signal + 0.25 * temp_anomaly + 0.20 * humidity_drop + 0.10 * temp_rate_norm,
                0.0,
                1.0,
            )
        )

        self.temp_history.append(temperature)
        self.humidity_history.append(humidity)
        self.flame_history.append(int(flame_signal))
        self.prev_temp = temperature
        self.prev_humidity = humidity

        return {
            "temperature": float(temperature),
            "humidity": float(humidity),
            "flame_signal": int(flame_signal),
            "temp_rate": float(temp_rate),
            "humidity_rate": float(humidity_rate),
            "temp_zscore": float(temp_zscore),
            "humidity_zscore": float(humidity_zscore),
            "rolling_temp": float(rolling_temp),
            "rolling_humidity": float(rolling_humidity),
            "dryness_index": float(dryness_index),
            "flame_persistence": float(flame_persistence),
            "rule_score": rule_score,
        }


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    builder = FeatureBuilder()
    feature_rows = []
    current_sequence = None

    for row in df.itertuples(index=False):
        sequence_id = getattr(row, "sequence_id", None)
        if sequence_id != current_sequence:
            builder.reset()
            current_sequence = sequence_id

        feature_row = builder.transform_reading(
            temperature=float(row.temperature),
            humidity=float(row.humidity),
            flame_signal=int(row.flame_signal),
        )
        feature_rows.append(feature_row)

    feature_df = pd.DataFrame(feature_rows)[FEATURE_COLUMNS]
    metadata_df = df.drop(columns=[col for col in FEATURE_COLUMNS if col in df.columns], errors="ignore")
    return pd.concat([metadata_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

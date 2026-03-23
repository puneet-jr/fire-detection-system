from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, LABEL_MAP, MODEL_BUNDLE_PATH
from .features import FeatureBuilder


@dataclass
class HybridFireDetectionEngine:
    bundle_path: str | None = None

    def __post_init__(self) -> None:
        bundle = joblib.load(self.bundle_path or MODEL_BUNDLE_PATH)
        self.model = bundle["model"]
        self.anomaly_model = bundle["anomaly_model"]
        self.feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
        self.label_map = bundle.get("label_map", LABEL_MAP)
        self.builder = FeatureBuilder()

    def reset(self) -> None:
        self.builder.reset()

    def predict(self, temperature: float, humidity: float, flame_signal: int) -> dict[str, Any]:
        features = self.builder.transform_reading(temperature, humidity, flame_signal)
        sample = pd.DataFrame([features])[self.feature_columns]

        pred_class = int(self.model.predict(sample)[0])
        proba = self.model.predict_proba(sample)[0]
        confidence = float(np.max(proba))

        anomaly_decision = float(self.anomaly_model.decision_function(sample)[0])
        raw_anomaly = bool(self.anomaly_model.predict(sample)[0] == -1)
        anomaly_flag = bool(anomaly_decision < -0.03 or (raw_anomaly and features["rule_score"] > 0.35))

        # Safety-first fusion: keep hardware flame detection and strong anomalies dominant.
        if flame_signal and pred_class == 0:
            pred_class = 1
        if anomaly_flag and pred_class == 0:
            pred_class = 1
        if flame_signal and features["rule_score"] > 0.65:
            pred_class = max(pred_class, 2)
        if temperature > 42.0 and features["temp_zscore"] > 3.0 and features["humidity_zscore"] < -2.0:
            pred_class = max(pred_class, 2)
        if not flame_signal and pred_class == 2 and temperature < 42.0 and features["rule_score"] < 0.75:
            pred_class = 1

        reasons = []
        if flame_signal:
            reasons.append("flame sensor active")
        if features["temp_rate"] > 2.0:
            reasons.append("rapid temperature rise")
        if features["humidity_zscore"] < -2.0:
            reasons.append("abnormal humidity drop")
        if anomaly_flag:
            reasons.append("anomaly detector triggered")
        if features["rule_score"] > 0.55:
            reasons.append("high fused rule score")

        return {
            "prediction_id": pred_class,
            "prediction": self.label_map[pred_class],
            "confidence": round(confidence, 4),
            "anomaly_score": round(anomaly_decision, 4),
            "anomaly_flag": anomaly_flag,
            "buzzer": bool(pred_class >= 2 or flame_signal),
            "reasons": reasons,
            "features": {key: round(float(value), 4) for key, value in features.items()},
        }

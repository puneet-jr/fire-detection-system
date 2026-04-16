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

        classifier_pred_class = int(self.model.predict(sample)[0])
        proba = self.model.predict_proba(sample)[0]
        confidence = float(np.max(proba))
        probability_breakdown = {
            self.label_map[index]: round(float(score), 4)
            for index, score in enumerate(proba)
        }

        anomaly_decision = float(self.anomaly_model.decision_function(sample)[0])
        raw_anomaly = bool(self.anomaly_model.predict(sample)[0] == -1)
        anomaly_flag = bool(anomaly_decision < -0.03 or (raw_anomaly and features["rule_score"] > 0.35))

        final_pred_class = classifier_pred_class
        overrides = []

        # Safety-first fusion: keep hardware flame detection and strong anomalies dominant.
        if flame_signal and final_pred_class == 0:
            final_pred_class = 1
            overrides.append("Flame sensor lifted the decision from NORMAL to WARNING.")
        if anomaly_flag and final_pred_class == 0:
            final_pred_class = 1
            overrides.append("Isolation Forest anomaly lifted the decision from NORMAL to WARNING.")
        if flame_signal and features["rule_score"] > 0.65:
            previous = final_pred_class
            final_pred_class = max(final_pred_class, 2)
            if final_pred_class != previous:
                overrides.append("High flame-backed rule score escalated the result to FIRE.")
        if temperature > 42.0 and features["temp_zscore"] > 3.0 and features["humidity_zscore"] < -2.0:
            previous = final_pred_class
            final_pred_class = max(final_pred_class, 2)
            if final_pred_class != previous:
                overrides.append("Combined thermal anomaly and humidity drop escalated the result to FIRE.")
        if not flame_signal and final_pred_class == 2 and temperature < 42.0 and features["rule_score"] < 0.75:
            final_pred_class = 1
            overrides.append("Weak fire evidence without flame softened the result back to WARNING.")

        temp_anomaly_strength = float(np.clip(max(features["temp_zscore"], 0.0) / 4.0, 0.0, 1.0))
        humidity_drop_strength = float(np.clip(max(-features["humidity_zscore"], 0.0) / 4.0, 0.0, 1.0))
        temp_rate_strength = float(np.clip(max(features["temp_rate"], 0.0) / 10.0, 0.0, 1.0))
        rule_components = [
            {
                "name": "Flame evidence",
                "weight": 0.45,
                "signal": round(float(flame_signal), 4),
                "contribution": round(float(0.45 * flame_signal), 4),
                "summary": "Direct binary support from the flame sensor.",
            },
            {
                "name": "Temperature anomaly",
                "weight": 0.25,
                "signal": round(temp_anomaly_strength, 4),
                "contribution": round(float(0.25 * temp_anomaly_strength), 4),
                "summary": "Positive temperature z-score converted into a bounded anomaly strength.",
            },
            {
                "name": "Humidity drop",
                "weight": 0.20,
                "signal": round(humidity_drop_strength, 4),
                "contribution": round(float(0.20 * humidity_drop_strength), 4),
                "summary": "Negative humidity z-score converted into drying-risk evidence.",
            },
            {
                "name": "Heat acceleration",
                "weight": 0.10,
                "signal": round(temp_rate_strength, 4),
                "contribution": round(float(0.10 * temp_rate_strength), 4),
                "summary": "Fast upward movement in temperature across consecutive readings.",
            },
        ]

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

        classifier_label = self.label_map[classifier_pred_class]
        final_label = self.label_map[final_pred_class]

        return {
            "prediction_id": final_pred_class,
            "prediction": final_label,
            "confidence": round(confidence, 4),
            "anomaly_score": round(anomaly_decision, 4),
            "anomaly_flag": anomaly_flag,
            "buzzer": bool(final_pred_class >= 2 or flame_signal),
            "reasons": reasons,
            "features": {key: round(float(value), 4) for key, value in features.items()},
            "classifier": {
                "prediction_id": classifier_pred_class,
                "prediction": classifier_label,
                "confidence": round(confidence, 4),
                "probabilities": probability_breakdown,
            },
            "fusion": {
                "final_prediction_id": final_pred_class,
                "final_prediction": final_label,
                "safety_overrides": overrides,
            },
            "rule_breakdown": {
                "score": round(float(features["rule_score"]), 4),
                "components": rule_components,
            },
            "analysis": {
                "current_temperature": round(float(temperature), 4),
                "current_humidity": round(float(humidity), 4),
                "rolling_temperature": round(float(features["rolling_temp"]), 4),
                "rolling_humidity": round(float(features["rolling_humidity"]), 4),
                "dryness_index": round(float(features["dryness_index"]), 4),
                "flame_persistence": round(float(features["flame_persistence"]), 4),
            },
        }

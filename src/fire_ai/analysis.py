from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .config import FEATURE_IMPORTANCE_PATH, LABEL_MAP, METRICS_PATH, RAW_DATASET_PATH


FORMULA_LIBRARY = [
    {
        "name": "Temperature Rate",
        "formula": "temp_rate =\ncurrent_temperature - previous_temperature",
        "purpose": "Detects sudden heat rise between consecutive sensor readings.",
    },
    {
        "name": "Humidity Rate",
        "formula": "humidity_rate =\ncurrent_humidity - previous_humidity",
        "purpose": "Detects rapid moisture drop that often appears before a visible flame.",
    },
    {
        "name": "Temperature Z-Score",
        "formula": "temp_zscore =\n(temperature - mean(temp_history)) /\nmax(std(temp_history), 1.0)",
        "purpose": "Measures how abnormal the current temperature is versus the recent baseline.",
    },
    {
        "name": "Humidity Z-Score",
        "formula": "humidity_zscore =\n(humidity - mean(humidity_history)) /\nmax(std(humidity_history), 2.0)",
        "purpose": "Measures how abnormal the current humidity is versus the recent baseline.",
    },
    {
        "name": "Rolling Context",
        "formula": "rolling_temp = mean(last_12_temperatures)\nrolling_humidity = mean(last_12_humidity_values)",
        "purpose": "Keeps short-term history so the model sees trends, not only one raw reading.",
    },
    {
        "name": "Dryness Index",
        "formula": "dryness_index =\ntemperature * (100 - humidity) / 100",
        "purpose": "Combines hot and dry conditions into one fire-risk indicator.",
    },
    {
        "name": "Flame Persistence",
        "formula": "flame_persistence =\nmean(last_12_flame_signals)",
        "purpose": "Rewards repeated flame detections instead of trusting a single spike.",
    },
    {
        "name": "Hybrid Rule Score",
        "formula": "rule_score = clip(\n0.45*flame + 0.25*temp_anomaly +\n0.20*humidity_drop + 0.10*temp_rate_norm,\n0, 1\n)",
        "purpose": "Fuses sensor evidence into a safety-oriented score used with ML and anomaly logic.",
    },
]

FUSION_RULES = [
    "If the flame sensor is active and the classifier says NORMAL, the final output is raised to WARNING.",
    "If the anomaly detector triggers and the classifier says NORMAL, the final output is raised to WARNING.",
    "If flame is active and the fused rule score is above 0.65, the final output is raised to FIRE.",
    "If temperature is above 42 C with strong temperature anomaly and strong humidity drop, the final output is raised to FIRE.",
    "If the classifier says FIRE but there is no flame and the thermal evidence is still weak, the final output is softened to WARNING.",
]

PIPELINE_STAGES = [
    {
        "name": "Data Collection",
        "summary": "Temperature, humidity, and flame signals are collected from sensor logs plus two public forest-fire datasets.",
    },
    {
        "name": "Feature Engineering",
        "summary": "The system builds temporal, anomaly, persistence, and dryness features from the raw sensor stream.",
    },
    {
        "name": "Supervised Classification",
        "summary": "Random Forest and Extra Trees are trained for multiclass prediction, and the better macro-F1 model is saved.",
    },
    {
        "name": "Anomaly Detection",
        "summary": "Isolation Forest is trained only on NORMAL readings to flag unusual situations even when the classifier is cautious.",
    },
    {
        "name": "Safety Fusion",
        "summary": "Classifier output, anomaly score, flame sensor, and rule score are combined into the final NORMAL, WARNING, or FIRE decision.",
    },
]

DATASET_NOTES = [
    "Sensor log files are the closest match to the embedded hardware because they directly contain temperature, humidity, flame, and status values.",
    "The Algerian and UCI forest-fire datasets broaden environmental variation so the model does not learn only one narrow operating range.",
    "Synthetic sequences add controlled warning-to-fire transitions that make temporal learning easier to demonstrate during simulation.",
]

MODEL_NOTES = [
    {
        "name": "Random Forest / Extra Trees",
        "summary": "Two tree-based classifiers are trained and compared using macro F1 on the held-out test set.",
    },
    {
        "name": "Isolation Forest",
        "summary": "A separate anomaly detector learns only normal-state behavior and raises concern when a new reading looks isolated.",
    },
]

LIVE_SENSOR_RANGES = [
    {
        "label": "NORMAL",
        "temperature": "27.0 C to 32.5 C",
        "humidity": "52% to 66%",
        "flame": "Usually 0, with only about 2% random spikes",
        "summary": "Represents a safe room-like state with moderate temperature and healthy humidity.",
    },
    {
        "label": "WARNING",
        "temperature": "35.0 C to 43.0 C",
        "humidity": "28% to 42%",
        "flame": "Usually 0, but about 18% chance of a flame signal",
        "summary": "Represents heat build-up and drying conditions before a confirmed fire.",
    },
    {
        "label": "FIRE",
        "temperature": "48.0 C to 66.0 C",
        "humidity": "10% to 24%",
        "flame": "Usually 1, with about 88% chance of flame detection",
        "summary": "Represents strong fire-like conditions, which is why values like 56 C can appear here.",
    },
]

LIVE_FEED_NOTES = [
    "A new random reading is generated every 2.2 seconds while auto monitoring is running.",
    "The phase does not stay fixed. The generator can move between NORMAL, WARNING, and FIRE to simulate changing conditions.",
    "A reading such as 56 C is not a hard limit. It is simply one sampled value inside the FIRE temperature range of 48.0 C to 66.0 C.",
    "After the phase is chosen, a small random adjustment is applied and then values are clamped to safe simulation limits of 20 C to 80 C and 5% to 90% humidity.",
]

LIVE_FEED_TRANSITIONS = [
    "If the current state is NORMAL, it usually stays NORMAL, sometimes moves to WARNING, and rarely jumps directly to FIRE.",
    "If the current state is WARNING, it can stay WARNING, cool down to NORMAL, or escalate to FIRE.",
    "If the current state is FIRE, it may stay FIRE or de-escalate back toward WARNING or NORMAL in later readings.",
]


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def load_dataset_summary(dataset_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(dataset_path or RAW_DATASET_PATH)
    if not path.exists():
        return {
            "path": str(path),
            "available": False,
            "rows": 0,
            "source_breakdown": [],
            "class_balance": [],
            "scenario_examples": [],
        }

    df = pd.read_csv(path)
    source_counts = df["source"].value_counts().to_dict() if "source" in df.columns else {}
    label_counts = df["label"].value_counts().sort_index().to_dict() if "label" in df.columns else {}
    total_rows = int(len(df))

    return {
        "path": str(path),
        "available": True,
        "rows": total_rows,
        "sensor_columns": [str(col) for col in df.columns],
        "source_breakdown": [
            {
                "name": source.replace("_", " ").title(),
                "key": source,
                "count": int(count),
                "share": round((count / total_rows) * 100, 2) if total_rows else 0.0,
            }
            for source, count in source_counts.items()
        ],
        "class_balance": [
            {
                "label_id": int(label_id),
                "label": LABEL_MAP.get(int(label_id), str(label_id)),
                "count": int(count),
                "share": round((count / total_rows) * 100, 2) if total_rows else 0.0,
            }
            for label_id, count in label_counts.items()
        ],
        "scenario_examples": sorted(df["scenario"].dropna().astype(str).unique().tolist())[:8]
        if "scenario" in df.columns
        else [],
    }


@lru_cache(maxsize=1)
def load_model_summary() -> dict[str, Any]:
    metrics = _safe_read_json(METRICS_PATH)

    top_features = []
    if FEATURE_IMPORTANCE_PATH.exists():
        feature_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        top_features = [
            {
                "feature": str(row.feature),
                "importance": round(float(row.importance), 4),
            }
            for row in feature_df.head(8).itertuples(index=False)
        ]

    class_report = metrics.get("class_report", {})
    class_scores = []
    for label_name in LABEL_MAP.values():
        if label_name not in class_report:
            continue
        stats = class_report[label_name]
        class_scores.append(
            {
                "label": label_name,
                "precision": round(float(stats.get("precision", 0.0)), 4),
                "recall": round(float(stats.get("recall", 0.0)), 4),
                "f1": round(float(stats.get("f1-score", 0.0)), 4),
                "support": int(stats.get("support", 0)),
            }
        )

    return {
        "metrics_available": bool(metrics),
        "best_model": metrics.get("best_model", "unknown"),
        "accuracy": round(float(metrics.get("accuracy", 0.0)), 4),
        "macro_f1": round(float(metrics.get("macro_f1", 0.0)), 4),
        "rows": int(metrics.get("rows", 0)),
        "class_scores": class_scores,
        "top_features": top_features,
    }


@lru_cache(maxsize=1)
def build_project_analysis() -> dict[str, Any]:
    dataset_summary = load_dataset_summary()
    model_summary = load_model_summary()

    return {
        "title": "Hybrid AI Fire Detection Analysis",
        "subtitle": "A clear view of data collection, feature formulas, tree-based ML reasoning, anomaly logic, and final fire decisions.",
        "dataset": dataset_summary,
        "model": model_summary,
        "pipeline": PIPELINE_STAGES,
        "formulas": FORMULA_LIBRARY,
        "fusion_rules": FUSION_RULES,
        "dataset_notes": DATASET_NOTES,
        "model_notes": MODEL_NOTES,
        "live_ranges": LIVE_SENSOR_RANGES,
        "live_feed_notes": LIVE_FEED_NOTES,
        "live_feed_transitions": LIVE_FEED_TRANSITIONS,
        "input_story": [
            "Temperature gives the main heat signal.",
            "Humidity helps capture drying conditions and rapid moisture loss.",
            "Flame signal is treated as the highest-priority safety cue.",
        ],
    }

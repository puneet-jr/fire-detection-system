from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from .config import (
    DEFAULT_EXTERNAL_DATASET_ROOT,
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE_PATH,
    LABEL_MAP,
    METRICS_PATH,
    MODEL_BUNDLE_PATH,
    RAW_DATASET_PATH,
)
from .data import save_unified_dataset
from .features import build_feature_frame


def train_model(
    dataset_path: str | None = None,
    dataset_root: str | None = None,
    regenerate_dataset: bool = False,
    include_synthetic: bool = True,
    synthetic_sequences: int = 160,
) -> dict:
    active_dataset_path = dataset_path or RAW_DATASET_PATH
    if regenerate_dataset or not pd.io.common.file_exists(active_dataset_path):
        save_unified_dataset(
            dataset_root=dataset_root or DEFAULT_EXTERNAL_DATASET_ROOT,
            output_path=active_dataset_path,
            include_synthetic=include_synthetic,
            synthetic_sequences=synthetic_sequences,
        )

    raw_df = pd.read_csv(active_dataset_path)
    df = build_feature_frame(raw_df)

    X = df[FEATURE_COLUMNS]
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=220,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=260,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
        ),
    }

    best_name = None
    best_model = None
    best_f1 = -1.0

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = f1_score(y_test, pred, average="macro")
        if score > best_f1:
            best_f1 = score
            best_name = name
            best_model = model

    assert best_model is not None
    y_pred = best_model.predict(X_test)

    normal_train = X_train[y_train == 0]
    anomaly_model = IsolationForest(
        n_estimators=180,
        contamination=0.08,
        random_state=42,
    )
    anomaly_model.fit(normal_train)

    report = classification_report(
        y_test,
        y_pred,
        target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "best_model": best_name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "class_report": report,
        "rows": int(len(df)),
        "class_balance": y.value_counts().sort_index().to_dict(),
    }

    importances = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": best_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    MODEL_BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "anomaly_model": anomaly_model,
            "feature_columns": FEATURE_COLUMNS,
            "label_map": LABEL_MAP,
            "metrics": metrics,
        },
        MODEL_BUNDLE_PATH,
    )
    importances.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default=str(RAW_DATASET_PATH))
    parser.add_argument("--dataset-root", default=str(DEFAULT_EXTERNAL_DATASET_ROOT))
    parser.add_argument("--regenerate-dataset", action="store_true")
    parser.add_argument("--no-synthetic", action="store_true")
    parser.add_argument("--synthetic-sequences", type=int, default=160)
    args = parser.parse_args()

    metrics = train_model(
        dataset_path=args.dataset_path,
        dataset_root=args.dataset_root,
        regenerate_dataset=args.regenerate_dataset,
        include_synthetic=not args.no_synthetic,
        synthetic_sequences=args.synthetic_sequences,
    )
    print("Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

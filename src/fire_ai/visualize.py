from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from .config import ARTIFACT_DIR, RAW_DATASET_PATH
from .features import build_feature_frame


def create_plots(dataset_path: str | None = None) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(dataset_path or RAW_DATASET_PATH)
    df = build_feature_frame(raw_df)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df["temperature"],
        df["humidity"],
        c=df["label"],
        cmap="inferno",
        alpha=0.65,
    )
    plt.xlabel("Temperature (C)")
    plt.ylabel("Humidity (%)")
    plt.title("Temperature vs Humidity by Fire State")
    plt.colorbar(scatter, ticks=[0, 1, 2], label="Label")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "temp_humidity_scatter.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    for label, name in [(0, "NORMAL"), (1, "WARNING"), (2, "FIRE")]:
        subset = df[df["label"] == label]
        plt.hist(subset["rule_score"], bins=25, alpha=0.5, label=name)
    plt.xlabel("Rule Score")
    plt.ylabel("Count")
    plt.title("Hybrid Rule Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "rule_score_distribution.png", dpi=180)
    plt.close()

    print(f"Saved plots to: {ARTIFACT_DIR}")


def main() -> None:
    create_plots()


if __name__ == "__main__":
    main()

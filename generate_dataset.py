import argparse

from src.fire_ai.config import DEFAULT_EXTERNAL_DATASET_ROOT, RAW_DATASET_PATH
from src.fire_ai.data import save_unified_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(DEFAULT_EXTERNAL_DATASET_ROOT))
    parser.add_argument("--output", default=str(RAW_DATASET_PATH))
    parser.add_argument("--synthetic-sequences", type=int, default=160)
    parser.add_argument("--no-synthetic", action="store_true")
    args = parser.parse_args()

    output_path = save_unified_dataset(
        dataset_root=args.dataset_root,
        output_path=args.output,
        include_synthetic=not args.no_synthetic,
        synthetic_sequences=args.synthetic_sequences,
    )
    import pandas as pd

    df = pd.read_csv(output_path)
    print(f"Dataset created: {output_path}")
    print("Class balance:", df["label"].value_counts().sort_index().to_dict())
    print("Sources:", df["source"].value_counts().to_dict())


if __name__ == "__main__":
    main()

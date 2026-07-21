from pathlib import Path
import argparse

import pandas as pd


def validate_training_data(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    max_imbalance_ratio: float = 2.0,
) -> list[str]:
    issues: list[str] = []

    required = {text_column, label_column}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        issues.append(f"Missing required columns: {', '.join(missing)}")
        return issues

    texts = df[text_column].astype(str).str.strip()
    labels = df[label_column].astype(str).str.strip()

    if len(df) == 0:
        issues.append("Dataset is empty")
        return issues

    if (texts == "").any():
        issues.append("Found empty text rows")

    class_counts = labels.value_counts()
    if len(class_counts) < 2:
        issues.append("At least two classes are required")
    else:
        min_count = int(class_counts.min())
        max_count = int(class_counts.max())
        if min_count == 0:
            issues.append("At least one class has zero samples")
        else:
            ratio = max_count / min_count
            if ratio > max_imbalance_ratio:
                issues.append(
                    "Class imbalance too high: "
                    f"max/min={ratio:.2f} (threshold={max_imbalance_ratio:.2f})"
                )

    normalized = texts.str.lower().str.replace(r"\s+", " ", regex=True)
    duplicate_mask = normalized.duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count > 0:
        issues.append(
            f"Found {duplicate_count} duplicate text rows (case/space-insensitive)"
        )

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate dataset quality before model training"
    )
    parser.add_argument("--input-csv", default="data/sample_news.csv")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--max-imbalance-ratio", type=float, default=2.0)
    args = parser.parse_args()

    data_path = Path(args.input_csv)
    if not data_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {data_path}")

    df = pd.read_csv(data_path)
    issues = validate_training_data(
        df=df,
        text_column=args.text_column,
        label_column=args.label_column,
        max_imbalance_ratio=args.max_imbalance_ratio,
    )

    if issues:
        print("Data quality check failed:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print("Data quality check passed.")


if __name__ == "__main__":
    main()
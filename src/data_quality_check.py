"""Dataset quality checks used before training."""

from pathlib import Path
import argparse

import pandas as pd


def validate_training_data(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    max_imbalance_ratio: float = 2.0,
) -> list[str]:
    # Return all issues found so callers can show one consolidated error.
    issues: list[str] = []

    # 1) Schema check: ensure required columns exist before any other logic.
    required = {text_column, label_column}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        issues.append(f"Missing required columns: {', '.join(missing)}")
        return issues

    # Normalize raw values to strings and trim spaces for robust downstream checks.
    texts = df[text_column].astype(str).str.strip()
    labels = df[label_column].astype(str).str.strip()

    # 2) Empty dataset check: model training cannot proceed with zero rows.
    if len(df) == 0:
        issues.append("Dataset is empty")
        return issues

    # 3) Empty text check: blank rows add noise and can break quality expectations.
    if (texts == "").any():
        issues.append("Found empty text rows")

    # 4) Class coverage/balance checks: avoid one-class or heavily skewed training.
    # Example: counts [30, 10] => ratio 3.0, which fails when threshold is 2.0.
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

    # 5) Duplicate detection after normalization.
    # "Breaking News" and "breaking   news" are treated as duplicates.
    normalized = texts.str.lower().str.replace(r"\s+", " ", regex=True)
    # Duplicate detection ignores case and extra spaces to catch near-identical rows.
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
    # CLI wrapper around shared validator used by training.
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
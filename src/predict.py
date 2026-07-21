"""Prediction CLI for single text and batch CSV scoring."""

from pathlib import Path
import argparse
import math

import joblib
import pandas as pd

from preprocess import clean_batch, clean_text


def _softmax(values: list[float]) -> list[float]:
    # Stable softmax for turning decision-function margins into comparable scores.
    max_value = max(values)
    exps = [math.exp(v - max_value) for v in values]
    total = sum(exps)
    if total == 0:
        return [0.0 for _ in exps]
    return [value / total for value in exps]


def _rank_predictions_with_scores(
    model: object, cleaned_texts: list[str], top_k: int
) -> list[list[tuple[str, float]]]:
    classes = [str(item) for item in getattr(model, "classes_", [])]
    if not classes:
        raise ValueError("Loaded model does not expose class labels.")

    top_k = max(1, min(top_k, len(classes)))

    if hasattr(model, "predict_proba"):
        raw_scores = model.predict_proba(cleaned_texts)
        score_rows = [[float(value) for value in row] for row in raw_scores]
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(cleaned_texts)
        # LinearSVC returns margins; convert each row to normalized softmax scores.
        if hasattr(raw_scores, "ndim") and getattr(raw_scores, "ndim", 1) == 1:
            score_rows = [[-float(value), float(value)] for value in raw_scores]
        else:
            score_rows = [[float(value) for value in row] for row in raw_scores]
        score_rows = [_softmax(row) for row in score_rows]
    else:
        raise ValueError(
            "Model does not support score output (predict_proba or decision_function)."
        )

    ranked_rows: list[list[tuple[str, float]]] = []
    for row in score_rows:
        ranked = sorted(
            zip(classes, row),
            key=lambda item: item[1],
            reverse=True,
        )
        ranked_rows.append([(label, float(score)) for label, score in ranked[:top_k]])

    return ranked_rows


def main() -> None:
    # CLI flags support one-off prediction and dataset-level scoring.
    parser = argparse.ArgumentParser(description="Predict news topics from text or CSV")
    parser.add_argument("--text", help="Single text input")
    parser.add_argument("--input-csv", help="CSV file containing a text column")
    parser.add_argument("--text-column", default="text", help="Text column for CSV mode")
    parser.add_argument(
        "--output-csv",
        help="Optional output path for CSV predictions (default: <input>_predictions.csv)",
    )
    parser.add_argument(
        "--model",
        default="models/news_topic_model.joblib",
        help="Path to trained model",
    )
    parser.add_argument(
        "--show-scores",
        action="store_true",
        help="Show top-k label scores in output",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many ranked labels to show when --show-scores is enabled",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    # Exactly one input mode must be provided.
    if bool(args.text) == bool(args.input_csv):
        raise ValueError("Provide exactly one of --text or --input-csv")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run src/train.py first."
        )

    # Model artifact is a scikit-learn pipeline persisted with joblib.
    model = joblib.load(model_path)

    if args.text:
        # Single text mode: clean one input and return one label.
        cleaned = clean_text(args.text)
        pred = str(model.predict([cleaned])[0])
        print(f"Prediction: {pred}")

        if args.show_scores:
            ranked = _rank_predictions_with_scores(model, [cleaned], args.top_k)[0]
            print("Top scores:")
            for label, score in ranked:
                print(f"- {label}: {score:.4f}")
        return

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if args.text_column not in df.columns:
        raise ValueError(
            f"Column '{args.text_column}' not found in input CSV: {input_path}"
        )

    # Batch mode: clean all rows, score, and append predicted_topic column.
    cleaned_batch = clean_batch(df[args.text_column].astype(str).tolist())
    preds = model.predict(cleaned_batch)

    result_df = df.copy()
    result_df["predicted_topic"] = preds

    if args.show_scores:
        ranked_rows = _rank_predictions_with_scores(model, cleaned_batch, args.top_k)
        result_df["predicted_score"] = [row[0][1] for row in ranked_rows]
        result_df["top_k_rankings"] = [
            "; ".join(f"{label}:{score:.4f}" for label, score in row)
            for row in ranked_rows
        ]

    # If output path is omitted, create a sibling file named <input>_predictions.csv.
    output_path = (
        Path(args.output_csv)
        if args.output_csv
        else input_path.with_name(f"{input_path.stem}_predictions.csv")
    )
    result_df.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")
    print(f"Rows scored: {len(result_df)}")
    if args.show_scores:
        print(f"Included top-{args.top_k} scores in output CSV")


if __name__ == "__main__":
    main()

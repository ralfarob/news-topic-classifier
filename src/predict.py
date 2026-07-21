from pathlib import Path
import argparse

import joblib
import pandas as pd

from preprocess import clean_batch, clean_text


def main() -> None:
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
    args = parser.parse_args()

    if bool(args.text) == bool(args.input_csv):
        raise ValueError("Provide exactly one of --text or --input-csv")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run src/train.py first."
        )

    model = joblib.load(model_path)

    if args.text:
        cleaned = clean_text(args.text)
        pred = str(model.predict([cleaned])[0])
        print(f"Prediction: {pred}")
        return

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if args.text_column not in df.columns:
        raise ValueError(
            f"Column '{args.text_column}' not found in input CSV: {input_path}"
        )

    cleaned_batch = clean_batch(df[args.text_column].astype(str).tolist())
    preds = model.predict(cleaned_batch)

    result_df = df.copy()
    result_df["predicted_topic"] = preds

    output_path = (
        Path(args.output_csv)
        if args.output_csv
        else input_path.with_name(f"{input_path.stem}_predictions.csv")
    )
    result_df.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")
    print(f"Rows scored: {len(result_df)}")


if __name__ == "__main__":
    main()

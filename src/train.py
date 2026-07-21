from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data_quality_check import validate_training_data
from preprocess import clean_batch


def build_pipelines() -> dict[str, Pipeline]:
    return {
        "LogisticRegression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                (
                    "clf",
                    LogisticRegression(max_iter=2000, solver="liblinear", random_state=42),
                ),
            ]
        ),
        "LinearSVC": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("clf", LinearSVC()),
            ]
        ),
    }


def main() -> None:
    data_path = Path("data/sample_news.csv")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must include 'text' and 'label' columns")

    quality_issues = validate_training_data(df)
    if quality_issues:
        issue_list = "\n".join(f"- {issue}" for issue in quality_issues)
        raise ValueError(f"Dataset quality check failed:\n{issue_list}")

    x = clean_batch(df["text"].astype(str).tolist())
    y = df["label"].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    pipelines = build_pipelines()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_scores: dict[str, list[float]] = {}
    print("Model comparison with cross-validation (5-fold):")
    for model_name, pipeline in pipelines.items():
        cv_scores = cross_val_score(pipeline, x, y, cv=cv, scoring="accuracy")
        model_scores[model_name] = cv_scores.tolist()
        print(f"- {model_name} mean: {cv_scores.mean():.4f}")
        print(f"  std: {cv_scores.std():.4f}")
        print("  folds:", ", ".join(f"{score:.4f}" for score in cv_scores))

    best_model_name = max(
        model_scores, key=lambda key: sum(model_scores[key]) / len(model_scores[key])
    )
    best_pipeline = pipelines[best_model_name]
    print(f"\nSelected model: {best_model_name}\n")

    best_pipeline.fit(x_train, y_train)
    preds = best_pipeline.predict(x_test)

    accuracy = accuracy_score(y_test, preds)
    labels_sorted = sorted(y.unique())
    report_dict = classification_report(y_test, preds, zero_division=0, output_dict=True)

    print("Accuracy:", f"{accuracy:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds, zero_division=0))

    cm = confusion_matrix(y_test, preds, labels=labels_sorted)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, preds, display_labels=labels_sorted
    )
    disp.figure_.savefig(models_dir / "confusion_matrix.png", bbox_inches="tight")

    joblib.dump(best_pipeline, models_dir / "news_topic_model.joblib")

    metrics_payload = {
        "selected_model": best_model_name,
        "accuracy": round(float(accuracy), 4),
        "cv_scores": model_scores,
        "labels": labels_sorted,
        "classification_report": report_dict,
        "confusion_matrix": {
            "labels": labels_sorted,
            "matrix": cm.tolist(),
        },
    }
    with (models_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\nSaved model to models/news_topic_model.joblib")
    print("Saved confusion matrix to models/confusion_matrix.png")
    print("Saved metrics to models/metrics.json")


if __name__ == "__main__":
    main()

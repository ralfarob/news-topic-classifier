"""Training pipeline for multi-class news topic classification."""

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
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data_quality_check import validate_training_data
from preprocess import clean_batch


def build_pipelines() -> dict[str, Pipeline]:
    # Compare two linear models over the same TF-IDF feature space.
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


def build_param_grids() -> dict[str, dict[str, list[object]]]:
    # Keep grids intentionally small to improve quality without slowing training too much.
    return {
        "LogisticRegression": {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2],
            "clf__C": [0.5, 1.0, 2.0],
        },
        "LinearSVC": {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2],
            "clf__C": [0.5, 1.0, 2.0],
        },
    }


def main() -> None:
    data_path = Path("data/sample_news.csv")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must include 'text' and 'label' columns")

    # Fail fast when dataset quality is not good enough for training.
    quality_issues = validate_training_data(df)
    if quality_issues:
        issue_list = "\n".join(f"- {issue}" for issue in quality_issues)
        raise ValueError(f"Dataset quality check failed:\n{issue_list}")

    # Clean all texts before split so train/test and CV use identical preprocessing.
    x = clean_batch(df["text"].astype(str).tolist())
    y = df["label"].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    pipelines = build_pipelines()
    param_grids = build_param_grids()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    # Evaluate each candidate with repeated CV for a more stable estimate on small data.
    model_scores: dict[str, list[float]] = {}
    model_best_params: dict[str, dict[str, object]] = {}
    tuned_pipelines: dict[str, Pipeline] = {}

    print("Model comparison with repeated cross-validation (5 folds x 3 repeats):")
    for model_name, pipeline in pipelines.items():
        search = GridSearchCV(
            pipeline,
            param_grid=param_grids[model_name],
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        search.fit(x_train, y_train)

        # Re-score the tuned estimator to retain explicit fold scores for reporting/export.
        tuned_pipeline = search.best_estimator_
        cv_scores = cross_val_score(
            tuned_pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )

        tuned_pipelines[model_name] = tuned_pipeline
        model_scores[model_name] = cv_scores.tolist()
        model_best_params[model_name] = {
            key: value for key, value in search.best_params_.items()
        }

        print(f"- {model_name} mean: {cv_scores.mean():.4f}")
        print(f"  std: {cv_scores.std():.4f}")
        print(f"  scores: {len(cv_scores)} folds")
        print(f"  best params: {search.best_params_}")

    # Pick the model with the highest average CV accuracy.
    # Using explicit average keeps selection logic transparent for beginners.
    best_model_name = max(
        model_scores, key=lambda key: sum(model_scores[key]) / len(model_scores[key])
    )
    best_pipeline = tuned_pipelines[best_model_name]
    print(f"\nSelected model: {best_model_name}\n")

    # Train selected model on train split and evaluate on holdout split.
    best_pipeline.fit(x_train, y_train)
    preds = best_pipeline.predict(x_test)

    accuracy = accuracy_score(y_test, preds)
    labels_sorted = sorted(y.unique())
    report_dict = classification_report(y_test, preds, zero_division=0, output_dict=True)

    print("Accuracy:", f"{accuracy:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds, zero_division=0))

    # Keep numeric confusion matrix values for machine-readable export.
    cm = confusion_matrix(y_test, preds, labels=labels_sorted)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, preds, display_labels=labels_sorted
    )
    # Save image artifact for quick visual inspection outside terminal output.
    disp.figure_.savefig(models_dir / "confusion_matrix.png", bbox_inches="tight")

    joblib.dump(best_pipeline, models_dir / "news_topic_model.joblib")

    # Export structured metrics for CLI summaries and future comparisons.
    metrics_payload = {
        "selected_model": best_model_name,
        "accuracy": round(float(accuracy), 4),
        "cv_scores": model_scores,
        "cv_summary": {
            model_name: {
                "mean": round(float(sum(scores) / len(scores)), 4),
                "std": round(
                    float(pd.Series(scores, dtype="float64").std(ddof=0)), 4
                ),
            }
            for model_name, scores in model_scores.items()
        },
        "best_params": model_best_params,
        "data_summary": {
            "dataset_rows": int(len(df)),
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
        },
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

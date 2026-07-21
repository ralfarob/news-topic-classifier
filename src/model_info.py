"""Print model artifact metadata and training summary details."""

from pathlib import Path
import argparse
import json

import joblib


def _safe_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show saved model information and training metadata"
    )
    parser.add_argument("--model", default="models/news_topic_model.joblib")
    parser.add_argument("--metrics", default="models/metrics.json")
    args = parser.parse_args()

    model_path = Path(args.model)
    metrics_path = Path(args.metrics)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run src/train.py first."
        )

    model = joblib.load(model_path)

    print("Model Artifact Info")
    print("===================")
    print(f"Model path: {model_path}")
    print(f"Model type: {model.__class__.__name__}")

    if hasattr(model, "named_steps"):
        step_names = ", ".join(model.named_steps.keys())
        print(f"Pipeline steps: {step_names}")

    classes = [str(item) for item in getattr(model, "classes_", [])]
    if classes:
        print(f"Classes: {', '.join(classes)}")

    if not metrics_path.exists():
        print("")
        print(f"Metrics file not found: {metrics_path}")
        return

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    selected_model = str(metrics.get("selected_model", "unknown"))
    accuracy = _safe_float(metrics.get("accuracy", 0.0))
    cv_summary = metrics.get("cv_summary", {})
    best_params = metrics.get("best_params", {})
    data_summary = metrics.get("data_summary", {})

    print("")
    print("Training Summary")
    print("================")
    print(f"Selected model: {selected_model}")
    print(f"Holdout accuracy: {accuracy:.4f}")

    if isinstance(data_summary, dict) and data_summary:
        dataset_rows = int(_safe_float(data_summary.get("dataset_rows", 0.0)))
        train_rows = int(_safe_float(data_summary.get("train_rows", 0.0)))
        test_rows = int(_safe_float(data_summary.get("test_rows", 0.0)))
        print(f"Rows: dataset={dataset_rows} train={train_rows} test={test_rows}")

    if isinstance(cv_summary, dict) and cv_summary:
        print("CV summary:")
        for model_name, stats in cv_summary.items():
            mean_score = 0.0
            std_score = 0.0
            if isinstance(stats, dict):
                mean_score = _safe_float(stats.get("mean", 0.0))
                std_score = _safe_float(stats.get("std", 0.0))
            print(f"- {model_name}: mean={mean_score:.4f} std={std_score:.4f}")

    if isinstance(best_params, dict) and selected_model in best_params:
        print(f"Best params ({selected_model}): {best_params[selected_model]}")


if __name__ == "__main__":
    main()
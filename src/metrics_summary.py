from pathlib import Path
import argparse
import json


def _safe_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a compact summary from metrics.json"
    )
    parser.add_argument(
        "--metrics",
        default="models/metrics.json",
        help="Path to metrics JSON file",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}. Run src/train.py first."
        )

    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    selected_model = str(payload.get("selected_model", "unknown"))
    accuracy = _safe_float(payload.get("accuracy", 0.0))
    cv_scores = payload.get("cv_scores", {})
    report = payload.get("classification_report", {})
    labels = payload.get("labels", [])

    print("Evaluation Summary")
    print("==================")
    print(f"Selected model: {selected_model}")
    print(f"Holdout accuracy: {accuracy:.4f}")
    print("")

    print("CV mean accuracy by model:")
    rows: list[tuple[str, float]] = []
    if isinstance(cv_scores, dict):
        for model_name, scores in cv_scores.items():
            if isinstance(scores, list) and scores:
                mean_score = sum(float(v) for v in scores) / len(scores)
            else:
                mean_score = 0.0
            rows.append((str(model_name), mean_score))
    rows.sort(key=lambda item: item[1], reverse=True)

    for model_name, mean_score in rows:
        print(f"- {model_name}: {mean_score:.4f}")

    print("")
    print("Per-class metrics:")
    for label in labels:
        label_key = str(label)
        label_metrics = report.get(label_key, {}) if isinstance(report, dict) else {}
        precision = _safe_float(label_metrics.get("precision", 0.0))
        recall = _safe_float(label_metrics.get("recall", 0.0))
        f1 = _safe_float(label_metrics.get("f1-score", 0.0))
        support = int(_safe_float(label_metrics.get("support", 0.0)))
        print(
            f"- {label_key}: precision={precision:.2f} recall={recall:.2f} "
            f"f1={f1:.2f} support={support}"
        )


if __name__ == "__main__":
    main()
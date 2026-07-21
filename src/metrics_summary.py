"""Terminal summary view for training metrics artifacts."""

from pathlib import Path
import argparse
import json


def _safe_float(value: object) -> float:
    # Defensive cast for JSON values that may be missing or typed differently.
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _print_confusion_matrix(labels: list[str], matrix: list[list[int]]) -> None:
    if not labels or not matrix:
        print("Confusion matrix: unavailable")
        return

    if len(matrix) != len(labels) or any(len(row) != len(labels) for row in matrix):
        print("Confusion matrix: invalid shape")
        return

    # Compute dynamic widths so short and long labels align in a readable grid.
    row_label_width = max(len("true\\pred"), max(len(label) for label in labels))
    value_width = max(3, max(len(str(value)) for row in matrix for value in row))

    print("Confusion matrix (rows=true, cols=pred):")
    header = "true\\pred".rjust(row_label_width)
    for label in labels:
        header += f" | {label.rjust(value_width)}"
    print(header)
    print("-" * len(header))

    for label, row in zip(labels, matrix):
        line = label.rjust(row_label_width)
        for value in row:
            line += f" | {str(value).rjust(value_width)}"
        print(line)


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
    best_params = payload.get("best_params", {})
    report = payload.get("classification_report", {})
    labels = payload.get("labels", [])
    confusion_payload = payload.get("confusion_matrix", {})

    print("Evaluation Summary")
    print("==================")
    print(f"Selected model: {selected_model}")
    print(f"Holdout accuracy: {accuracy:.4f}")
    print("")

    print("CV mean/std accuracy by model:")
    # Build a tiny leaderboard sorted by CV mean score.
    rows: list[tuple[str, float, float]] = []
    if isinstance(cv_scores, dict):
        for model_name, scores in cv_scores.items():
            if isinstance(scores, list) and scores:
                mean_score = sum(float(v) for v in scores) / len(scores)
                variance = sum((float(v) - mean_score) ** 2 for v in scores) / len(scores)
                std_score = variance**0.5
            else:
                mean_score = 0.0
                std_score = 0.0
            rows.append((str(model_name), mean_score, std_score))
    rows.sort(key=lambda item: item[1], reverse=True)

    for model_name, mean_score, std_score in rows:
        print(f"- {model_name}: mean={mean_score:.4f} std={std_score:.4f}")
        if isinstance(best_params, dict) and model_name in best_params:
            print(f"  best params: {best_params[model_name]}")

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

    print("")
    cm_labels = []
    cm_matrix = []
    if isinstance(confusion_payload, dict):
        # Read labels/matrix from the same structure exported by train.py.
        raw_labels = confusion_payload.get("labels", [])
        raw_matrix = confusion_payload.get("matrix", [])
        if isinstance(raw_labels, list):
            cm_labels = [str(item) for item in raw_labels]
        if isinstance(raw_matrix, list):
            # Cast values defensively so malformed JSON does not crash output.
            cm_matrix = [
                [int(_safe_float(value)) for value in row]
                for row in raw_matrix
                if isinstance(row, list)
            ]

    # Render confusion matrix using the label order stored in metrics.json.
    _print_confusion_matrix(cm_labels, cm_matrix)


if __name__ == "__main__":
    main()
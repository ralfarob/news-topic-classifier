"""Integration tests for training, prediction, and reporting CLIs."""

from pathlib import Path
import json
import subprocess
import sys


def test_train_creates_artifacts() -> None:
    # Runs full training and validates required output artifacts and metrics keys.
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "src/train.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Model comparison with repeated cross-validation (5 folds x 3 repeats):" in result.stdout
    assert "Selected model:" in result.stdout
    assert "Accuracy:" in result.stdout
    assert (repo_root / "models" / "news_topic_model.joblib").exists()
    assert (repo_root / "models" / "confusion_matrix.png").exists()
    assert (repo_root / "models" / "metrics.json").exists()

    metrics_path = repo_root / "models" / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "selected_model" in metrics
    assert "accuracy" in metrics
    assert "cv_scores" in metrics
    assert "cv_summary" in metrics
    assert "best_params" in metrics
    assert "data_summary" in metrics
    assert "labels" in metrics
    assert "classification_report" in metrics
    assert "confusion_matrix" in metrics
    assert "matrix" in metrics["confusion_matrix"]


def test_predict_single_text() -> None:
    # Ensures single-text prediction mode responds with a label.
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "models" / "news_topic_model.joblib"

    if not model_path.exists():
        train_result = subprocess.run(
            [sys.executable, "src/train.py"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert train_result.returncode == 0, train_result.stderr

    result = subprocess.run(
        [
            sys.executable,
            "src/predict.py",
            "--text",
            "Team wins final match after extra time",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Prediction:" in result.stdout


def test_predict_single_text_with_scores() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "models" / "news_topic_model.joblib"

    if not model_path.exists():
        train_result = subprocess.run(
            [sys.executable, "src/train.py"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert train_result.returncode == 0, train_result.stderr

    result = subprocess.run(
        [
            sys.executable,
            "src/predict.py",
            "--text",
            "Government approves new fiscal reform",
            "--show-scores",
            "--top-k",
            "2",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Prediction:" in result.stdout
    assert "Top scores:" in result.stdout


def test_predict_batch_from_csv() -> None:
    # Ensures CSV batch mode writes output with predicted_topic column.
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "models" / "news_topic_model.joblib"

    if not model_path.exists():
        train_result = subprocess.run(
            [sys.executable, "src/train.py"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert train_result.returncode == 0, train_result.stderr

    input_path = repo_root / "tests" / "tmp_news_input.csv"
    output_path = repo_root / "tests" / "tmp_news_output.csv"
    input_path.write_text(
        "text\nParliament votes on trade bill\nStartup launches a new processor\n",
        encoding="utf-8",
    )

    try:
        result = subprocess.run(
            [
                sys.executable,
                "src/predict.py",
                "--input-csv",
                str(input_path),
                "--output-csv",
                str(output_path),
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "Saved predictions to:" in result.stdout
        assert output_path.exists()

        content = output_path.read_text(encoding="utf-8")
        assert "predicted_topic" in content
    finally:
        if input_path.exists():
            input_path.unlink()
        if output_path.exists():
            output_path.unlink()


def test_predict_batch_from_csv_with_scores() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "models" / "news_topic_model.joblib"

    if not model_path.exists():
        train_result = subprocess.run(
            [sys.executable, "src/train.py"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert train_result.returncode == 0, train_result.stderr

    input_path = repo_root / "tests" / "tmp_news_input_scores.csv"
    output_path = repo_root / "tests" / "tmp_news_output_scores.csv"
    input_path.write_text(
        "text\nParliament votes on trade bill\nStartup launches a new processor\n",
        encoding="utf-8",
    )

    try:
        result = subprocess.run(
            [
                sys.executable,
                "src/predict.py",
                "--input-csv",
                str(input_path),
                "--output-csv",
                str(output_path),
                "--show-scores",
                "--top-k",
                "2",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "Included top-2 scores in output CSV" in result.stdout
        assert output_path.exists()

        content = output_path.read_text(encoding="utf-8")
        assert "predicted_topic" in content
        assert "predicted_score" in content
        assert "top_k_rankings" in content
    finally:
        if input_path.exists():
            input_path.unlink()
        if output_path.exists():
            output_path.unlink()


def test_metrics_summary_cli() -> None:
    # Confirms terminal summary output includes all expected report sections.
    repo_root = Path(__file__).resolve().parents[1]
    metrics_path = repo_root / "models" / "metrics.json"

    if not metrics_path.exists():
        train_result = subprocess.run(
            [sys.executable, "src/train.py"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert train_result.returncode == 0, train_result.stderr

    result = subprocess.run(
        [sys.executable, "src/metrics_summary.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Evaluation Summary" in result.stdout
    assert "Selected model:" in result.stdout
    assert "CV mean/std accuracy by model:" in result.stdout
    assert "best params:" in result.stdout
    assert "Per-class metrics:" in result.stdout
    assert "Confusion matrix (rows=true, cols=pred):" in result.stdout


def test_model_info_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "models" / "news_topic_model.joblib"
    metrics_path = repo_root / "models" / "metrics.json"

    if not model_path.exists() or not metrics_path.exists():
        train_result = subprocess.run(
            [sys.executable, "src/train.py"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert train_result.returncode == 0, train_result.stderr

    result = subprocess.run(
        [sys.executable, "src/model_info.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Model Artifact Info" in result.stdout
    assert "Model type:" in result.stdout
    assert "Pipeline steps:" in result.stdout
    assert "Training Summary" in result.stdout
    assert "Selected model:" in result.stdout
    assert "Holdout accuracy:" in result.stdout


def test_data_quality_check_cli_passes_for_sample_data() -> None:
    # Baseline dataset should pass the validator with default thresholds.
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "src/data_quality_check.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Data quality check passed." in result.stdout


def test_data_quality_check_cli_fails_for_imbalanced_and_duplicate_data() -> None:
    # Synthetic dataset should trigger both imbalance and duplicate checks.
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "tests" / "tmp_bad_quality_news.csv"
    input_path.write_text(
        "text,label\n"
        "same headline,business\n"
        "same headline,business\n"
        "market update,business\n"
        "sports brief,sports\n",
        encoding="utf-8",
    )

    try:
        result = subprocess.run(
            [
                sys.executable,
                "src/data_quality_check.py",
                "--input-csv",
                str(input_path),
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        assert "Data quality check failed:" in result.stdout
        assert "Class imbalance too high:" in result.stdout
        assert "Class distribution:" in result.stdout
        assert "Suggestion: add at least" in result.stdout
        assert "Per-class deficit to match largest class:" in result.stdout
        assert "duplicate text rows" in result.stdout
        assert "Duplicate example 1:" in result.stdout
        assert "Suggestion: deduplicate repeated texts" in result.stdout
    finally:
        if input_path.exists():
            input_path.unlink()

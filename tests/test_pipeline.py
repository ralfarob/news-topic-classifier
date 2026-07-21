from pathlib import Path
import json
import subprocess
import sys


def test_train_creates_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "src/train.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Model comparison with cross-validation (5-fold):" in result.stdout
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
    assert "labels" in metrics
    assert "classification_report" in metrics
    assert "confusion_matrix" in metrics
    assert "matrix" in metrics["confusion_matrix"]


def test_predict_single_text() -> None:
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


def test_predict_batch_from_csv() -> None:
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


def test_metrics_summary_cli() -> None:
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
    assert "CV mean accuracy by model:" in result.stdout
    assert "Per-class metrics:" in result.stdout
    assert "Confusion matrix (rows=true, cols=pred):" in result.stdout

# Week 2: News Topic Classifier

This project trains a multi-class text classifier for news topics using TF-IDF and compares Logistic Regression vs LinearSVC.

## Topics
- politics
- sports
- technology
- business

## Stack
- Python
- pandas
- scikit-learn
- matplotlib
- joblib
- pytest

## Project Structure
- data/sample_news.csv: sample labeled dataset (text, label)
- src/data_quality_check.py: dataset validator (class balance and duplicate checks)
- src/preprocess.py: text cleaning helpers
- src/train.py: model comparison, training, and artifact export
- src/predict.py: single-text and batch CSV prediction
- src/metrics_summary.py: compact evaluation report from metrics.json
- tests/test_preprocess.py: unit tests for preprocessing
- tests/test_pipeline.py: integration tests for train and predict

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```powershell
python src/train.py
```

Training now runs a fail-fast data quality check first and stops if it finds:
- class imbalance above threshold
- duplicate texts (case/space-insensitive)

Expected output:
- cross-validation comparison per model
- selected best model
- accuracy and classification report
- models/news_topic_model.joblib
- models/confusion_matrix.png
- models/metrics.json

## Data Quality Check

```powershell
python src/data_quality_check.py
```

Optional custom input and threshold:

```powershell
python src/data_quality_check.py --input-csv data/sample_news.csv --max-imbalance-ratio 2.0
```

## Predict

Single text:

```powershell
python src/predict.py --text "Government announces a new budget plan"
```

Batch CSV:

```powershell
python src/predict.py --input-csv data/sample_news.csv
python src/predict.py --input-csv data/sample_news.csv --output-csv data/sample_news_predictions.csv
```

## Metrics Summary

```powershell
python src/metrics_summary.py
```

Optional custom metrics path:

```powershell
python src/metrics_summary.py --metrics models/metrics.json
```

The summary includes:
- selected model and holdout accuracy
- CV mean accuracy by model
- per-class precision/recall/F1/support
- confusion matrix table (rows=true labels, columns=predicted labels)

## Run Tests

```powershell
pytest -q
```

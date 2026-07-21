# Week 2: News Topic Classifier

This project trains a multi-class text classifier for news topics using TF-IDF, tunes Logistic Regression and LinearSVC with grid search, and selects the best model with repeated cross-validation.

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
- data/sample_news.csv: expanded labeled dataset (text, label)
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
- empty text rows or missing required columns

Model selection strategy:
- repeated stratified cross-validation (5 folds x 3 repeats)
- small grid search for TF-IDF and model hyperparameters
- best model chosen by highest CV mean accuracy

Expected output:
- repeated cross-validation mean/std per model
- best hyperparameters per model
- selected best model
- accuracy and classification report
- models/news_topic_model.joblib
- models/confusion_matrix.png
- models/metrics.json

metrics.json now stores:
- selected model and accuracy
- full CV scores per model
- CV mean/std summary per model
- best hyperparameters per model
- dataset/train/test row counts
- per-class classification report
- confusion matrix labels and numeric matrix values

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

Show top-k scores (single text):

```powershell
python src/predict.py --text "Government announces a new budget plan" --show-scores --top-k 3
```

Show top-k scores in batch CSV output:

```powershell
python src/predict.py --input-csv data/sample_news.csv --show-scores --top-k 3
```

When score output is enabled:
- models with predict_proba use true probabilities
- models with decision_function (for example LinearSVC) use softmax-normalized margins
- batch output adds predicted_score and top_k_rankings columns

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
- CV mean/std accuracy by model
- best hyperparameters by model
- per-class precision/recall/F1/support
- confusion matrix table (rows=true labels, columns=predicted labels)

## Run Tests

```powershell
pytest -q
```

VS Code task option:

```powershell
# Terminal > Run Task > Run tests
```

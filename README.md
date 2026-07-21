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
- src/preprocess.py: text cleaning helpers
- src/train.py: model comparison, training, and artifact export
- src/predict.py: single-text and batch CSV prediction
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

Expected output:
- cross-validation comparison per model
- selected best model
- accuracy and classification report
- models/news_topic_model.joblib
- models/confusion_matrix.png
- models/metrics.json

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

## Run Tests

```powershell
pytest -q
```

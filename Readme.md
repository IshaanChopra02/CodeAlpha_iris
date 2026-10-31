# CodeAlpha_iris: Iris Flower Classification

CodeAlpha Data Science Internship, Task 1. The project classifies Iris flowers (Setosa, Versicolor, Virginica) from four measurements using scikit-learn, and shows the result in an interactive Streamlit dashboard.

## Live demo

https://iris.ishaanchopra.me

## Features

- Trains a K-Nearest Neighbors model with standard scaling
- Test accuracy: 93.33% (30 held-out flowers)
- Interactive dashboard: sliders for flower measurements, predicted species with confidence, and class probabilities
- Overview page with KPI cards, a petal length trend, a species donut chart, and a searchable flower log
- CSV report export

## Project files

| File | Purpose |
|---|---|
| `Iris.csv` | Dataset (150 flowers) |
| `train_model.py` | Trains the model and saves the model, scaler and metrics |
| `iris_dashboard_pro.py` | Streamlit dashboard |
| `iris_model.pkl`, `iris_scaler.pkl`, `iris_metrics.json` | Saved model files |
| `requirements.txt` | Python packages |
| `.streamlit/config.toml` | Dark theme settings |

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run iris_dashboard_pro.py
```

Then open http://localhost:8501 in your browser.

## Tech stack

Python, scikit-learn, pandas, Streamlit, Altair, joblib

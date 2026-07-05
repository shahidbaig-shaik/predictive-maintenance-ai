# Predictive Maintenance AI

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn) ![pandas](https://img.shields.io/badge/pandas-2.0-150458?logo=pandas) ![License](https://img.shields.io/badge/license-MIT-green)

> ML system that predicts industrial equipment failure from sensor telemetry, enabling proactive maintenance before costly breakdowns occur.

## Overview

Unplanned equipment downtime costs manufacturers an estimated $50 billion annually. This project builds a classification pipeline on sensor telemetry data (temperature, vibration, pressure, RPM) to predict failure events before they happen. Feature engineering extracts degradation signals from raw time-series readings, and the trained model outputs failure probability scores that maintenance teams can act on.

## Tech Stack

| Component | Technology |
|---|---|
| ML Pipeline | scikit-learn (classification, cross-validation) |
| Feature Engineering | pandas, NumPy (rolling stats, lag features) |
| Models | Random Forest, Gradient Boosting, Logistic Regression |
| Evaluation | Precision, Recall, F1, Confusion Matrix |
| Visualization | matplotlib, seaborn |

## How It Works

- **Ingest** sensor telemetry (temperature, vibration, pressure, RPM) with labeled failure events
- **Engineer** features: rolling mean/std, lag values, rate-of-change signals
- **Train** classification models to predict failure within a configurable lookahead window
- **Evaluate** using precision/recall trade-offs (prioritizing recall to minimize missed failures)
- **Score** new telemetry to output per-machine failure probability

## Results / Key Metrics

| Metric | Value |
|---|---|
| Failure Recall | > 0.85 (catch most real failures) |
| False Positive Rate | < 0.10 (avoid unnecessary maintenance) |
| Lookahead Window | Configurable (default: 24h before failure) |

## Quick Start

```bash
git clone https://github.com/shahidbaig-shaik/predictive-maintenance-ai
cd predictive-maintenance-ai
pip install -r requirements.txt
python train.py --data data/sensor_data.csv
```

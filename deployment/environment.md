# Environment Setup

How to run the whole project locally. Everything runs on CPU; **no cloud
account, API key, or secret is required** for any step.

## Prerequisites

| Tool | Version used here | Notes |
|---|---|---|
| Python | 3.13 | Any 3.10+ should work; pin is `requirements.txt` |
| pip | bundled | plain pip is fine |
| Git | 2.x | |
| Docker (optional) | 29.x | Only for the containerized path |
| OpenMP (macOS only) | `libomp.dylib` | Required by LightGBM — see below |

### macOS OpenMP note (LightGBM)

LightGBM's native library needs an OpenMP runtime. Options:

```bash
brew install libomp                       # standard route
# or point DYLD_LIBRARY_PATH at an existing libomp, e.g. R's:
export DYLD_LIBRARY_PATH=/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib
```

Only `scripts/run_training.py` (LightGBM) needs this — the PyTorch script,
API, dashboard, and Docker image do not. On Linux, `libgomp` comes with gcc
or `apt-get install libgomp1` (already in the Dockerfile).

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment variables

| Variable | Default | Used by | Purpose |
|---|---|---|---|
| `PORT` | `5001` | `python api/app.py` | API listen port |
| `DYLD_LIBRARY_PATH` | unset | `scripts/run_training.py` (macOS) | locates `libomp.dylib` for LightGBM |

Nothing else. No AWS keys, no database URLs, no API tokens.

## Ports

| Port | Service |
|---|---|
| 5001 | Flask REST API (5000 avoided — macOS AirPlay uses it) |
| 8501 | Streamlit dashboard |

## Running everything

```bash
# 1) Training / evaluation artifacts (already committed; rerun to reproduce)
python scripts/run_training.py          # Phase 1: baselines + metrics JSON/CSV
python scripts/run_torch_training.py    # Phase 3: PyTorch checkpoint + metrics

# 2) Tests
python -m pytest tests/ -q

# 3) Dashboard
streamlit run app.py

# 4) API (dev server)
python api/app.py                       # or: gunicorn -b 0.0.0.0:5001 -w 1 api.app:app

# 5) Docker equivalents
docker build -t amazon-sales-forecasting .
docker compose up -d
```

## Artifacts consumed at runtime

| Path | Written by | Read by |
|---|---|---|
| `models/pytorch_model.pt` | `scripts/run_torch_training.py` | Flask API |
| `results/pytorch_metrics.json` | `scripts/run_torch_training.py` | API `/model`, dashboard |
| `results/model_metrics.json` | `scripts/run_training.py` | dashboard |
| `results/holdout_predictions.csv` | `scripts/run_training.py` | dashboard |
| `results/pytorch_holdout_predictions.csv` | `scripts/run_torch_training.py` | dashboard |
| `data/*.csv` | (committed source data) | training scripts, dashboard |

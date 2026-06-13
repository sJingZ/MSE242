# Deep Order-Flow Imbalance in Prediction Markets

Forecasting short-horizon mid-price returns on Polymarket NBA series-winner contracts from limit-order-book order flow, using LSTM, LSTM-MLP, and CNN-LSTM models.

**Authors:** Christy Yang, Jing Zou

## Final Report

**The final report for grading is [`MSE242_FinalReport.pdf`](MSE242_FinalReport.pdf).**


## Repository layout

| Path | Contents |
|---|---|
| `src/` | Model definitions (`cnn_lstm.py`, `lstm.py`, `lstm_mlp.py`) and Modal training apps |
| `scripts/` | EDA, hyperparameter sweeps, per-market evaluation, and analysis scripts that produced results used in the final report |
| `notebooks/` | Data EDA and pipeline notebooks for exploration |
| `data/`, `datasets/` | Raw, processed, and order-flow datasets |
| `results/` | Experiment logs, figures, market-group analysis, and report drafts |
| `MODAL_SETUP.md` | Instructions for running training on Modal |

## Setup

```bash
pip install -r requirements.txt
```

Training was run on Modal (H100 / A10G GPUs); see `MODAL_SETUP.md`. The pipeline uses Python with NumPy, pandas, PyArrow, scikit-learn, PyTorch, SciPy, and Matplotlib.

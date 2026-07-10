# IPL Score Prediction using Deep Learning (Stacked LSTM)

Predicts the final score of a T20 IPL innings from ball-by-ball match state, using
a Stacked LSTM trained on the [IPL Complete Dataset (2008–2024)](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020).

**Test set performance:** MAE ≈ 1.98 runs · RMSE ≈ 2.64 · R² ≈ 0.993

## Contents

- [`ipl_score_prediction_lstm.ipynb`](ipl_score_prediction_lstm.ipynb) — full pipeline: data loading, preprocessing, EDA, sequence building, Stacked LSTM training, evaluation
- `ipl_lstm_model.keras` — trained model (architecture + weights)
- `scaler.pkl`, `le_team.pkl`, `le_venue.pkl`, `y_scale.pkl` — preprocessing artifacts required alongside the model
- [`GUIDE.md`](GUIDE.md) — how to load the model and artifacts to make predictions (for building a UI on top of this)
- `training and testing metrices/` — loss curve, actual-vs-predicted plot, MAE/RMSE/R² comparison
- `score_distribution.png`, `avg_score_by_team.png`, `correlation_heatmap.png` — EDA plots

## Model

Stacked LSTM: `LSTM(128, return_sequences=True) → Dropout(0.3) → LSTM(64) → Dropout(0.3) → Dense(32, relu) → Dense(1, linear)`,
trained on 20-timestep (one per over) sequences of 7 features per innings, with the target scaled to [0,1] for training stability.

## Raw data

`matches.csv` / `deliveries.csv` are not included in this repo (see `.gitignore`) — download them from the
[Kaggle dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020) and place them
alongside the notebook to re-run it.

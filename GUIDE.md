# IPL Score Prediction — Model Usage Guide

For whoever is building the Flask UI. This explains what each file is and exactly
how to load the model and run a prediction.

## 1. Files you need (all in this folder)

| File | What it is |
|---|---|
| `ipl_lstm_model.keras` | The trained model — architecture + weights + optimizer state, all in one file. This is the only model file. |
| `scaler.pkl` | `MinMaxScaler` fitted on the training features. Scales raw feature values to [0,1] before feeding the model. |
| `le_team.pkl` | `LabelEncoder` that maps team name (string) -> integer id. Used for both batting and bowling team. |
| `le_venue.pkl` | `LabelEncoder` that maps venue name (string) -> integer id. |
| `y_scale.pkl` | A single float (`300.0`). The model's raw output is in [0,1] range and must be multiplied by this to get the actual predicted score in runs. |

All 5 files are required together. The model alone cannot make a correct prediction —
it only understands encoded/scaled numbers, not raw team names or run counts.

**Training summary:** Stacked LSTM, trained up to 150 epochs with early stopping
(patience 12) — training actually stopped at **epoch 110**, with the best-validation-loss
weights restored. Test set: MAE ≈ 1.98 runs, RMSE ≈ 2.64, R² ≈ 0.993.

**What "patience" means:** `EarlyStopping` watches validation loss each epoch. If it
doesn't improve for `patience` (12) consecutive epochs, training stops and the weights
from the best epoch are restored (`restore_best_weights=True`). This is why training
stopped at epoch 110 even though `epochs=150` was the max — the best epoch was around
98, then 12 epochs with no improvement triggered the stop.

## 2. Install

```bash
pip install tensorflow scikit-learn numpy
```

## 3. Load everything

```python
import pickle
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("ipl_lstm_model.keras")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("le_team.pkl", "rb") as f:
    le_team = pickle.load(f)
with open("le_venue.pkl", "rb") as f:
    le_venue = pickle.load(f)
with open("y_scale.pkl", "rb") as f:
    Y_SCALE = pickle.load(f)
```

## 4. What the model expects as input

The model takes one sequence per innings: **20 timesteps (one per over), 7 features
per timestep**, in this exact order:

1. `cum_runs` — total runs scored so far in the innings
2. `cum_wickets` — total wickets fallen so far
3. `current_run_rate` — `cum_runs / (legal_balls_bowled / 6)`
4. `current_over` — over number, 1 to 20
5. `batting_team_enc` — batting team, label-encoded
6. `bowling_team_enc` — bowling team, label-encoded
7. `venue_enc` — venue, label-encoded

If the innings is still in progress (fewer than 20 overs bowled), the remaining
timesteps are **zero-padded at the end** — the model was trained this way, so
inference must match.

## 5. Prediction function

```python
def predict_score(overs_so_far, batting_team, bowling_team, venue, max_overs=20):
    """
    overs_so_far: list of dicts, one per completed over, in order, e.g.
        [
            {"cum_runs": 8,  "cum_wickets": 0, "current_run_rate": 8.0,  "current_over": 1},
            {"cum_runs": 15, "cum_wickets": 1, "current_run_rate": 7.5,  "current_over": 2},
            ...
        ]
    batting_team, bowling_team, venue: strings, must match names seen in training data
    """
    batting_enc = le_team.transform([batting_team])[0]
    bowling_enc = le_team.transform([bowling_team])[0]
    venue_enc = le_venue.transform([venue])[0]

    seq = []
    for over in overs_so_far:
        seq.append([
            over["cum_runs"],
            over["cum_wickets"],
            over["current_run_rate"],
            over["current_over"],
            batting_enc,
            bowling_enc,
            venue_enc,
        ])

    # pad with zero-rows up to max_overs (post-padding, same as training)
    while len(seq) < max_overs:
        seq.append([0, 0, 0, 0, 0, 0, 0])
    seq = seq[:max_overs]

    X = np.array(seq, dtype="float32")               # shape (20, 7)
    X = scaler.transform(X)                            # scale features
    X = X.reshape(1, max_overs, 7)                      # add batch dimension

    pred_scaled = model.predict(X, verbose=0)[0][0]
    predicted_score = pred_scaled * Y_SCALE
    return round(float(predicted_score), 1)
```

## 6. Example call

```python
overs_so_far = [
    {"cum_runs": 8,  "cum_wickets": 0, "current_run_rate": 8.0,  "current_over": 1},
    {"cum_runs": 15, "cum_wickets": 1, "current_run_rate": 7.5,  "current_over": 2},
    {"cum_runs": 24, "cum_wickets": 1, "current_run_rate": 8.0,  "current_over": 3},
]

score = predict_score(
    overs_so_far,
    batting_team="Mumbai Indians",
    bowling_team="Chennai Super Kings",
    venue="Wankhede Stadium",
)
print(score)  # e.g. 178.4
```

## 7. Minimal Flask endpoint

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    score = predict_score(
        data["overs_so_far"],
        data["batting_team"],
        data["bowling_team"],
        data["venue"],
    )
    return jsonify({"predicted_score": score})
```

## 8. Important caveats

- **Team/venue names must match training data exactly** (case-sensitive, same spelling).
  Calling `le_team.transform(["some new team"])` on a name the encoder never saw
  raises a `ValueError`. Validate input against `le_team.classes_` /
  `le_venue.classes_` before calling.
- **Feature order matters.** The 7 features must be in the exact order listed in
  section 4 — the model has no idea what a "column name" is, it just sees a
  fixed-position vector.
- **`current_over` starts at 1**, not 0 (matches the notebook's `over + 1` convention).
- The model predicts the **final innings score**, not "runs remaining" — same
  number regardless of how many overs are already known.

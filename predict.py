import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from exceptions import (
    ModelConfigurationError,
    InvalidMatchConfigError,
    InvalidInningsProgressionError
)

# Constants
MODEL_PATH = "ipl_lstm_model.keras"
SCALER_PATH = "scaler.pkl"
TEAM_ENCODER_PATH = "le_team.pkl"
VENUE_ENCODER_PATH = "le_venue.pkl"
Y_SCALE_PATH = "y_scale.pkl"

class IPLPredictor:
    def __init__(self):
        # Load keras model
        if not os.path.exists(MODEL_PATH):
            raise ModelConfigurationError(f"Model file {MODEL_PATH} not found.")
        try:
            self.model = load_model(MODEL_PATH)
        except Exception as e:
            raise ModelConfigurationError(f"Failed to load Keras model: {str(e)}")
        
        # Load scaler
        if not os.path.exists(SCALER_PATH):
            raise ModelConfigurationError(f"Scaler file {SCALER_PATH} not found.")
        try:
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
        except Exception as e:
            raise ModelConfigurationError(f"Failed to unpickle scaler: {str(e)}")
            
        # Load label encoders
        if not os.path.exists(TEAM_ENCODER_PATH):
            raise ModelConfigurationError(f"Team encoder file {TEAM_ENCODER_PATH} not found.")
        if not os.path.exists(VENUE_ENCODER_PATH):
            raise ModelConfigurationError(f"Venue encoder file {VENUE_ENCODER_PATH} not found.")
        try:
            with open(TEAM_ENCODER_PATH, "rb") as f:
                self.le_team = pickle.load(f)
            with open(VENUE_ENCODER_PATH, "rb") as f:
                self.le_venue = pickle.load(f)
        except Exception as e:
            raise ModelConfigurationError(f"Failed to load label encoders: {str(e)}")
            
        # Load Y scale
        if not os.path.exists(Y_SCALE_PATH):
            raise ModelConfigurationError(f"Y scale file {Y_SCALE_PATH} not found.")
        try:
            with open(Y_SCALE_PATH, "rb") as f:
                self.y_scale = pickle.load(f)
        except Exception as e:
            raise ModelConfigurationError(f"Failed to load Y scale: {str(e)}")
            
        # Extract list of teams and venues
        self.teams = sorted(list(self.le_team.classes_))
        self.venues = sorted(list(self.le_venue.classes_))

    def project_sequence(self, overs_so_far, max_overs=20):
        """
        Takes a partial list of completed overs and projects the remaining overs
        up to max_overs (20) at the current run rate and wickets rate.
        If the team is already "all out" (10 wickets), no further projection occurs.
        """
        k = len(overs_so_far)
        if k == 0:
            return []
            
        if k >= max_overs:
            return overs_so_far[:max_overs]
            
        projected = list(overs_so_far)
        last_over = overs_so_far[-1]
        
        cum_runs_at_k = float(last_over["cum_runs"])
        cum_wickets_at_k = float(last_over["cum_wickets"])
        
        # All Out Check: If they are 10 wickets down, they cannot score any more.
        # Stop projecting and return sequence as is (will be padded with trailing zeros).
        if cum_wickets_at_k >= 10.0:
            return projected
            
        crr = cum_runs_at_k / k
        w_rate = cum_wickets_at_k / k
        
        for j in range(k + 1, max_overs + 1):
            cum_runs = cum_runs_at_k + (j - k) * crr
            cum_wickets = min(9.0, cum_wickets_at_k + (j - k) * w_rate) # Cap wickets at 9 to keep projection active
            projected.append({
                "cum_runs": cum_runs,
                "cum_wickets": int(round(cum_wickets)),
                "current_run_rate": cum_runs / j,
                "current_over": j
            })
            
        return projected

    def generate_sequence_from_summary(self, current_over, current_runs, current_wickets):
        """
        Generates an over-by-over sequence from a high-level summary.
        e.g., current_over = 15, current_runs = 120, current_wickets = 4.
        Generates 15 steps representing a smooth progression to that point.
        """
        seq = []
        if current_over < 1:
            return seq
            
        crr = current_runs / current_over
        w_rate = current_wickets / current_over
        
        for i in range(1, current_over + 1):
            cum_runs = i * crr
            cum_wickets = min(10.0, i * w_rate) # Limit wickets to 10
            seq.append({
                "cum_runs": cum_runs,
                "cum_wickets": int(round(cum_wickets)),
                "current_run_rate": cum_runs / i,
                "current_over": i
            })
            
        return seq

    def predict(self, overs_so_far, batting_team, bowling_team, venue, prediction_type="projected", max_overs=20):
        """
        overs_so_far: list of dicts, e.g.
            [
                {"cum_runs": 8,  "cum_wickets": 0, "current_run_rate": 8.0,  "current_over": 1},
                {"cum_runs": 15, "cum_wickets": 1, "current_run_rate": 7.5,  "current_over": 2},
                ...
            ]
        prediction_type: "raw" or "projected"
        """
        # Validate match config
        if batting_team not in self.teams:
            raise InvalidMatchConfigError(f"Unknown batting team: {batting_team}")
        if bowling_team not in self.teams:
            raise InvalidMatchConfigError(f"Unknown bowling team: {bowling_team}")
        if batting_team == bowling_team:
            raise InvalidMatchConfigError("Batting team and Bowling team cannot be the same!")
        if venue not in self.venues:
            raise InvalidMatchConfigError(f"Unknown venue: {venue}")

        # Validate innings progression logic
        prev_runs = -1.0
        prev_wickets = -1.0
        prev_over = 0.0

        for idx, over in enumerate(overs_so_far):
            try:
                over_num = float(over["current_over"])
                cum_runs = float(over["cum_runs"])
                cum_wickets = float(over["cum_wickets"])
            except (KeyError, ValueError, TypeError):
                raise InvalidInningsProgressionError("Invalid format in over snapshot details.")

            if over_num <= 0 or over_num > max_overs:
                raise InvalidInningsProgressionError(f"Over number {over_num} is out of bounds (1 to {max_overs}).")
            
            # Boundary Capping Rules: runs max 350, wickets max 10
            if cum_runs < 0:
                raise InvalidInningsProgressionError(f"Cumulative runs ({cum_runs}) cannot be negative (Over {over_num}).")
            if cum_runs > 350.0:
                raise InvalidInningsProgressionError(f"Cumulative runs ({cum_runs}) cannot exceed 350 runs (Over {over_num}).")
            if cum_runs > 36.0 * over_num:
                raise InvalidInningsProgressionError(
                    f"Cumulative runs ({cum_runs}) exceeds the theoretical limit of {int(36 * over_num)} runs in {int(over_num)} overs."
                )
            
            if cum_wickets < 0 or cum_wickets > 10:
                raise InvalidInningsProgressionError(f"Wickets lost ({cum_wickets}) must be between 0 and 10 (Over {over_num}).")

            # Check sequential progress
            if over_num <= prev_over:
                raise InvalidInningsProgressionError(f"Over indices must be strictly increasing. Found over {over_num} after over {prev_over}.")
            
            # Check cumulative constraints (non-decreasing scores)
            if cum_runs < prev_runs:
                raise InvalidInningsProgressionError(
                    f"Cumulative runs decreased from {prev_runs} to {cum_runs} at over {over_num}."
                )
            if cum_wickets < prev_wickets:
                raise InvalidInningsProgressionError(
                    f"Cumulative wickets decreased from {prev_wickets} to {cum_wickets} at over {over_num}."
                )

            prev_runs = cum_runs
            prev_wickets = cum_wickets
            prev_over = over_num

        # If projected prediction is requested, extrapolate the sequence first
        if prediction_type == "projected" and len(overs_so_far) < max_overs:
            sequence_to_feed = self.project_sequence(overs_so_far, max_overs)
        else:
            sequence_to_feed = list(overs_so_far)

        # Encode categorical features
        batting_enc = self.le_team.transform([batting_team])[0]
        bowling_enc = self.le_team.transform([bowling_team])[0]
        venue_enc = self.le_venue.transform([venue])[0]

        # Build sequence of length max_overs
        seq = []
        for over in sequence_to_feed:
            seq.append([
                float(over["cum_runs"]),
                float(over["cum_wickets"]),
                float(over["current_run_rate"]),
                float(over["current_over"]),
                float(batting_enc),
                float(bowling_enc),
                float(venue_enc)
            ])

        # Post-padding with zeros up to max_overs
        while len(seq) < max_overs:
            seq.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        seq = seq[:max_overs]

        # Scale and reshape input
        X = np.array(seq, dtype="float32")          # Shape: (20, 7)
        X = self.scaler.transform(X)                # Scale features
        X = X.reshape(1, max_overs, 7)              # Add batch dimension

        # Run prediction
        try:
            pred_scaled = self.model.predict(X, verbose=0)[0][0]
            predicted_score = pred_scaled * self.y_scale
            
            # Clamp the prediction to be at least the runs scored so far
            if len(overs_so_far) > 0:
                last_real_over = overs_so_far[-1]
                min_runs = float(last_real_over["cum_runs"])
                predicted_score = max(min_runs, predicted_score)
        except Exception as e:
            raise InvalidInningsProgressionError(f"Model inference failed: {str(e)}")
        
        return round(float(predicted_score), 1)

# Singleton instance to avoid reloading model on every request
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = IPLPredictor()
    return predictor

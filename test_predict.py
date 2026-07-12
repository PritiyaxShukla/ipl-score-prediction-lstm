import sys
from predict import get_predictor

try:
    print("Loading predictor...")
    p = get_predictor()
    print("Predictor loaded successfully.")
    
    # Test teams and venues count
    print(f"Loaded {len(p.teams)} teams and {len(p.venues)} venues.")
    
    # Test prediction
    overs_so_far = [
        {"cum_runs": 8,  "cum_wickets": 0, "current_run_rate": 8.0,  "current_over": 1},
        {"cum_runs": 15, "cum_wickets": 1, "current_run_rate": 7.5,  "current_over": 2},
        {"cum_runs": 24, "cum_wickets": 1, "current_run_rate": 8.0,  "current_over": 3},
    ]
    
    score = p.predict(
        overs_so_far,
        batting_team="Mumbai Indians",
        bowling_team="Chennai Super Kings",
        venue="Wankhede Stadium",
    )
    print(f"Test prediction successful! Predicted score: {score}")
except Exception as e:
    print(f"Error during test: {e}")
    sys.exit(1)

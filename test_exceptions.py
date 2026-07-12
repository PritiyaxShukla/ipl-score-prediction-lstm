import sys
from predict import get_predictor
from exceptions import (
    IPLPredictorException,
    InvalidMatchConfigError,
    InvalidInningsProgressionError
)

try:
    print("Initializing predictor...")
    p = get_predictor()
    print("Predictor loaded successfully.")
    
    # Test Case 1: Identical teams validation
    print("\n[Test 1] Testing identical teams validation...")
    try:
        p.predict(
            overs_so_far=[],
            batting_team="Mumbai Indians",
            bowling_team="Mumbai Indians",
            venue="Wankhede Stadium"
        )
        print("FAIL: Identical teams did not raise InvalidMatchConfigError!")
        sys.exit(1)
    except InvalidMatchConfigError as e:
        print(f"SUCCESS: Caught expected exception: {e}")

    # Test Case 2: Unknown batting team
    print("\n[Test 2] Testing unknown team validation...")
    try:
        p.predict(
            overs_so_far=[],
            batting_team="Unknown Team FC",
            bowling_team="Chennai Super Kings",
            venue="Wankhede Stadium"
        )
        print("FAIL: Unknown team did not raise InvalidMatchConfigError!")
        sys.exit(1)
    except InvalidMatchConfigError as e:
        print(f"SUCCESS: Caught expected exception: {e}")

    # Test Case 3: Decreasing runs progression
    print("\n[Test 3] Testing decreasing runs progression sequence validation...")
    invalid_runs_seq = [
        {"current_over": 1, "cum_runs": 10, "cum_wickets": 0, "current_run_rate": 10.0},
        {"current_over": 2, "cum_runs": 8, "cum_wickets": 0, "current_run_rate": 4.0} # Score decreased from 10 to 8
    ]
    try:
        p.predict(
            overs_so_far=invalid_runs_seq,
            batting_team="Mumbai Indians",
            bowling_team="Chennai Super Kings",
            venue="Wankhede Stadium"
        )
        print("FAIL: Decreasing runs did not raise InvalidInningsProgressionError!")
        sys.exit(1)
    except InvalidInningsProgressionError as e:
        print(f"SUCCESS: Caught expected exception: {e}")

    # Test Case 4: Decreasing wickets progression
    print("\n[Test 4] Testing decreasing wickets progression sequence validation...")
    invalid_wickets_seq = [
        {"current_over": 1, "cum_runs": 10, "cum_wickets": 2, "current_run_rate": 10.0},
        {"current_over": 2, "cum_runs": 15, "cum_wickets": 1, "current_run_rate": 7.5} # Wickets decreased from 2 to 1
    ]
    try:
        p.predict(
            overs_so_far=invalid_wickets_seq,
            batting_team="Mumbai Indians",
            bowling_team="Chennai Super Kings",
            venue="Wankhede Stadium"
        )
        print("FAIL: Decreasing wickets did not raise InvalidInningsProgressionError!")
        sys.exit(1)
    except InvalidInningsProgressionError as e:
        print(f"SUCCESS: Caught expected exception: {e}")

    # Test Case 5: Out of bounds over
    print("\n[Test 5] Testing out of bounds over value validation...")
    oob_over_seq = [
        {"current_over": 22, "cum_runs": 150, "cum_wickets": 3, "current_run_rate": 6.8} # Over 22 > 20
    ]
    try:
        p.predict(
            overs_so_far=oob_over_seq,
            batting_team="Mumbai Indians",
            bowling_team="Chennai Super Kings",
            venue="Wankhede Stadium"
        )
        print("FAIL: Out of bounds over did not raise InvalidInningsProgressionError!")
        sys.exit(1)
    except InvalidInningsProgressionError as e:
        print(f"SUCCESS: Caught expected exception: {e}")

    # Test Case 6: Runs > 350 validation
    print("\n[Test 6] Testing runs > 350 validation...")
    high_runs_seq = [
        {"current_over": 10, "cum_runs": 360, "cum_wickets": 3, "current_run_rate": 36.0}
    ]
    try:
        p.predict(
            overs_so_far=high_runs_seq,
            batting_team="Mumbai Indians",
            bowling_team="Chennai Super Kings",
            venue="Wankhede Stadium"
        )
        print("FAIL: Runs > 350 did not raise InvalidInningsProgressionError!")
        sys.exit(1)
    except InvalidInningsProgressionError as e:
        print(f"SUCCESS: Caught expected exception: {e}")

    # Test Case 7: Wickets > 10 validation
    print("\n[Test 7] Testing wickets > 10 validation...")
    high_wickets_seq = [
        {"current_over": 10, "cum_runs": 120, "cum_wickets": 11, "current_run_rate": 12.0}
    ]
    try:
        p.predict(
            overs_so_far=high_wickets_seq,
            batting_team="Mumbai Indians",
            bowling_team="Chennai Super Kings",
            venue="Wankhede Stadium"
        )
        print("FAIL: Wickets > 10 did not raise InvalidInningsProgressionError!")
        sys.exit(1)
    except InvalidInningsProgressionError as e:
        print(f"SUCCESS: Caught expected exception: {e}")

    # Test Case 8: Wickets = 10 validation (All Out)
    print("\n[Test 8] Testing wickets = 10 (All Out) projection truncation...")
    all_out_seq = [
        {"current_over": 15, "cum_runs": 110, "cum_wickets": 10, "current_run_rate": 7.33}
    ]
    try:
        score = p.predict(
            overs_so_far=all_out_seq,
            batting_team="Mumbai Indians",
            bowling_team="Chennai Super Kings",
            venue="Wankhede Stadium"
        )
        print(f"SUCCESS: Wickets = 10 (All Out) predicted score: {score}")
    except Exception as e:
        print(f"FAIL: Wickets = 10 threw unexpected exception: {e}")
        sys.exit(1)

    # Test Case 9: Impossible runs per over validation (runs > 36 * over)
    print("\n[Test 9] Testing runs > 36 * over impossibility validation...")
    impossible_runs_seq = [
        {"current_over": 2, "cum_runs": 80, "cum_wickets": 1, "current_run_rate": 40.0} # 80 runs in 2 overs is > 72 (max 36 per over)
    ]
    try:
        p.predict(
            overs_so_far=impossible_runs_seq,
            batting_team="Mumbai Indians",
            bowling_team="Chennai Super Kings",
            venue="Wankhede Stadium"
        )
        print("FAIL: Impossible runs per over did not raise InvalidInningsProgressionError!")
        sys.exit(1)
    except InvalidInningsProgressionError as e:
        print(f"SUCCESS: Caught expected exception: {e}")

    print("\nALL BACKEND EXCEPTION AND VALIDATION TESTS COMPLETED SUCCESSFULLY!")

except Exception as ex:
    print(f"ERROR: Unexpected exception during test execution: {ex}")
    sys.exit(1)

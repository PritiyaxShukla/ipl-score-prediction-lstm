import sys
import logging
import traceback
from flask import Flask, request, jsonify, render_template
from predict import get_predictor
from exceptions import IPLPredictorException

# Configure logging
# Logs write to stdout and to app.log in the project root
log_format = "%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("IPLPredictorApp")

app = Flask(__name__)

# Initialize predictor on startup to fail fast if configurations are broken
try:
    logger.info("Initializing prediction engine and loading models/scalers...")
    predictor = get_predictor()
    logger.info("Prediction engine loaded successfully!")
except Exception as e:
    logger.critical(f"CRITICAL MODEL STARTUP FAILURE: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

@app.route("/")
def index():
    logger.info(f"Serving index.html dashboard (Request from: {request.remote_addr})")
    return render_template("index.html")

@app.route("/teams-venues", methods=["GET"])
def get_teams_venues():
    logger.info("Serving teams and venues listing metadata.")
    return jsonify({
        "success": True,
        "teams": predictor.teams,
        "venues": predictor.venues
    })

@app.route("/predict", methods=["POST"])
def predict():
    client_ip = request.remote_addr
    logger.info(f"Received prediction request from {client_ip}")
    
    try:
        data = request.get_json()
        if not data:
            logger.warning(f"Rejected prediction request from {client_ip} due to empty JSON body.")
            return jsonify({"success": False, "error": "Missing JSON request body."}), 400

        # Required match config parameters
        batting_team = data.get("batting_team")
        bowling_team = data.get("bowling_team")
        venue = data.get("venue")
        mode = data.get("mode", "quick")
        prediction_type = data.get("prediction_type", "projected")

        logger.info(
            f"Prediction configuration - Matchup: {batting_team} vs {bowling_team} | Venue: {venue} | Mode: {mode} | Type: {prediction_type}"
        )

        # Check required params
        if not batting_team or not bowling_team or not venue:
            logger.warning(f"Rejected request from {client_ip}: Missing match setup fields.")
            return jsonify({"success": False, "error": "Missing match configuration fields (batting team, bowling team, or venue)."}), 400

        # Determine overs_so_far based on mode
        if mode == "quick":
            try:
                current_over = float(data.get("current_over", 0))
                current_runs = float(data.get("current_runs", 0))
                current_wickets = float(data.get("current_wickets", 0))
            except (ValueError, TypeError):
                logger.warning(f"Rejected request from {client_ip}: Invalid float numeric inputs in quick tracker.")
                return jsonify({"success": False, "error": "Invalid format for over, runs, or wickets."}), 400

            completed_overs_count = int(current_over)
            
            # Generate the sequence of completed overs
            overs_so_far = predictor.generate_sequence_from_summary(
                completed_overs_count, 
                current_runs, 
                current_wickets
            )
            
            runs_so_far = current_runs
            wickets_so_far = current_wickets
            over_val = current_over
        else:
            overs_raw = data.get("overs_so_far", [])
            overs_so_far = []
            
            for over in overs_raw:
                try:
                    overs_so_far.append({
                        "cum_runs": float(over["cum_runs"]),
                        "cum_wickets": float(over["cum_wickets"]),
                        "current_run_rate": float(over["current_run_rate"]),
                        "current_over": float(over["current_over"])
                    })
                except (KeyError, ValueError, TypeError):
                    logger.warning(f"Rejected request from {client_ip}: Invalid detailed over array layout.")
                    return jsonify({"success": False, "error": "Invalid format in detailed overs list."}), 400

            # Sort progression list
            overs_so_far.sort(key=lambda x: x["current_over"])

            if len(overs_so_far) > 0:
                last_over = overs_so_far[-1]
                runs_so_far = last_over["cum_runs"]
                wickets_so_far = last_over["cum_wickets"]
                over_val = last_over["current_over"]
            else:
                runs_so_far = 0.0
                wickets_so_far = 0.0
                over_val = 0.0

        # Calculate run rate and linear projection (baseline)
        if over_val > 0:
            crr = runs_so_far / over_val
            linear_projection = crr * 20.0
        else:
            crr = 0.0
            linear_projection = 0.0

        # Run model prediction
        predicted_score = predictor.predict(
            overs_so_far,
            batting_team,
            bowling_team,
            venue,
            prediction_type=prediction_type
        )

        # Get full sequence evaluated
        if prediction_type == "projected":
            full_seq = predictor.project_sequence(overs_so_far, max_overs=20)
        else:
            full_seq = list(overs_so_far)
            while len(full_seq) < 20:
                full_seq.append({
                    "cum_runs": 0.0,
                    "cum_wickets": 0.0,
                    "current_run_rate": 0.0,
                    "current_over": len(full_seq) + 1
                })

        logger.info(
            f"Prediction SUCCESS from {client_ip} - Current Score: {runs_so_far}/{wickets_so_far} at Over {over_val} -> Predicted Final Score: {predicted_score}"
        )

        return jsonify({
            "success": True,
            "predicted_score": predicted_score,
            "current_run_rate": round(crr, 2),
            "linear_projection": round(linear_projection, 1),
            "full_sequence": full_seq
        })

    except IPLPredictorException as ipl_err:
        logger.error(f"IPLPredictorError processing request from {client_ip}: {ipl_err.message}")
        return jsonify({"success": False, "error": ipl_err.message}), ipl_err.status_code
    except Exception as e:
        logger.error(f"Unexpected prediction execution error from {client_ip}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": f"An unexpected error occurred: {str(e)}"}), 500


# Centralized error handlers for standard HTTP codes
@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"404 Page Not Found: {request.url} (IP: {request.remote_addr})")
    return jsonify({"success": False, "error": "Requested resource not found"}), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"500 Internal Server Error (IP: {request.remote_addr})")
    return jsonify({"success": False, "error": "Internal server error occurred"}), 500


if __name__ == "__main__":
    logger.info("Starting Flask application web server on http://0.0.0.0:5001...")
    # Port is set to 5001 to prevent collisions on macOS
    app.run(debug=True, host="0.0.0.0", port=5001)

import argparse
import os
import re
import sys
from flask import Flask, jsonify, request, send_from_directory
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from solution_guidance.model import model_train, model_load, model_predict, MODEL_VERSION, MODEL_VERSION_NOTE

app = Flask(__name__)

@app.route('/status', methods=['GET'])
def health_check():
    """Check if the API is up and running."""
    return jsonify({'status': 'ok'})

def serialize_numpy_data(data):
    """Convert numpy data structures to a serializable format."""
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in data.items()}

@app.route('/forecast', methods=['POST'])
def forecast():
    """
    Handle prediction requests and return the forecasted revenue for specified countries.
    """
    if not request.is_json:
        print("ERROR: Invalid request format (expected JSON).")
        return jsonify([]), 400

    request_data = request.get_json()

    if 'query' not in request_data:
        print("ERROR: Missing 'query' in the request data.")
        return jsonify([]), 400

    test_mode = request_data.get('mode', 'prod') == 'test'
    query = request_data['query']

    countries_to_forecast = query['country'].split(',') if query['country'] != 'all' else [
        'portugal', 'united_kingdom', 'hong_kong', 'eire', 'spain', 'france', 
        'singapore', 'norway', 'germany', 'netherlands']

    forecast_results = {}
    for country in countries_to_forecast:
        prediction = model_predict(country, query['year'], query['month'], query['day'], test_mode)
        print(f"Predicted revenue for {country}: {prediction['y_pred'][0]}")
        forecast_results[country] = serialize_numpy_data(prediction)

    return jsonify(forecast_results)

@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Trigger the model retraining process. Accepts a 'mode' flag for test or production mode.
    """
    if not request.is_json:
        print("ERROR: Invalid request format (expected JSON).")
        return jsonify(False), 400

    request_data = request.get_json()

    test_mode = request_data.get('mode', 'prod') == 'test'
    print("... Starting model training...")
    model_train(test_mode)
    print("... Model training completed.")

    return jsonify(True)

@app.route('/logs/<log_filename>', methods=['GET'])
def retrieve_logs(log_filename):
    """
    Provide a downloadable log file if it exists.
    """
    if not re.search(r"\.log$", log_filename):
        print(f"ERROR: Invalid log file request: {log_filename}")
        return jsonify([]), 400

    log_directory = os.path.join(".", "log")
    if not os.path.isdir(log_directory):
        print("ERROR: Log directory not found.")
        return jsonify([]), 404

    log_file_path = os.path.join(log_directory, log_filename)
    if not os.path.exists(log_file_path):
        print(f"ERROR: Log file does not exist: {log_filename}")
        return jsonify([]), 404

    return send_from_directory(log_directory, log_filename, as_attachment=True)

if __name__ == '__main__':
    # Argument parsing for debug mode
    parser = argparse.ArgumentParser(description="Run the Flask API for model training and prediction.")
    parser.add_argument("-d", "--debug", action="store_true", help="Run Flask in debug mode")
    args = parser.parse_args()

    if args.debug:
        app.run(debug=True, port=8080)
    else:
        app.run(host='0.0.0.0', threaded=True, port=8080)

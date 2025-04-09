from flask import Flask, request, jsonify
import pandas as pd
import os
import joblib

from retrain_model import retrain_model

app = Flask(__name__)
FEEDBACK_FILE = "feedback_data.csv"

@app.route("/api/feedback", methods=["POST"])
def collect_feedback():
    data = request.json  # Expecting JSON from frontend

    df = pd.DataFrame([data])
    if os.path.exists(FEEDBACK_FILE):
        df.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(FEEDBACK_FILE, index=False)

    return jsonify({"message": "Feedback received!"}), 200

@app.route("/api/retrain", methods=["POST"])
def retrain():
    try:
        retrain_model(FEEDBACK_FILE)
        return jsonify({"message": "Model retrained successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000)

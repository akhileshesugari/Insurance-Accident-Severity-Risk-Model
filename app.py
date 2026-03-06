"""
Accident Severity Prediction API
Author: Akhilesh Esugari
Description: REST API for predicting road accident severity using a
             Random Forest model trained on UK road accident data (2022).
"""

from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Load model artifacts ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model    = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"),    "rb"))
imputer  = pickle.load(open(os.path.join(BASE_DIR, "imputer.pkl"),  "rb"))
scaler   = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"),   "rb"))
features = pickle.load(open(os.path.join(BASE_DIR, "features.pkl"), "rb"))

SEVERITY_LABELS = {0: "Slight", 1: "Serious or Fatal"}


@app.route("/", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model": "Random Forest Accident Severity Classifier",
        "version": "1.0.0",
        "features_expected": features
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict accident severity from input features.

    Expected JSON body:
    {
        "number_of_vehicles": 2,
        "number_of_casualties": 1,
        "day_of_week": 3,
        "speed_limit": 30,
        "road_surface_conditions": 1,
        "vehicle_manoeuvre": 9,
        "sex_of_driver": 1,
        "age_of_driver": 35,
        "age_of_vehicle": 5,
        "driver_home_area_type": 1,
        "sex_of_casualty": 1,
        "age_of_casualty": 30,
        "pedestrian_location": 0,
        "car_passenger": 0,
        "casualty_home_area_type": 1
    }

    Returns:
    {
        "prediction": "Slight",
        "confidence": 0.87,
        "probabilities": {"Slight": 0.87, "Serious or Fatal": 0.13}
    }
    """
    try:
        data = request.get_json(force=True)

        # Validate all required features are present
        missing = [f for f in features if f not in data]
        if missing:
            return jsonify({
                "error": "Missing required features",
                "missing_fields": missing
            }), 400

        # Build input array in correct feature order
        input_values = [[data[f] for f in features]]

        # Preprocess: impute → scale
        input_imputed = imputer.transform(input_values)
        input_scaled  = scaler.transform(input_imputed)

        # Predict
        prediction    = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        prob_dict = {
            SEVERITY_LABELS[i]: round(float(p), 4)
            for i, p in enumerate(probabilities)
        }

        return jsonify({
            "prediction":    SEVERITY_LABELS[int(prediction)],
            "confidence":    round(float(max(probabilities)), 4),
            "probabilities": prob_dict
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Predict severity for multiple records at once.

    Expected JSON body:
    {
        "records": [
            { ...feature dict 1... },
            { ...feature dict 2... }
        ]
    }
    """
    try:
        data    = request.get_json(force=True)
        records = data.get("records", [])

        if not records:
            return jsonify({"error": "No records provided"}), 400

        results = []
        for i, record in enumerate(records):
            missing = [f for f in features if f not in record]
            if missing:
                results.append({
                    "record_index": i,
                    "error": f"Missing fields: {missing}"
                })
                continue

            input_values  = [[record[f] for f in features]]
            input_imputed = imputer.transform(input_values)
            input_scaled  = scaler.transform(input_imputed)

            prediction    = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]

            results.append({
                "record_index": i,
                "prediction":   SEVERITY_LABELS[int(prediction)],
                "confidence":   round(float(max(probabilities)), 4)
            })

        return jsonify({"results": results, "total": len(results)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

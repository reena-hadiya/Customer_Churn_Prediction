from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import pandas as pd
import joblib

app = Flask(__name__)

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "MODELS" / "churn_pipeline.pkl"
FRONTEND_DIR = BASE_DIR / "FRONTEND"

# Load saved model pipeline
try:
    model = joblib.load(MODEL_PATH)
    print("Customer Churn model loaded successfully!")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")


# Open frontend
@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


# Serve CSS and other frontend files
@app.route("/<path:filename>")
def frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Model could not be loaded. Check the model file path."
        }), 500

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Please send valid customer details."
            }), 400

        # Convert frontend data into a DataFrame.
        # Column names must match the training features.
        required_features = [
            "tenure",
            "TotalCharges",
            "MonthlyCharges",
            "InternetService",
            "Contract",
            "PaymentMethod",
            "OnlineSecurity",
            "TechSupport",
            "OnlineBackup",
            "PaperlessBilling",
            "StreamingTV",
            "DeviceProtection",
            "StreamingMovies"
        ]

        missing_features = [
            feature for feature in required_features
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing fields: " + ", ".join(missing_features)
            }), 400

        # Numeric fields
        numeric_features = [
            "tenure",
            "TotalCharges",
            "MonthlyCharges"
        ]

        customer_data = {}

        for feature in numeric_features:
            try:
                value = float(data[feature])
            except (TypeError, ValueError):
                return jsonify({
                    "error": f"{feature} must be a valid number."
                }), 400

            if not pd.notna(value):
                return jsonify({
                    "error": f"{feature} must be a valid number."
                }), 400

            customer_data[feature] = value

        if customer_data["tenure"] < 0 or customer_data["tenure"] > 100:
            return jsonify({
                "error": "Tenure must be between 0 and 100 months."
            }), 400

        if customer_data["TotalCharges"] < 0:
            return jsonify({
                "error": "Total Charges cannot be negative."
            }), 400

        if customer_data["MonthlyCharges"] < 0:
            return jsonify({
                "error": "Monthly Charges cannot be negative."
            }), 400

        # Categorical fields: retain their original values
        categorical_features = [
            "InternetService",
            "Contract",
            "PaymentMethod",
            "OnlineSecurity",
            "TechSupport",
            "OnlineBackup",
            "PaperlessBilling",
            "StreamingTV",
            "DeviceProtection",
            "StreamingMovies"
        ]

        for feature in categorical_features:
            value = data[feature]

            if not isinstance(value, str) or not value.strip():
                return jsonify({
                    "error": f"Please select a valid value for {feature}."
                }), 400

            customer_data[feature] = value.strip()

        # Preserve training column order
        input_df = pd.DataFrame(
            [customer_data],
            columns=required_features
        )

        # Saved pipeline handles preprocessing automatically
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        churn_probability = float(probabilities[1])

        return jsonify({
            "prediction": prediction,
            "result": (
                "Customer Likely to Churn"
                if prediction == 1
                else "Customer Likely to Stay"
            ),
            "churn_probability": round(churn_probability * 100, 2)
        })

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Please check the input values and saved model."
        }), 400


if __name__ == "__main__":
    app.run(debug=True)


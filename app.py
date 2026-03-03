from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import traceback

app = Flask(__name__)

# ==============================
# Load Pipeline + Threshold
# ==============================

try:
    pipeline = joblib.load("models/full_pipeline.pkl")
    threshold = joblib.load("models/threshold.pkl")
    print("✓ Pipeline loaded successfully")

except Exception as e:
    print("❌ Loading error:")
    traceback.print_exc()
    raise e


# ==============================
# Home Route
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# Prediction Route
# ==============================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No input data provided"
            }), 400

        # Create raw input dataframe
        input_df = pd.DataFrame({
            "step": [int(data["step"])],
            "amount": [float(data["amount"])],
            "age": [str(data["age"])],
            "gender": [str(data["gender"])],
            "merchant": [str(data["merchant"])],
            "category": [str(data["category"])]
        })

        # Pipeline handles preprocessing + prediction
        prob = pipeline.predict_proba(input_df)[0][1]
        prediction = 1 if prob >= threshold else 0

        return jsonify({
            "success": True,
            "is_fraud": bool(prediction),
            "fraud_probability": round(float(prob), 4),
            "threshold_used": threshold
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# ==============================
# Run App
# ==============================

if __name__ == "__main__":
    app.run(debug=True)
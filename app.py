from flask import Flask, request, jsonify
import joblib
import pandas as pd
from constants import VALID_VALUES

app = Flask(__name__)

model = joblib.load(r"C:\Users\games\churn-predictor\model/churn_model.pkl")

@app.route("/predict", methods=["Post"])
def predict():
    data = request.json

    for field, valid_options in VALID_VALUES.items():
        if data.get(field) not in valid_options:
            return jsonify({"error": f"Invalid value for {field}. Must be one of {valid_options}"}), 400
        
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return jsonify({
        "churned": bool(prediction),
        "churn_probability": round(probability, 3)
    })

if __name__ == "__main__":
    app.run(debug=True)
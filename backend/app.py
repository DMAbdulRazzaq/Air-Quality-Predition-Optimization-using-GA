from flask import Flask, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# Fix paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "saved_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

def categorize_aqi(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy (Sensitive)"
    elif aqi <= 200:
        return "Unhealthy"
    else:
        return "Hazardous"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        features = [
            data["PM2.5"],
            data["PM10"],
            data["NO2"],
            data["SO2"],
            data["CO"],
            data["Ozone"]
        ]

        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)

        prediction = model.predict(features)[0]
        category = categorize_aqi(prediction)

        return jsonify({
            "AQI": float(prediction),
            "category": category
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
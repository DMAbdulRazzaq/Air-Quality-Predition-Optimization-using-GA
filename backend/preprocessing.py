from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

def scale_data(X_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    pickle.dump(scaler, open("backend/scaler.pkl", "wb"))
    return X_scaled

# Load scaler once
scaler = pickle.load(open("backend/scaler.pkl", "rb"))

def preprocess_input(data):
    """
    data = {
        "PM2.5": value,
        "PM10": value,
        "NO2": value,
        "SO2": value,
        "CO": value,
        "Ozone": value
    }
    """

    try:
        features = [
            data["PM2.5"],
            data["PM10"],
            data["NO2"],
            data["SO2"],
            data["CO"],
            data["Ozone"]
        ]

        features = np.array(features).reshape(1, -1)

        scaled = scaler.transform(features)

        return scaled

    except Exception as e:
        raise ValueError(f"Invalid input: {e}")
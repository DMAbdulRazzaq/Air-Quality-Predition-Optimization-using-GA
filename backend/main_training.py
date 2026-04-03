import numpy as np
import pandas as pd
import pickle

from preprocessing import scale_data
from ga_optimizer import GAOptimizer
from model import train_model

# Load real dataset
data = pd.read_csv("data/dataset.csv")

# Drop unnecessary columns
drop_cols = ["Date", "Month", "Year", "Holidays_Count", "Days"]
data = data.drop(columns=[col for col in drop_cols if col in data.columns])

# Remove missing values
data = data.dropna()

# Select correct features
features = ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]

X = data[features].values
y = data["AQI"].values
y = y.clip(0, 500)

# Scale data
X_scaled = scale_data(X)

# GA Optimization
ga = GAOptimizer(X_scaled, y, pop_size=15, generations=15)
best_params = ga.run()

print("Best Params:", best_params)

# Train final model
model = train_model(X_scaled, y, best_params)

# Save model
pickle.dump(model, open("backend/saved_model.pkl", "wb"))

# Save cleaned dataset (optional)
data.to_csv("data/final_dataset.csv", index=False)
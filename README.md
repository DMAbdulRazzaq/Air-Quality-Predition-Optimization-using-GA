# Air Quality Prediction Model Optimization using CUDA-based Genetic Algorithm

A complete full-stack academic mini-project for Air Quality Index (AQI) prediction using a **RandomForestRegressor**, optimized with a **Genetic Algorithm**, with **CUDA-based RMSE evaluation** via `numba.cuda`, exposed through a **Flask API**, and presented with a **Streamlit frontend**.

## Features

* Real-world AQI dataset support
* Data preprocessing and scaling with scikit-learn
* RandomForestRegressor baseline model
* Genetic Algorithm hyperparameter optimization
* CUDA-based RMSE computation with CPU fallback
* Flask REST API for real-time prediction
* Streamlit UI with landing page and prediction screen
* AQI category mapping:

  * 0–50 → Good
  * 51–100 → Moderate
  * 101–150 → Unhealthy (Sensitive)
  * 151–200 → Unhealthy
  * 201+ → Hazardous

## Project Structure

```text
air_quality_project/
│── backend/
│   ├── __init__.py
│   ├── app.py
│   ├── model.py
│   ├── ga_optimizer.py
│   ├── cuda_module.py
│   ├── preprocessing.py
│   ├── main_training.py
│   ├── saved_model.pkl
│   └── scaler.pkl
│
│── frontend/
│   └── streamlit_app.py
│
│── data/
│   └── dataset.csv
│
├── requirements.txt
└── README.md
```

## Dataset

The project is built around an AQI dataset containing columns like:

* `Date`
* `Month`
* `Year`
* `Holidays_Count`
* `Days`
* `PM2.5`
* `PM10`
* `NO2`
* `SO2`
* `CO`
* `Ozone`
* `AQI`

The model uses the pollutant features:

* `PM2.5`
* `PM10`
* `NO2`
* `SO2`
* `CO`
* `Ozone`

## Setup

### 1) Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Training the Model

Run the training script from the project root:

```bash
python backend/main_training.py
```

This will:

* load the dataset
* preprocess and scale features
* run GA optimization
* train the final model
* save the trained model and scaler

Outputs:

* `backend/saved_model.pkl`
* `backend/scaler.pkl`

## Running the Flask Backend

Start the API server:

```bash
python backend/app.py
```

The backend will run at:

```text
http://127.0.0.1:5000
```

### API Endpoint

#### `POST /predict`

**Request body:**

```json
{
  "PM2.5": 120,
  "PM10": 200,
  "NO2": 50,
  "SO2": 20,
  "CO": 1.5,
  "Ozone": 30
}
```

**Response:**

```json
{
  "AQI": 225.6,
  "Category": "Hazardous"
}
```

## Running the Streamlit Frontend

Start the UI:

```bash
streamlit run frontend/streamlit_app.py
```

The frontend provides:

* a landing page
* AQI prediction form
* category display
* simple AQI visualization

## CUDA Notes

This project includes CUDA-based RMSE computation using `numba.cuda`.

Important:

* If CUDA is available, the GPU path is used.
* If CUDA is not available, the code automatically falls back to CPU RMSE.
* This keeps the project runnable on systems without a CUDA-capable GPU.

## Troubleshooting

### `ModuleNotFoundError: No module named 'backend'`

Run training from the project root:

```bash
python backend/main_training.py
```

Make sure `backend/__init__.py` exists.

### CUDA / NVVM errors

If your system does not have the CUDA toolkit installed, the fallback path will be used automatically. The project still runs correctly.

### `KeyError` in Streamlit or Flask

Make sure the frontend and backend use the same JSON keys:

* `AQI`
* `Category`

Also ensure the request keys match exactly:

* `PM2.5`
* `PM10`
* `NO2`
* `SO2`
* `CO`
* `Ozone`

## Model Pipeline

1. Load dataset
2. Clean and preprocess data
3. Scale features
4. Train/test split
5. Genetic Algorithm hyperparameter search
6. Final RandomForest training
7. Save model and scaler
8. Serve prediction through Flask
9. Display results in Streamlit

## Example Workflow

1. Train the model:

```bash
python backend/main_training.py
```

2. Start backend:

```bash
python backend/app.py
```

3. Start frontend:

```bash
streamlit run frontend/streamlit_app.py
```

4. Enter pollutant values in the UI and get AQI prediction.

## Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **scikit-learn**
* **Numba CUDA**
* **Flask**
* **Streamlit**
* **Matplotlib**

## License

For academic use and project demonstration.
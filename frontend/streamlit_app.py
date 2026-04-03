import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="AQI Predictor", layout="centered")

# ------------------------------
# Navigation (Landing + Predictor)
# ------------------------------
menu = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Home", "📊 AQI Predictor"]
)

# ------------------------------
# LANDING PAGE
# ------------------------------
if menu == "🏠 Home":
    st.title("🌍 Air Quality Prediction System")

    st.markdown("""
    ### 🚀 Project Overview
    This system predicts **Air Quality Index (AQI)** using:
    
    - 🤖 Machine Learning (Random Forest)
    - 🧬 Genetic Algorithm Optimization
    - ⚡ CUDA-based GPU Acceleration
    - 🌐 Flask API Backend
    - 🎨 Streamlit Frontend
    
    ---
    
    ### 📌 Features
    - Real-time AQI prediction
    - GPU vs CPU performance comparison
    - Optimized ML model
    - Interactive UI
    
    ---
    
    ### 📊 AQI Categories
    | AQI Range | Category |
    |----------|---------|
    | 0–50     | Good 🟢 |
    | 51–100   | Moderate 🟡 |
    | 101–150  | Unhealthy (Sensitive) 🟠 |
    | 151–200  | Unhealthy 🔴 |
    | 201+     | Hazardous ⚫ |
    """)

    st.info("👉 Go to 'AQI Predictor' from sidebar to test the model")

# ------------------------------
# PREDICTION PAGE
# ------------------------------
elif menu == "📊 AQI Predictor":

    st.title("📊 AQI Predictor")

    # Input fields
    pm25 = st.number_input("PM2.5", min_value=0.0)
    pm10 = st.number_input("PM10", min_value=0.0)
    no2 = st.number_input("NO2", min_value=0.0)
    so2 = st.number_input("SO2", min_value=0.0)
    co = st.number_input("CO", min_value=0.0)
    ozone = st.number_input("Ozone", min_value=0.0)

    if st.button("🚀 Predict AQI"):

        data = {
            "PM2.5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": co,
            "Ozone": ozone
        }

        try:
            response = requests.post(
                "http://127.0.0.1:5000/predict",
                json=data
            )

            result = response.json()

            if "error" in result:
                st.error(result["error"])

            else:
                aqi = result["AQI"]
                category = result.get("Category") or result.get("category")

                # Main output
                st.success(f"AQI: {aqi:.2f}")
                st.info(f"Category: {category}")

                # Color indicator
                if aqi <= 50:
                    st.success("🟢 Good Air Quality")
                elif aqi <= 100:
                    st.info("🟡 Moderate Air Quality")
                elif aqi <= 150:
                    st.warning("🟠 Unhealthy for Sensitive Groups")
                elif aqi <= 200:
                    st.warning("🔴 Unhealthy")
                else:
                    st.error("⚫ Hazardous")


                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=aqi,
                    title={'text': "AQI Level"},
                    gauge={
                        'axis': {'range': [0, 500]},
                        'steps': [
                            {'range': [0, 50], 'color': "green"},
                            {'range': [50, 100], 'color': "yellow"},
                            {'range': [100, 150], 'color': "orange"},
                            {'range': [150, 200], 'color': "red"},
                            {'range': [200, 500], 'color': "black"},
                        ],
                    }
                ))

                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"❌ Connection error: {e}")
import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")
PIPELINE_PATH = os.path.join(BASE_DIR, "..", "models", "pipeline.pkl")

# check model
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found! Run: python src/main.py train")
    st.stop()

model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)

# 🔥 TITLE
st.title("🏠 House Price Prediction App")

# 🔥 SIDEBAR HELP
st.sidebar.header("📘 Input Guide")

st.sidebar.write("Example values you can try:")

st.sidebar.markdown("""
- Longitude: -122.23  
- Latitude: 37.88  
- Housing Median Age: 20  
- Total Rooms: 1000  
- Total Bedrooms: 200  
- Population: 500  
- Households: 150  
- Median Income: 3.5  
- Ocean Proximity: NEAR BAY  
""")

st.sidebar.info("💡 Tip: Enter realistic values for better predictions")

# 🔹 INPUT FIELDS (no default values)
col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Longitude")
    latitude = st.number_input("Latitude")
    housing_median_age = st.number_input("Housing Median Age")

with col2:
    total_rooms = st.number_input("Total Rooms")
    total_bedrooms = st.number_input("Total Bedrooms")
    population = st.number_input("Population")

households = st.number_input("Households")
median_income = st.number_input("Median Income")

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["INLAND", "NEAR BAY", "NEAR OCEAN", "<1H OCEAN", "ISLAND"]
)

# 🔥 BUTTON
if st.button("Predict Price"):

    input_data = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    transformed = pipeline.transform(input_data)
    prediction = model.predict(transformed)

    st.success(f"💰 Predicted Price: {prediction[0]:,.2f}")
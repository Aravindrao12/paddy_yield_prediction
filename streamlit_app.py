import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL","http://127.0.0.1:8000").rstrip("/")

st.title("Paddy Yield Prediction")

# Dropdown values come from FastAPI (/options)
try:
    opts = requests.get(f"{API_URL}/options").json()
except requests.exceptions.ConnectionError:
    st.error("FastAPI server is not running.")
    st.stop()

# Input widgets

hectares = st.number_input("Hectares", min_value=1, value=5)

agriblock = st.selectbox("Agriblock", opts["Agriblock"])
variety = st.selectbox("Variety", opts["Variety"])
soil_types = st.selectbox("Soil Types", opts["Soil Types"])
nursery = st.selectbox("Nursery", opts["Nursery"])

rain_30 = st.number_input("30DRain (in mm)", value=400.0)
rain_30_50 = st.number_input("30_50DRain (in mm)", value=300.0)

min_temp_1 = st.number_input("Min temp D1-D30", value=25.0)
max_temp_1 = st.number_input("Max temp D1-D30", value=35.0)
min_temp_2 = st.number_input("Min temp D31-D60", value=25.0)
max_temp_2 = st.number_input("Max temp D31-D60", value=35.0)
max_temp_3 = st.number_input("Max temp D61-D90", value=34.0)
min_temp_4 = st.number_input("Min temp D91-D120", value=24.0)

wind_speed_1 = st.number_input("Wind Speed D1-D30 (Knots)", value=10.0)
wind_speed_2 = st.number_input("Wind Speed D31-D60 (Knots)", value=10.0)
wind_speed_3 = st.number_input("Wind Speed D61-D90 (Knots)", value=10.0)
wind_speed_4 = st.number_input("Wind Speed D91-D120 (Knots)", value=10.0)

wind_dir_1 = st.selectbox("Wind Direction D1-D30", opts["Wind Direction_D1_D30"])
wind_dir_2 = st.selectbox("Wind Direction D31-D60", opts["Wind Direction_D31_D60"])
wind_dir_3 = st.selectbox("Wind Direction D61-D90", opts["Wind Direction_D61_D90"])
wind_dir_4 = st.selectbox("Wind Direction D91-D120", opts["Wind Direction_D91_D120"])

humidity_1 = st.number_input("Humidity D1-D30", value=80.0)
humidity_2 = st.number_input("Humidity D31-D60", value=80.0)
humidity_3 = st.number_input("Humidity D61-D90", value=80.0)
humidity_4 = st.number_input("Humidity D91-D120", value=80.0)

# Prediction

if st.button("Predict Yield"):

    data = {
        "Hectares": hectares,
        "Agriblock": agriblock,
        "Variety": variety,
        "Soil Types": soil_types,
        "Nursery": nursery,
        "30DRain( in mm)": rain_30,
        "30_50DRain( in mm)": rain_30_50,
        "Min temp_D1_D30": min_temp_1,
        "Max temp_D1_D30": max_temp_1,
        "Min temp_D31_D60": min_temp_2,
        "Max temp_D31_D60": max_temp_2,
        "Max temp_D61_D90": max_temp_3,
        "Min temp_D91_D120": min_temp_4,
        "Inst Wind Speed_D1_D30(in Knots)": wind_speed_1,
        "Inst Wind Speed_D31_D60(in Knots)": wind_speed_2,
        "Inst Wind Speed_D61_D90(in Knots)": wind_speed_3,
        "Inst Wind Speed_D91_D120(in Knots)": wind_speed_4,
        "Wind Direction_D1_D30": wind_dir_1,
        "Wind Direction_D31_D60": wind_dir_2,
        "Wind Direction_D61_D90": wind_dir_3,
        "Wind Direction_D91_D120": wind_dir_4,
        "Relative Humidity_D1_D30": humidity_1,
        "Relative Humidity_D31_D60": humidity_2,
        "Relative Humidity_D61_D90": humidity_3,
        "Relative Humidity_D91_D120": humidity_4,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=data)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Yield: {result['predicted_yield_kg']} Kg")
        else:
            st.error(f"Prediction failed: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("FastAPI server is not running.")
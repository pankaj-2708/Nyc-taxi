import streamlit as st
from utility import *
from datetime import datetime

st.set_page_config(page_title="NYC taxi trip duration predictor", layout="wide")

st.title("Nyc taxi trip duration predictor")

cols = st.columns(4)

with cols[0]:
    vendor_id = st.selectbox("Vendor id", [1, 2])
    passenger_cnt = st.number_input("Passenger count", min_value=1)
with cols[1]:
    pickup_longitude = st.number_input("Pickup Longitude")
    pickup_latitude = st.number_input("Pickup Latitude")
with cols[2]:
    dropoff_longitude = st.number_input("Dropoff Longitude")
    dropoff_latitude = st.number_input("Dropoff Latitude")
with cols[3]:
    store_and_fwd_flag = st.selectbox("store_and_fwd_flag", ["Y", "N"])
    datetime_ = st.text_input("datetime of taking cab format", value=datetime.now())

cols = st.columns(5)
with cols[2]:
    pred = st.button("Predict")

if pred:
    params = {
        "vendor_id": vendor_id,
        "pickup_datetime": datetime_,
        "passenger_count": passenger_cnt,
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": pickup_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_latitude": dropoff_latitude,
        "store_and_fwd_flag": store_and_fwd_flag,
    }
    trip_duration = float(get_trip_duration(params))
    st.success(f"Predicted trip duration (in seconds ) is {round(trip_duration,2)}")

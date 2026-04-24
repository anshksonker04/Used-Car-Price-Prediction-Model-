import streamlit as st
import pandas as pd
import joblib
import datetime


model = joblib.load('car_price_predictor.pkl')


st.title("🚗 Vehicle Price Prediction App")
st.write("Enter the details of the vehicle to estimate its resale value.")

with st.form("prediction_form"):
    st.subheader("Vehicle Details")
    
    manufacturer = st.selectbox("Manufacturer", ["ford", "chevrolet", "toyota", "honda", "bmw"])
    year = st.number_input("Year of Manufacture", min_value=1990, max_value=datetime.date.today().year, value=2015)
    odometer = st.number_input("Mileage (Odometer)", min_value=0, max_value=500000, value=50000)
    fuel = st.selectbox("Fuel Type", ["gas", "diesel", "hybrid", "electric"])
    transmission = st.selectbox("Transmission", ["automatic", "manual", "other"])
    condition = st.selectbox("Condition", ["excellent", "good", "fair", "like new"])

    car_age = datetime.date.today().year - year

    submit_button = st.form_submit_button("Predict Price")


if submit_button:
    
    input_data = pd.DataFrame({
        'year': [year],
        'odometer': [odometer],
        'manufacturer': [manufacturer],
        'fuel': [fuel],
        'transmission': [transmission],
        'condition': [condition],
        'car_age': [car_age]
    })
    
   

    try:
        prediction = model.predict(input_data)
        st.success(f"### Estimated Price: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.write("Hint: Ensure the input features exactly match what the XGBoost model was trained on (including one-hot encoded columns).")

import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load the trained model and scaler
model = joblib.load("gold_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("Gold Price Prediction App")

# Sidebar for user input
st.sidebar.header("Input Options")
selected_date = st.sidebar.date_input("Select a Date", datetime.today())

# Instructions
st.write("""
### Instructions:
- Use the sidebar to select a date.
- Click "Predict" to see the predicted gold price.
""")

# Predict button
if st.sidebar.button("Predict"):
    try:
        # Preprocess the date for the model
        year = selected_date.year
        month = selected_date.month
        day = selected_date.day

        # Prepare input data for the model
        input_data = pd.DataFrame({"year": [year], "month": [month], "day": [day]})
        input_scaled = scaler.transform(input_data)  # Scale the input

        # Predict the gold price
        predicted_price = model.predict(input_scaled)
        st.success(f"The predicted gold price for {selected_date.strftime('%Y-%m-%d')} is ${predicted_price[0]:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

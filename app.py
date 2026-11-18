import streamlit as st
import joblib
import pandas as pd

# Page setup
st.set_page_config(page_title="Real Estate Price Predictor", page_icon="🏠")
st.title("🏠 Real Estate Price Prediction")
st.write("Enter the property details below to estimate its price in Indian Rupees (₹).")

# Load the saved model and columns
model = joblib.load("models/house_price_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# USD to INR conversion rate (Nov 2026)
USD_TO_INR = 93.45

def format_inr(amount):
    """Formats a number in Indian numbering style and Lakhs/Crores."""
    if amount >= 10000000:
        short_text = f"₹{amount / 10000000:.2f} Crore"
    elif amount >= 100000:
        short_text = f"₹{amount / 100000:.2f} Lakhs"
    else:
        short_text = f"₹{amount:,.2f}"
    
    # Standard Indian comma formatting (e.g., 1,97,38,233)
    integer_part, decimal_part = f"{amount:.2f}".split(".")
    if len(integer_part) > 3:
        last_three = integer_part[-3:]
        remaining = integer_part[:-3]
        groups = []
        while len(remaining) > 2:
            groups.append(remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.append(remaining)
        formatted_int = ",".join(reversed(groups)) + "," + last_three
    else:
        formatted_int = integer_part
        
    return f"₹{formatted_int}.{decimal_part} ({short_text})"

# Extract neighborhood names from saved columns
neighborhoods = sorted([
    col.replace("Neighborhood_", "")
    for col in feature_columns
    if col.startswith("Neighborhood_")
])

# Input form
col1, col2 = st.columns(2)

with col1:
    lot_area = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=100000, value=8500, step=500)
    gr_liv_area = st.number_input("Living Area (sq ft)", min_value=300, max_value=10000, value=1800, step=100)
    overall_qual = st.slider("Overall Quality (1 - 10)", 1, 10, 7)
    overall_cond = st.slider("Overall Condition (1 - 10)", 1, 10, 5)
    year_built = st.number_input("Year Built", min_value=1870, max_value=2026, value=2005)

with col2:
    full_bath = st.selectbox("Full Bathrooms", [0, 1, 2, 3, 4], index=2)
    bedroom = st.selectbox("Bedrooms Above Ground", [1, 2, 3, 4, 5, 6], index=2)
    tot_rms = st.slider("Total Rooms Above Ground", 2, 14, 7)
    garage_cars = st.selectbox("Garage Capacity (Cars)", [0, 1, 2, 3, 4], index=2)
    neighborhood = st.selectbox("Neighborhood", neighborhoods)

# Predict button
if st.button("Predict House Price", type="primary"):
    house_details = {
        'LotArea': lot_area,
        'OverallQual': overall_qual,
        'OverallCond': overall_cond,
        'YearBuilt': year_built,
        'GrLivArea': gr_liv_area,
        'FullBath': full_bath,
        'BedroomAbvGr': bedroom,
        'TotRmsAbvGrd': tot_rms,
        'GarageCars': garage_cars,
        'Neighborhood': neighborhood
    }

    input_df = pd.DataFrame([house_details])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    predicted_usd = model.predict(input_df)[0]
    predicted_inr = predicted_usd * USD_TO_INR
    
    st.success(f"### Estimated House Price: {format_inr(predicted_inr)}")
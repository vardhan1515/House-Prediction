import streamlit as st
import pandas as pd
import joblib

# Load the optimized model and feature names
model = joblib.load('optimized_house_price_model.pkl')
feature_names = joblib.load('feature_names.pkl')  # Load saved feature names

# User input for key features only
st.title("Simplified House Price Prediction App")

# Key numerical features
inputs = {
    'MSSubClass': st.number_input("Enter MSSubClass", value=20.0),
    'LotFrontage': st.number_input("Enter LotFrontage (e.g., 70)", value=70.0),
    'LotArea': st.number_input("Enter LotArea (e.g., 8500)", value=8500.0),
    'OverallQual': st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=5),
    'OverallCond': st.slider("Overall Condition (1-10)", min_value=1, max_value=10, value=5),
    'GrLivArea': st.number_input("Enter Above Ground Living Area (e.g., 1200)", value=1200.0),
    'GarageArea': st.number_input("Enter Garage Area (e.g., 400)", value=400.0),
}

# Dropdowns for categorical features
neighborhood = st.selectbox(
    "Select Neighborhood", 
    ['Blueste', 'CollgCr', 'Edwards', 'Gilbert', 'NWAmes', 'OldTown', 'Sawyer', 'Somerst']
)
sale_condition = st.selectbox(
    "Select Sale Condition", 
    ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial']
)

# Encode categorical variables dynamically
categorical_inputs = {
    f'Neighborhood_{neighborhood}': 1,
    f'SaleCondition_{sale_condition}': 1,
}

# Combine all inputs
for col in feature_names:
    if col not in inputs:
        # Set defaults for all other features
        inputs[col] = categorical_inputs.get(col, 0)

# Convert to DataFrame
input_data = pd.DataFrame([inputs])[feature_names]

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted House Price: ${prediction:,.2f}")

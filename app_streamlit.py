import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Function to load external CSS
def load_css(file_name):
    with open(file_name, 'r') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Load saved artifacts
@st.cache_resource
def load_model_artifacts():
    model = joblib.load('lightgbm_house_price_model.pkl')
    numerical_imputer = joblib.load('numerical_imputer.pkl')
    categorical_imputer = joblib.load('categorical_imputer.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, numerical_imputer, categorical_imputer, feature_names

# Add custom features to the dataset
def add_custom_features(data):
    if 'TotalBsmtSF' in data.columns and 'LotArea' in data.columns:
        data['TotalBsmtSF_LotArea'] = data['TotalBsmtSF'] / (data['LotArea'] + 1e-5)
    if 'GrLivArea' in data.columns and 'OverallQual' in data.columns:
        data['GrLivArea_OverallQual'] = data['GrLivArea'] * data['OverallQual']
    if 'GarageArea' in data.columns and 'GrLivArea' in data.columns:
        data['GarageArea_GrLivArea'] = data['GarageArea'] / (data['GrLivArea'] + 1e-5)
    if 'YearBuilt' in data.columns and 'YrSold' in data.columns:
        data['AgeAtSale'] = data['YrSold'] - data['YearBuilt']
    if 'YearRemodAdd' in data.columns and 'YrSold' in data.columns:
        data['YearsSinceRemodel'] = data['YrSold'] - data['YearRemodAdd']
    if 'TotalBsmtSF' in data.columns and '1stFlrSF' in data.columns and '2ndFlrSF' in data.columns:
        data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    return data

# Preprocess data for prediction
def preprocess_data(data, numerical_imputer, categorical_imputer, feature_names):
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    data[numerical_cols] = numerical_imputer.transform(data[numerical_cols])
    data[categorical_cols] = categorical_imputer.transform(data[categorical_cols])

    data = pd.get_dummies(data, drop_first=True)

    for col in set(feature_names) - set(data.columns):
        data[col] = 0

    data = data[feature_names]
    return data

# Load the CSS file
load_css('style.css')

# Streamlit app
st.title("House Price Prediction App")

st.write("Input house details below to predict the sale price:")

# Input form for single house features
with st.form("prediction_form"):
    OverallQual = st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=5)
    GrLivArea = st.number_input("Above grade (ground) living area (in sq ft)", value=1500)
    TotalBsmtSF = st.number_input("Total basement area (in sq ft)", value=1000)
    GarageArea = st.number_input("Garage area (in sq ft)", value=500)
    YearBuilt = st.number_input("Year Built", min_value=1800, max_value=2023, value=2000)
    YearRemodAdd = st.number_input("Year Remodeled", min_value=1800, max_value=2023, value=2010)
    LotArea = st.number_input("Lot Area (in sq ft)", value=8000)
    YrSold = st.number_input("Year Sold", min_value=2000, max_value=2023, value=2023)
    submit_button = st.form_submit_button("Predict Sale Price")

# Handle form submission
if submit_button:
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        "OverallQual": [OverallQual],
        "GrLivArea": [GrLivArea],
        "TotalBsmtSF": [TotalBsmtSF],
        "GarageArea": [GarageArea],
        "YearBuilt": [YearBuilt],
        "YearRemodAdd": [YearRemodAdd],
        "LotArea": [LotArea],
        "YrSold": [YrSold],
    })

    # Load model and preprocessing artifacts
    model, numerical_imputer, categorical_imputer, feature_names = load_model_artifacts()

    # Add custom features
    input_data = add_custom_features(input_data)

    # Preprocess data
    preprocessed_data = preprocess_data(input_data, numerical_imputer, categorical_imputer, feature_names)

    # Make predictions
    prediction = np.expm1(model.predict(preprocessed_data))[0]

    # Display the prediction
    st.write(f"### Predicted Sale Price: ${prediction:,.2f}")

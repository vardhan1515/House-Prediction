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
    if 'Id' in data.columns:
        ids = data['Id']
        data.drop(['Id'], axis=1, inplace=True)
    else:
        ids = None

    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    data[numerical_cols] = numerical_imputer.transform(data[numerical_cols])
    data[categorical_cols] = categorical_imputer.transform(data[categorical_cols])

    data = pd.get_dummies(data, drop_first=True)

    for col in set(feature_names) - set(data.columns):
        data[col] = 0

    data = data[feature_names]
    return data, ids

# Get dataset description for variables
def get_variable_description(var_name):
    descriptions = {
        "MSSubClass": "Identifies the type of dwelling involved in the sale.",
        "MSZoning": "Identifies the general zoning classification of the sale.",
        "LotFrontage": "Linear feet of street connected to property.",
        "LotArea": "Lot size in square feet.",
        "Street": "Type of road access to property.",
        "Alley": "Type of alley access to property.",
        "LotShape": "General shape of property.",
        "LandContour": "Flatness of the property.",
        "Utilities": "Type of utilities available.",
        "LotConfig": "Lot configuration.",
        "LandSlope": "Slope of property.",
        "Neighborhood": "Physical locations within Ames city limits.",
        "Condition1": "Proximity to various conditions.",
        "Condition2": "Proximity to various conditions (if more than one is present).",
        "BldgType": "Type of dwelling.",
        "HouseStyle": "Style of dwelling.",
        "OverallQual": "Rates the overall material and finish of the house.",
        "OverallCond": "Rates the overall condition of the house.",
        "YearBuilt": "Original construction date.",
        "YearRemodAdd": "Remodel date (same as construction date if no remodeling or additions).",
        "RoofStyle": "Type of roof.",
        "RoofMatl": "Roof material.",
        "Exterior1st": "Exterior covering on house.",
        "Exterior2nd": "Exterior covering on house (if more than one material).",
        "MasVnrType": "Masonry veneer type.",
        "MasVnrArea": "Masonry veneer area in square feet.",
        "ExterQual": "Evaluates the quality of the material on the exterior.",
        "ExterCond": "Evaluates the present condition of the material on the exterior.",
        "Foundation": "Type of foundation.",
        "TotalBsmtSF": "Total square feet of basement area.",
        "GrLivArea": "Above grade (ground) living area square feet.",
    }
    return descriptions.get(var_name, "No description available.")

# Load the CSS file
load_css('style.css')

# Streamlit app
st.title("House Price Prediction App")

st.write("Upload a CSV file containing house features to get predicted sale prices.")

# Display dataset description
if st.checkbox("Show Dataset Description"):
    for column in [
        "MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street",
        "Alley", "LotShape", "LandContour", "Utilities", "LotConfig",
        "LandSlope", "Neighborhood", "Condition1", "Condition2",
        "BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt",
        "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st",
        "Exterior2nd", "MasVnrType", "MasVnrArea", "ExterQual",
        "ExterCond", "Foundation", "TotalBsmtSF", "GrLivArea"
    ]:
        st.write(f"**{column}:** {get_variable_description(column)}")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read uploaded file
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:", data.head())

        # Load model and preprocessing artifacts
        model, numerical_imputer, categorical_imputer, feature_names = load_model_artifacts()

        # Add custom features
        data = add_custom_features(data)

        # Preprocess data
        preprocessed_data, ids = preprocess_data(data, numerical_imputer, categorical_imputer, feature_names)

        # Make predictions
        predictions = np.expm1(model.predict(preprocessed_data))

        # Display results
        result = pd.DataFrame({"Id": ids, "PredictedSalePrice": predictions})
        st.write("Predicted Sale Prices:")
        st.dataframe(result)

        # Download link
        csv = result.to_csv(index=False)
        st.download_button(label="Download Predictions", data=csv, file_name="house_price_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"An error occurred: {e}")

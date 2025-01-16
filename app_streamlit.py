import streamlit as st
import pandas as pd
import joblib
import base64
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="AI-Powered House Price Predictor", page_icon="🏡", layout="wide")

# Load background image
try:
    with open("image.png", "rb") as img_file:
        background_image = base64.b64encode(img_file.read()).decode()
except FileNotFoundError:
    st.error("Background image not found. Please ensure 'image.png' exists.")
    st.stop()

# Load CSS
try:
    with open("styles.css", "r") as css_file:
        css = css_file.read().replace("{background_image}", background_image)
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.error("CSS file not found. Please ensure 'styles.css' exists.")
    st.stop()

# Load model, feature names, and imputers
try:
    model = joblib.load('lightgbm_house_price_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    numerical_imputer = joblib.load('numerical_imputer.pkl')
    categorical_imputer = joblib.load('categorical_imputer.pkl')
except Exception as e:
    st.error("Error loading required files. Ensure the model, feature names, and imputers are available.")
    st.stop()

# Sidebar Section
st.sidebar.title("⚙️ Customize Your Property")
mo_sold = st.sidebar.slider("📅 Month Sold", 1, 12, 1, help="When was the house sold?")
yr_sold = st.sidebar.number_input("📅 Year Sold", 2000, 2025, 2025, help="Year of sale.")

# Main Section
st.title("🏡 AI-Powered House Price Predictor")
st.markdown("### Enter Property Details")

col1, col2 = st.columns(2)
inputs = {}
with col1:
    inputs['MSSubClass'] = st.selectbox("🏠 Building Class", [20, 30, 40], help="Type of dwelling.")
    inputs['LotFrontage'] = st.number_input("📏 Lot Frontage (ft)", value=70.0)
with col2:
    inputs['OverallQual'] = st.slider("🌟 Overall Quality", 1, 10, 5)
    inputs['GrLivArea'] = st.number_input("📏 Above Ground Living Area (sq. ft.)", value=1200.0)

# Make prediction
if st.button("🏡 Predict House Price"):
    prediction = model.predict(pd.DataFrame([inputs]))[0]
    st.markdown(f"### Predicted Price: ${np.expm1(prediction):,.2f}")

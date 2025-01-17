import streamlit as st
import pandas as pd
import joblib
import base64
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="AI-Powered House Price Predictor", page_icon="🏡", layout="wide")

# Load and encode background image
def load_background_image(image_path):
    """
    Load and encode the background image to Base64.
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Background image '{image_path}' not found. Please ensure it exists in the app directory.")
        st.stop()

# Load and apply CSS
def apply_css(css_path, background_image):
    """
    Load CSS file and replace the placeholder with the Base64 background image.
    """
    try:
        with open(css_path, "r") as css_file:
            css_content = css_file.read().replace("{background_image}", background_image)
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file '{css_path}' not found. Please ensure it exists in the app directory.")
        st.stop()

# Main logic
background_image_base64 = load_background_image("image.png")  # Path to your image
apply_css("styles.css", background_image_base64)  # Path to your CSS file

# Load the model, feature names, and imputers
try:
    model = joblib.load('lightgbm_house_price_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    numerical_imputer = joblib.load('numerical_imputer.pkl')
    categorical_imputer = joblib.load('categorical_imputer.pkl')
except Exception as e:
    st.error("Error loading required files. Ensure the model, feature names, and imputers are available.")
    st.stop()

# Initialize session state
if "reset_triggered" not in st.session_state:
    st.session_state.reset_triggered = False
if "history" not in st.session_state:
    st.session_state["history"] = []

# Reset and Clear Buttons
def reset_inputs():
    st.session_state.clear()
    st.session_state.reset_triggered = True

if st.sidebar.button("🔄 Reset All Inputs"):
    reset_inputs()

if st.sidebar.button("🗑 Clear Prediction History"):
    st.session_state["history"] = []

# Header Section
st.title("🏡 AI-Powered House Price Predictor")
st.markdown(
    """
    ## Your Real Estate Insights Hub 🌟  
    - **Accurately estimate property prices** using AI-powered algorithms.  
    - **Customize property details** for precise predictions.  
    - **Visualize key factors driving price estimates.**  
    """
)

# Feature Explanation Section
with st.expander("ℹ️ Key Features and Their Meanings", expanded=False):
    st.markdown("""...""")  # Truncated for brevity. Add your feature explanations here.

# Sidebar Section
st.sidebar.title("⚙️ Customize Your Property")
mo_sold = st.sidebar.slider("📅 Month Sold", min_value=1, max_value=12, value=1, help="When was the house sold?")
yr_sold = st.sidebar.number_input("📅 Year Sold", min_value=2000, max_value=2025, value=2025, help="Year of sale.")
show_history = st.sidebar.checkbox("📜 Show Prediction History")

# Main Input Section
st.markdown("### Enter Property Details")
col1, col2 = st.columns(2)

# Input fields
inputs = {}
with col1:
    inputs['MSSubClass'] = st.selectbox("🏠 Building Class (MSSubClass)", [...], help="Type of dwelling.")
    inputs['LotFrontage'] = st.number_input("📏 Lot Frontage (ft)", value=70.0)
    inputs['LotArea'] = st.number_input("📐 Lot Area (sq. ft.)", value=8500.0)
    inputs['BedroomAbvGr'] = st.number_input("🛌 Bedrooms Above Ground", value=3)
    inputs['GarageArea'] = st.number_input("🚗 Garage Area (sq. ft.)", value=400.0)

with col2:
    inputs['OverallQual'] = st.slider("🌟 Overall Quality", min_value=1, max_value=10, value=5)
    inputs['OverallCond'] = st.slider("🔧 Overall Condition", min_value=1, max_value=10, value=5)
    inputs['GrLivArea'] = st.number_input("📏 Above Ground Living Area (sq. ft.)", value=1200.0)
    inputs['FullBath'] = st.number_input("🛁 Full Bathrooms", value=2)
    inputs['HalfBath'] = st.number_input("🚻 Half Bathrooms", value=1)

# Neighborhood and Sale Information
col3, col4 = st.columns(2)
neighborhoods = ['Blueste', 'CollgCr', 'Edwards', 'Gilbert', 'NWAmes', 'OldTown', 'Sawyer', 'Somerst']
sale_conditions = ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial']

with col3:
    neighborhood = st.selectbox("🏘 Neighborhood", neighborhoods)
with col4:
    sale_condition = st.selectbox("📄 Sale Condition", sale_conditions)

categorical_inputs = {f'Neighborhood_{neighborhood}': 1, f'SaleCondition_{sale_condition}': 1}

# Combine inputs
for col in feature_names:
    inputs[col] = categorical_inputs.get(col, 0)
inputs['MoSold'] = mo_sold
inputs['YrSold'] = yr_sold

# Validate DataFrame Matches Model Input
input_data = pd.DataFrame([inputs])
missing_cols = set(feature_names) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0
input_data = input_data[feature_names]

# Predict Price
if st.button("🏡 Predict House Price"):
    try:
        prediction = model.predict(input_data)[0]
        prediction_price = np.expm1(prediction)
        st.markdown(f"### 🎯 Predicted Price: **${prediction_price:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")


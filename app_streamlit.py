import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Set page configuration
st.set_page_config(page_title="House Price Prediction App", page_icon="🏡", layout="wide")

# Encode the uploaded image as a base64 string for use as a background
with open("image.png", "rb") as img_file:
    encoded_string = base64.b64encode(img_file.read()).decode()

# Custom CSS for enhanced styling with a background image
page_bg_img = f'''
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{encoded_string}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Arial', sans-serif;
}}

[data-testid="stSidebar"] {{
    background-color: rgba(0, 0, 0, 0.7);
    color: white;
}}

[data-testid="stHeader"] {{
    background-color: rgba(0, 0, 0, 0.9);
    color: white;
}}

h1, h2, h3 {{
    color: white;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
    font-weight: bold;
}}

div.stButton > button {{
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 10px;
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
}}

div.stButton > button:hover {{
    background-color: #45a049;
    transform: scale(1.05);
    transition: all 0.3s ease;
}}

.stSlider > div {{
    color: white !important;
}}

label {{
    color: white !important;
    font-size: 14px;
    font-weight: bold;
}}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the optimized model and feature names
model = joblib.load('optimized_house_price_model.pkl')
feature_names = joblib.load('feature_names.pkl')

# Header Section
st.title("🏡 House Price Prediction App")
st.markdown(
    """
    ## Welcome to the Next-Gen House Price Predictor! 🎉  
    Harness the power of machine learning to estimate house prices. Perfect for buyers, sellers, and real estate enthusiasts.  
    - 🛠 Adjust the property details below.  
    - 🎯 Get real-time predictions.  
    - 📊 Gain insights into what drives house pricing.  
    """
)

# Sidebar Section
st.sidebar.title("⚙️ Customize Additional Features")
st.sidebar.markdown(
    """
    Fine-tune property details with optional features.
    """
)
mo_sold = st.sidebar.slider("📅 Month Sold", min_value=1, max_value=12, value=6, help="When was the house sold?")
yr_sold = st.sidebar.number_input("📅 Year Sold", min_value=2000, max_value=2025, value=2020, help="Year of sale.")

# Main Input Section
st.markdown("### Property Details")
col1, col2 = st.columns(2)

with col1:
    inputs = {
        'MSSubClass': st.number_input("🏠 Building Class (MSSubClass)", value=20.0, help="E.g., 20 = 1-story 1946 & newer."),
        'LotFrontage': st.number_input("📏 Lot Frontage (ft)", value=70.0, help="Length of the street connected to the property."),
        'LotArea': st.number_input("📐 Lot Area (sq. ft.)", value=8500.0, help="Total property area in square feet."),
    }

with col2:
    inputs['OverallQual'] = st.slider("🌟 Overall Quality", min_value=1, max_value=10, value=5, help="1 = Very Poor, 10 = Excellent.")
    inputs['OverallCond'] = st.slider("🔧 Overall Condition", min_value=1, max_value=10, value=5, help="1 = Very Poor, 10 = Excellent.")
    inputs['GrLivArea'] = st.number_input("📏 Above Ground Living Area (sq. ft.)", value=1200.0, help="Living area above ground.")

# Neighborhood and Sale Information
st.markdown("### Neighborhood and Sale Details")
col3, col4 = st.columns(2)

with col3:
    neighborhood = st.selectbox("🏘 Neighborhood", ['Blueste', 'CollgCr', 'Edwards', 'Gilbert', 'NWAmes', 'OldTown', 'Sawyer', 'Somerst'], help="Location of the property.")

with col4:
    sale_condition = st.selectbox("📄 Sale Condition", ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial'], help="Condition under which the sale was made.")

# Encode categorical variables dynamically
categorical_inputs = {f'Neighborhood_{neighborhood}': 1, f'SaleCondition_{sale_condition}': 1}

# Combine inputs
for col in feature_names:
    if col not in inputs:
        inputs[col] = categorical_inputs.get(col, 0)
inputs['MoSold'] = mo_sold
inputs['YrSold'] = yr_sold

# Convert to DataFrame
input_data = pd.DataFrame([inputs])[feature_names]

# Predict Price
if st.button("🏡 Predict House Price"):
    prediction = model.predict(input_data)[0]
    st.markdown(f"### 🎯 Predicted Price: **${prediction:,.2f}**")

    # Enhanced Feature Importance Visualization
    st.markdown("### 🔍 Key Factors Affecting the Price")
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)

    # Bar chart for feature importance
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=importance_df.head(10), x="Importance", y="Feature", palette="coolwarm")
    ax.set_title("Top 10 Features Influencing Prediction", fontsize=16, color='white')
    ax.set_xlabel("Importance", fontsize=14, color='white')
    ax.set_ylabel("Feature", fontsize=14, color='white')
    plt.xticks(color='white', fontsize=12)
    plt.yticks(color='white', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    st.pyplot(fig)

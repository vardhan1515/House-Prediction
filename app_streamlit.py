import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Set page configuration
st.set_page_config(page_title="House Price Prediction App", page_icon="🏡", layout="wide")

# Sidebar Section for User Customizations
st.sidebar.title("🌟 Customize Your Experience")
st.sidebar.markdown("Enhance your app experience with these options:")

# Theme Selection
theme = st.sidebar.radio("Choose Theme:", options=["Light", "Dark"], index=1)

# Background Upload
uploaded_file = st.sidebar.file_uploader("Upload a Custom Background", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Encode uploaded image as base64
    encoded_string = base64.b64encode(uploaded_file.read()).decode()
else:
    # Default background image
    with open("image.png", "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

# Color Palette Selection for Chart
color_palette = st.sidebar.selectbox(
    "Choose Chart Palette:", options=["coolwarm", "viridis", "plasma", "magma"], index=0
)

# Custom CSS based on user-selected theme
if theme == "Light":
    text_color = "#000000"
    shadow_color = "rgba(255, 255, 255, 0.8)"
    button_color = "#4CAF50"
else:
    text_color = "#FFFFFF"
    shadow_color = "rgba(0, 0, 0, 0.8)"
    button_color = "#FF6347"

page_bg_img = f'''
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{encoded_string}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Arial', sans-serif;
    color: {text_color};
    text-shadow: 1px 1px 2px {shadow_color};
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
    color: {text_color};
    text-shadow: 2px 2px 4px {shadow_color};
    font-weight: bold;
}}

.stSlider > div {{
    color: {text_color} !important;
}}

label {{
    color: {text_color} !important;
    font-size: 14px;
    font-weight: bold;
}}

div.stButton > button {{
    background-color: {button_color};
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
    transform: scale(1.05);
    transition: all 0.3s ease;
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
    with st.expander("🔍 Key Factors Affecting the Price", expanded=True):
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(data=importance_df.head(10), x="Importance", y="Feature", palette=color_palette)
        ax.set_title("Top 10 Features Influencing Prediction", fontsize=16, color=text_color)
        ax.set_xlabel("Importance", fontsize=14, color=text_color)
        ax.set_ylabel("Feature", fontsize=14, color=text_color)
        plt.xticks(color=text_color, fontsize=12)
        plt.yticks(color=text_color, fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig)

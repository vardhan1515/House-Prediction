import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Set page configuration
st.set_page_config(page_title="AI-Powered House Price Predictor", page_icon="🏡", layout="wide")

# Load the optimized model and feature names
model = joblib.load('optimized_house_price_model.pkl')
feature_names = joblib.load('feature_names.pkl')

# Encode the uploaded background image
with open("image.png", "rb") as img_file:
    encoded_string = base64.b64encode(img_file.read()).decode()

custom_css = f"""
<style>
/* Background Styling */
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpeg;base64,{encoded_string}");
    background-size: cover;
    background-position: center;
    font-family: 'Arial', sans-serif;
}}

/* Main Page Text Styling */
h1, h2, h3, h4, h5, h6, p, li {{
    color: black !important; /* Keep all text on the main page black */
}}

/* Sidebar Styling */
[data-testid="stSidebar"] {{
    background-color: rgba(0, 0, 0, 0.8);
    color: white !important; /* Sidebar text remains white */
    border-radius: 10px;
    padding: 10px;
}}

/* Sidebar Text Styling */
label, div.stSlider > div, input[type="number"], [data-testid="stNumberInput"] > div {{
    color: white !important; /* Set all text in sidebar to white */
}}

input[type="checkbox"] + span {{
    color: white !important; /* Checkbox text in white */
}}

/* Tooltip Styling */
[data-testid="stTooltipIcon"] svg {{
    color: white !important; /* Tooltip (?) icons are white */
}}

/* Buttons */
div.stButton > button {{
    background-color: #008CBA;
    color: white;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    border-radius: 10px;
}}

div.stButton > button:hover {{
    background-color: #005F73;
    transition: all 0.3s ease;
}}

/* Plus/Minus Buttons */
button.step-up {{
    background-color: #ADD8E6 !important; /* Light blue for plus button */
    color: white !important;
    border-radius: 5px;
}}

button.step-down {{
    background-color: #ADD8E6 !important; /* Light blue for minus button */
    color: white !important;
    border-radius: 5px;
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)





# Initialize prediction history
if "history" not in st.session_state:
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

# Sidebar Section
st.sidebar.title("⚙️ Customize Your Property")
mo_sold = st.sidebar.slider("📅 Month Sold", min_value=1, max_value=12, value=1, help="When was the house sold?")
yr_sold = st.sidebar.number_input("📅 Year Sold", min_value=2000, max_value=2025, value=2025, help="Year of sale.")
show_history = st.sidebar.checkbox("📜 Show Prediction History")

# Reset and Clear Buttons
if st.sidebar.button("🔄 Reset All Inputs"):
    st.session_state["history"] = []
    st.experimental_rerun()

if st.sidebar.button("🗑 Clear Prediction History"):
    st.session_state["history"] = []

# Main Input Section
st.markdown("### Enter Property Details")
col1, col2 = st.columns(2)

with col1:
    inputs = {
        'MSSubClass': st.number_input("🏠 Building Class (MSSubClass)", value=20.0, help="E.g., 20 = 1-story 1946 & newer."),
        'LotFrontage': st.number_input("📏 Lot Frontage (ft)", value=70.0, help="Length of the street connected to the property."),
        'LotArea': st.number_input("📐 Lot Area (sq. ft.)", value=8500.0, help="Total property area in square feet."),
        'BedroomAbvGr': st.number_input("🛌 Bedrooms Above Ground", value=3, help="Number of bedrooms above ground."),
        'GarageArea': st.number_input("🚗 Garage Area (sq. ft.)", value=400.0, help="Total area of the garage."),
    }

with col2:
    inputs['OverallQual'] = st.slider("🌟 Overall Quality", min_value=1, max_value=10, value=5, help="1 = Very Poor, 10 = Excellent.")
    inputs['OverallCond'] = st.slider("🔧 Overall Condition", min_value=1, max_value=10, value=5, help="1 = Very Poor, 10 = Excellent.")
    inputs['GrLivArea'] = st.number_input("📏 Above Ground Living Area (sq. ft.)", value=1200.0, help="Living area above ground.")
    inputs['FullBath'] = st.number_input("🛁 Full Bathrooms", value=2, help="Number of full bathrooms.")
    inputs['HalfBath'] = st.number_input("🚻 Half Bathrooms", value=1, help="Number of half bathrooms.")

# Neighborhood and Sale Information
st.markdown("### Neighborhood and Sale Details")
col3, col4 = st.columns(2)

with col3:
    neighborhood = st.selectbox(
        "🏘 Neighborhood",
        ['Blueste', 'CollgCr', 'Edwards', 'Gilbert', 'NWAmes', 'OldTown', 'Sawyer', 'Somerst'],
        help="Location of the property.",
    )

with col4:
    sale_condition = st.selectbox(
        "📄 Sale Condition",
        ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial'],
        help="Condition under which the sale was made.",
    )

# Encode categorical variables dynamically
categorical_inputs = {f'Neighborhood_{neighborhood}': 1, f'SaleCondition_{sale_condition}': 1}

# Combine inputs
for col in feature_names:
    if col not in inputs:
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
    prediction = model.predict(input_data)[0]
    st.session_state["history"].append({"Inputs": inputs, "Prediction": prediction})

    st.markdown(f"### 🎯 Predicted Price: **${prediction:,.2f}**")

    # Feature Importance Chart
    st.markdown("### 🔍 Top Influential Features")
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df.head(10), x="Importance", y="Feature", palette="viridis")
    ax.set_title("Top 10 Features Influencing Prediction", fontsize=16)
    ax.set_xlabel("Importance", fontsize=14)
    ax.set_ylabel("Feature", fontsize=14)
    st.pyplot(fig)

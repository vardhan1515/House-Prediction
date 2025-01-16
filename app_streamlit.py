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

# Encode the uploaded image for the background
with open("image.png", "rb") as img_file:
    encoded_string = base64.b64encode(img_file.read()).decode()

# Custom CSS for styling
custom_css = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{encoded_string}");
    background-size: cover;
    background-position: center;
    font-family: 'Arial', sans-serif;
}}

[data-testid="stSidebar"] {{
    background-color: rgba(255, 255, 255, 0.9);
    border-radius: 10px;
    padding: 10px;
}}

[data-testid="stHeader"] {{
    background-color: rgba(255, 255, 255, 0.9);
}}

h1, h2, h3 {{
    color: black;
    font-weight: bold;
}}

label {{
    color: black !important;
    font-size: 14px;
    font-weight: bold;
}}

.stSlider > div {{
    color: black !important;
}}

div.stButton > button {{
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    border-radius: 10px;
}}

div.stButton > button:hover {{
    background-color: #45a049;
    transition: all 0.3s ease;
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
    ## Welcome to Your Personalized Real Estate Tool! 🎉  
    - Estimate property prices with precision using AI.
    - Explore insights into factors driving property values.
    - Save and analyze predictions for your needs.
    """
)

# Sidebar Section
st.sidebar.title("⚙️ Configure Your Settings")
mo_sold = st.sidebar.slider("📅 Month Sold", min_value=1, max_value=12, value=1, help="When was the house sold?")
yr_sold = st.sidebar.number_input("📅 Year Sold", min_value=2000, max_value=2025, value=2025, help="Year of sale.")
show_history = st.sidebar.checkbox("📜 Show Prediction History")

# Reset Button
if st.sidebar.button("🔄 Reset All Inputs"):
    st.session_state["history"] = []
    st.experimental_rerun()

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

# Convert to DataFrame
input_data = pd.DataFrame([inputs])[feature_names]

# Predict Price
if st.button("🏡 Predict House Price"):
    prediction = model.predict(input_data)[0]
    st.session_state["history"].append({"Inputs": inputs, "Prediction": prediction})

    st.markdown(f"### 🎯 Predicted Price: **${prediction:,.2f}**")

    # Advanced Insights Section
    st.markdown("### 📊 Advanced Insights")
    # Correlation Heatmap
    st.markdown("#### 🔥 Feature Correlation Heatmap")
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(importance_df.head(10).set_index("Feature").T, annot=True, cmap="coolwarm", cbar=True)
    st.pyplot(fig)

    # Distribution Plot
    st.markdown("#### 📈 Distribution of Predicted Prices")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=pd.DataFrame({"Prediction": [prediction]}), x="Prediction", kde=True, color="blue")
    ax.set_title("Distribution of Predicted Prices", fontsize=16)
    st.pyplot(fig)

# Show Prediction History
if show_history and len(st.session_state["history"]) > 0:
    st.markdown("### 🕒 Prediction History")
    df_history = pd.DataFrame(st.session_state["history"])
    st.dataframe(df_history)

    # Save Predictions
    if st.button("💾 Save Predictions to CSV"):
        df_history.to_csv("predictions_history.csv", index=False)
        st.success("Prediction history saved successfully!")

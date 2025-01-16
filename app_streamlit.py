import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Set page configuration
st.set_page_config(page_title="Enhanced House Price Prediction App", page_icon="🏡", layout="wide")

# Encode the uploaded image as a base64 string for use as a background
with open("image.png", "rb") as img_file:
    encoded_string = base64.b64encode(img_file.read()).decode()

# Custom CSS
page_bg_img = f'''
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{encoded_string}");
    background-size: cover;
    background-position: center;
    font-family: 'Arial', sans-serif;
    color: #000000;
}}

[data-testid="stSidebar"] {{
    background-color: rgba(0, 0, 0, 0.7);
    color: white;
}}

h1, h2, h3 {{
    color: #000000;
    font-weight: bold;
}}

label {{
    color: #000000 !important;
    font-size: 14px;
    font-weight: bold;
}}

.stSlider > div {{
    color: #000000 !important;
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
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the optimized model and feature names
model = joblib.load('optimized_house_price_model.pkl')
feature_names = joblib.load('feature_names.pkl')

# Initialize prediction history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Header Section
st.title("🏡 Enhanced House Price Prediction App")
st.markdown(
    """
    ## Welcome! 🎉  
    Harness the power of machine learning to estimate house prices. Perfect for buyers, sellers, and real estate enthusiasts.  
    - 🛠 Adjust the property details below.  
    - 🎯 Get real-time predictions.  
    - 📊 Gain insights into what drives house pricing.  
    """
)

# Sidebar Section
st.sidebar.title("⚙️ Customize Your Experience")
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

    # Enhanced Feature Importance Visualization
    st.markdown("### 🔍 Key Factors Affecting the Price")
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=importance_df.head(10), x="Importance", y="Feature", palette="coolwarm")
    ax.set_title("Top 10 Features Influencing Prediction", fontsize=16)
    ax.set_xlabel("Importance", fontsize=14)
    ax.set_ylabel("Feature", fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    st.pyplot(fig)

    # Heatmap for Feature Correlation
    st.markdown("### 🔥 Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(importance_df.head(10).set_index("Feature").T, annot=True, cmap="coolwarm", cbar=True)
    st.pyplot(fig)

    # Save predictions
    if st.button("💾 Save Predictions"):
        df_history = pd.DataFrame(st.session_state["history"])
        df_history.to_csv("predictions_history.csv", index=False)
        st.success("Prediction history saved successfully!")

# Display Prediction History
if st.sidebar.checkbox("📜 Show Prediction History"):
    if len(st.session_state["history"]) > 0:
        st.markdown("### 🕒 Prediction History")
        df_history = pd.DataFrame(st.session_state["history"])
        st.dataframe(df_history)
    else:
        st.markdown("No predictions made yet.")

# Reset Button
if st.sidebar.button("🔄 Reset"):
    st.session_state["history"] = []
    st.experimental_rerun()

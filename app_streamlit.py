import streamlit as st
import pandas as pd
import joblib
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="AI-Powered House Price Predictor", page_icon="🏡", layout="wide")

# Load background image and encode it
if "background_image" not in st.session_state:
    with open("image.png", "rb") as img_file:
        st.session_state["background_image"] = base64.b64encode(img_file.read()).decode()

# Link the external CSS
with open("styles.css", "r") as css_file:
    css = css_file.read().replace("{background_image}", st.session_state["background_image"])
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Load the optimized model and feature names
try:
    model = joblib.load('optimized_house_price_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
except Exception as e:
    st.error("Error loading model or feature names. Please ensure the files are in the correct location.")
    st.stop()

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
    # Reset all session state variables to their default values
    st.session_state.clear()  # Clears all session state variables
    
    # Optional: Add default values to session state if necessary
    st.session_state["history"] = []
    st.session_state["some_other_input"] = "default_value"  # Example for a specific input

    # Force a page rerun
    st.experimental_rerun()

# Feature Explanation Section
with st.expander("ℹ️ About the Features", expanded=False):
    st.markdown("""
    - **MSSubClass**: Numeric identifier for the type of dwelling. For example, 20 = 1-story, 1946 & newer.
    - **LotFrontage**: Linear feet of street connected to the property.
    - **LotArea**: Lot size in square feet.
    - **OverallQual**: Rates the overall material and finish of the house on a scale of 1 (poor) to 10 (excellent).
    - **OverallCond**: Rates the overall condition of the house on a scale of 1 (poor) to 10 (excellent).
    - **GrLivArea**: Above-ground (living) area in square feet.
    - **FullBath**: Number of full bathrooms (with a sink, toilet, and shower/tub) above ground.
    - **HalfBath**: Number of half bathrooms (sink and toilet only) above ground.
    - **BedroomAbvGr**: Number of bedrooms above ground.
    - **GarageArea**: Size of the garage in square feet.
    - **Neighborhood**: Location within the city, which can influence price.
    - **SaleCondition**: Condition of the sale, such as "Normal" or "Abnormal."
    """)

# Main Input Section
st.markdown("### Enter Property Details")
col1, col2 = st.columns(2)

with col1:
    inputs = {
        'MSSubClass': st.number_input("🏠 Building Class (MSSubClass)", value=20.0, help="E.g., 20 = 1-story, 1946 & newer."),
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

neighborhoods = ['Blueste', 'CollgCr', 'Edwards', 'Gilbert', 'NWAmes', 'OldTown', 'Sawyer', 'Somerst']
sale_conditions = ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial']

with col3:
    neighborhood = st.selectbox("🏘 Neighborhood", neighborhoods, help="Location of the property.")

with col4:
    sale_condition = st.selectbox("📄 Sale Condition", sale_conditions, help="Condition under which the sale was made.")

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
    ax.set_xlabel("Importance Score", fontsize=14)
    ax.set_ylabel("Features", fontsize=14)
    st.pyplot(fig)

# Show Prediction History
if show_history:
    with st.expander("📜 Prediction History"):
        for entry in st.session_state["history"]:
            st.write(f"Inputs: {entry['Inputs']}")
            st.write(f"Prediction: ${entry['Prediction']:,.2f}")

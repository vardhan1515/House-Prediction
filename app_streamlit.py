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

# Load the optimized model, feature names, and imputers
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
    st.session_state.clear()  # Clear all session state
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
with st.expander("ℹ️ About the Features", expanded=False):
    st.markdown("""
    ### **Key Features and Their Meanings**
    - **MSSubClass**: Identifies the type of dwelling involved in the sale.  
      - Example: `20` = 1-Story, 1946 & newer; `30` = 1-Story, 1945 & older.  
    - **MSZoning**: General zoning classification (e.g., RL = Residential Low Density).  
    - **LotFrontage**: Linear feet of street connected to the property.  
    - **OverallQual**: Rates the overall material and finish of the house (1 to 10).  
    - **OverallCond**: Rates the overall condition of the house (1 to 10).  
    - **GrLivArea**: Above-ground living area in square feet.  
    - **Neighborhood**: Physical location within Ames city limits (e.g., `Blueste`, `CollgCr`).  
    - **SaleCondition**: Condition of the sale (e.g., `Normal`, `Abnorml`).  
    """)

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
    inputs['MSSubClass'] = st.selectbox(
        "🏠 Building Class (MSSubClass)",
        [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190],
        help="Type of dwelling involved in the sale."
    )
    inputs['LotFrontage'] = st.number_input("📏 Lot Frontage (ft)", value=70.0, help="Length of the street connected to the property.")
    inputs['LotArea'] = st.number_input("📐 Lot Area (sq. ft.)", value=8500.0, help="Total property area in square feet.")
    inputs['BedroomAbvGr'] = st.number_input("🛌 Bedrooms Above Ground", value=3, help="Number of bedrooms above ground.")
    inputs['GarageArea'] = st.number_input("🚗 Garage Area (sq. ft.)", value=400.0, help="Total area of the garage.")

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
    prediction_price = np.expm1(prediction)  # Reverse log transformation if used
    st.session_state["history"].append({"Inputs": inputs, "Prediction": prediction_price})
    st.markdown(f"### 🎯 Predicted Price: **${prediction_price:,.2f}**")

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

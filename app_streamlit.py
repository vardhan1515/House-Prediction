import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Set page configuration (MUST be the first Streamlit command)
st.set_page_config(page_title="House Price Prediction App", page_icon="🏡")

# Load the optimized model and feature names
model = joblib.load('optimized_house_price_model.pkl')
feature_names = joblib.load('feature_names.pkl')  # Load saved feature names

# Page title and introduction
st.title("🏡 House Price Prediction App")
st.markdown(
    """
    **Welcome!**  
    This app uses advanced machine learning to predict house prices based on the features you provide.  
    Adjust the sliders and inputs below, and click **Predict** to get an estimate!  
    """
)

# Sidebar for optional features
st.sidebar.title("🏘 Optional Features")
st.sidebar.markdown(
    "You can tweak additional property details here to refine the prediction."
)
mo_sold = st.sidebar.slider("Month Sold", min_value=1, max_value=12, value=6, help="Month in which the house was sold.")
yr_sold = st.sidebar.number_input("Year Sold", min_value=2000, max_value=2025, value=2020, help="Year in which the house was sold.")

# Property features section
st.markdown("### Core Property Features")
col1, col2 = st.columns(2)

with col1:
    inputs = {
        'MSSubClass': st.number_input("🛠 Building Class (MSSubClass)", value=20.0, help="E.g., 20 = 1-story 1946 & newer."),
        'LotFrontage': st.number_input("📏 Lot Frontage (in feet)", value=70.0, help="Linear feet of street connected to the property."),
        'LotArea': st.number_input("📐 Lot Area (sq. ft.)", value=8500.0, help="Total size of the lot in square feet."),
    }

with col2:
    inputs['OverallQual'] = st.slider(
        "🌟 Overall Quality", min_value=1, max_value=10, value=5, 
        help="Rate the overall quality of the property (1 = Very Poor, 10 = Excellent)."
    )
    inputs['OverallCond'] = st.slider(
        "🔧 Overall Condition", min_value=1, max_value=10, value=5, 
        help="Rate the overall condition of the property (1 = Very Poor, 10 = Excellent)."
    )
    inputs['GrLivArea'] = st.number_input(
        "📏 Above Ground Living Area (sq. ft.)", value=1200.0, 
        help="Total square feet of living area above ground."
    )

# Dropdowns for categorical features
st.markdown("### Neighborhood and Sale Information")
neighborhood = st.selectbox(
    "🏘 Neighborhood", 
    ['Blueste', 'CollgCr', 'Edwards', 'Gilbert', 'NWAmes', 'OldTown', 'Sawyer', 'Somerst'],
    help="The neighborhood where the property is located."
)
sale_condition = st.selectbox(
    "📄 Sale Condition", 
    ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial'],
    help="Condition of the sale."
)

# Encode categorical variables dynamically
categorical_inputs = {
    f'Neighborhood_{neighborhood}': 1,
    f'SaleCondition_{sale_condition}': 1,
}

# Combine all inputs
for col in feature_names:
    if col not in inputs:
        # Set defaults for all other features
        inputs[col] = categorical_inputs.get(col, 0)

inputs['MoSold'] = mo_sold
inputs['YrSold'] = yr_sold

# Convert to DataFrame
input_data = pd.DataFrame([inputs])[feature_names]

# Predict the house price
if st.button("🏡 Predict Price"):
    prediction = model.predict(input_data)[0]
    st.markdown(
        f"""
        ### 🎯 Predicted House Price:  
        **${prediction:,.2f}**  
        """
    )

    # Feature importance
    st.markdown("### 🔍 Feature Importance")
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["Feature"].head(10), importance_df["Importance"].head(10))
    ax.set_title("Top 10 Features Affecting Prediction")
    st.pyplot(fig)

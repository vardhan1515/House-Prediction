# Import required libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

# Suppress warnings globally
warnings.filterwarnings('ignore')

# Limit threads for subprocess issues
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Add custom features
def add_custom_features(data):
    if 'TotalBsmtSF' in data.columns and 'LotArea' in data.columns:
        data['TotalBsmtSF_LotArea'] = data['TotalBsmtSF'] / (data['LotArea'] + 1e-5)
    if 'GrLivArea' in data.columns and 'OverallQual' in data.columns:
        data['GrLivArea_OverallQual'] = data['GrLivArea'] * data['OverallQual']
    if 'GarageArea' in data.columns and 'GrLivArea' in data.columns:
        data['GarageArea_GrLivArea'] = data['GarageArea'] / (data['GrLivArea'] + 1e-5)
    if 'YearBuilt' in data.columns and 'YrSold' in data.columns:
        data['AgeAtSale'] = data['YrSold'] - data['YearBuilt']
    if 'YearRemodAdd' in data.columns and 'YrSold' in data.columns:
        data['YearsSinceRemodel'] = data['YrSold'] - data['YearRemodAdd']
    if 'TotalBsmtSF' in data.columns and '1stFlrSF' in data.columns and '2ndFlrSF' in data.columns:
        data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    return data

# Preprocess the data
def preprocess_data(data, numerical_imputer=None, categorical_imputer=None, fit_imputers=False, feature_names=None):
    if 'Id' in data.columns:
        ids = data['Id']  # Save the Id column for merging later
        data.drop(['Id'], axis=1, inplace=True)
    else:
        ids = None

    # Separate numerical and categorical columns
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Handle missing values
    if fit_imputers:
        numerical_imputer = SimpleImputer(strategy='median')
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        data[numerical_cols] = numerical_imputer.fit_transform(data[numerical_cols])
        data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])
    else:
        data[numerical_cols] = numerical_imputer.transform(data[numerical_cols])
        data[categorical_cols] = categorical_imputer.transform(data[categorical_cols])

    # One-hot encode categorical variables
    data = pd.get_dummies(data, drop_first=True)

    # Align columns with training data
    if feature_names is not None:
        for col in set(feature_names) - set(data.columns):
            data[col] = 0
        data = data[feature_names]  # Ensure column order matches training data

    return data, ids, numerical_imputer, categorical_imputer

# Perform residual analysis
def residual_analysis(y_actual, y_pred):
    residuals = y_actual - y_pred

    # Plot residuals vs actual values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_actual, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residual Analysis: Residuals vs Actual Sale Prices")
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Residuals")
    plt.savefig("residual_analysis.png")
    print("Residual analysis saved as 'residual_analysis.png'.")

    # Plot residual distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title("Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.savefig("residual_distribution.png")
    print("Residual distribution saved as 'residual_distribution.png'.")

# Main function
def main():
    # File paths
    train_path = 'house_train.csv'
    test_path = 'house_test.csv'
    saleprice_test_path = 'saleprices_test.csv'

    # Load datasets
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    saleprice_data = pd.read_csv(saleprice_test_path)

    # Preprocess training data
    y_train = np.log1p(train_data['SalePrice'])  # Apply log transformation to the target
    train_data.drop(['SalePrice'], axis=1, inplace=True)

    # Add custom features
    train_data = add_custom_features(train_data)

    # Preprocess train_data
    train_data, _, numerical_imputer, categorical_imputer = preprocess_data(train_data, fit_imputers=True)
    y_train = y_train.loc[train_data.index]  # Align y_train with train_data

    # Save feature names for deployment
    feature_names = train_data.columns.tolist()
    with open('feature_names.pkl', 'wb') as f:
        joblib.dump(feature_names, f)
    print("Feature names saved as 'feature_names.pkl'.")

    # Save imputers for deployment
    with open('numerical_imputer.pkl', 'wb') as f:
        joblib.dump(numerical_imputer, f)
    with open('categorical_imputer.pkl', 'wb') as f:
        joblib.dump(categorical_imputer, f)
    print("Imputers saved as 'numerical_imputer.pkl' and 'categorical_imputer.pkl'.")

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(train_data, y_train, test_size=0.2, random_state=42)

    # Define the LightGBM model
    lgb_model = LGBMRegressor(
        random_state=42,
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=8,
        num_leaves=50,
        min_child_samples=5,
        reg_alpha=0.1,
        reg_lambda=0.5,
        verbose=-1,
        n_jobs=1  # Avoid subprocess issues
    )

    # Train the model with early stopping
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(period=100)
        ]
    )
    print("LightGBM model training completed.")

    # Save the trained model
    joblib.dump(lgb_model, 'lightgbm_house_price_model.pkl')
    print("Trained LightGBM model saved as 'lightgbm_house_price_model.pkl'.")

    # Evaluate the model
    y_pred_val = np.expm1(lgb_model.predict(X_val))  # Reverse the log transformation for predictions
    y_val_actual = np.expm1(y_val)  # Reverse the log transformation for actual values
    rmse = sqrt(mean_squared_error(y_val_actual, y_pred_val))
    r2 = r2_score(y_val_actual, y_pred_val)
    print(f"LightGBM Model RMSE: {rmse}")
    print(f"LightGBM Model R-squared: {r2}")

    # Perform residual analysis
    residual_analysis(y_val_actual, y_pred_val)

    # Preprocess test data
    test_data = add_custom_features(test_data)
    test_data, test_ids, _, _ = preprocess_data(test_data, numerical_imputer, categorical_imputer, feature_names=feature_names)

    # Make predictions
    predictions = np.expm1(lgb_model.predict(test_data))  # Reverse log transformation for predictions

    # Merge predictions with saleprice_test.csv
    result = pd.DataFrame({'Id': test_ids, 'PredictedSalePrice': predictions})
    final_output = saleprice_data.merge(result, on='Id', how='left')
    final_output.to_csv('merged_test_predictions.csv', index=False)
    print("Predictions saved to 'merged_test_predictions.csv'.")

if __name__ == "__main__":
    main()

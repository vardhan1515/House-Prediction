# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import joblib
import os

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
    y_train = train_data['SalePrice']
    train_data.drop(['SalePrice'], axis=1, inplace=True)
    train_data, _, numerical_imputer, categorical_imputer = preprocess_data(train_data, fit_imputers=True)

    # Save feature names
    feature_names = train_data.columns.tolist()
    with open('feature_names.pkl', 'wb') as f:
        joblib.dump(feature_names, f)
    print("Feature names saved as 'feature_names.pkl'.")

    # Train the model
    X_train, X_val, y_train, y_val = train_test_split(train_data, y_train, test_size=0.2, random_state=42)
    base_model = GradientBoostingRegressor(random_state=42)
    base_model.fit(X_train, y_train)
    print("Base model training completed.")

    # Save base model
    joblib.dump(base_model, 'base_house_price_model.pkl')
    print("Base model saved as 'base_house_price_model.pkl'.")

    # Evaluate base model
    y_pred_base = base_model.predict(X_val)
    rmse_base = sqrt(mean_squared_error(y_val, y_pred_base))
    r2_base = r2_score(y_val, y_pred_base)
    print(f"Base Model RMSE: {rmse_base}")
    print(f"Base Model R-squared: {r2_base}")

    # Optimize model with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
    }

    grid_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42), param_grid, 
        cv=3, scoring='neg_root_mean_squared_error', verbose=1
    )
    grid_search.fit(X_train, y_train)
    print("Grid search completed.")
    print("Best Parameters:", grid_search.best_params_)

    # Final model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'optimized_house_price_model.pkl')
    print("Optimized model saved as 'optimized_house_price_model.pkl'.")

    # Evaluate optimized model
    y_pred_best = best_model.predict(X_val)
    rmse_best = sqrt(mean_squared_error(y_val, y_pred_best))
    r2_best = r2_score(y_val, y_pred_best)
    print(f"Optimized Model RMSE: {rmse_best}")
    print(f"Optimized Model R-squared: {r2_best}")

    # Preprocess test data
    test_data, test_ids, _, _ = preprocess_data(test_data, numerical_imputer, categorical_imputer, feature_names=feature_names)

    # Make predictions
    predictions = best_model.predict(test_data)

    # Merge predictions with saleprice_test.csv
    result = pd.DataFrame({'Id': test_ids, 'PredictedSalePrice': predictions})
    final_output = saleprice_data.merge(result, on='Id', how='left')
    final_output.to_csv('merged_test_predictions.csv', index=False)
    print("Predictions saved to 'merged_test_predictions.csv'.")

    # Output model performance metrics
    with open('model_performance.txt', 'w') as f:
        f.write(f"Base Model RMSE: {rmse_base}\n")
        f.write(f"Base Model R-squared: {r2_base}\n")
        f.write(f"Optimized Model RMSE: {rmse_best}\n")
        f.write(f"Optimized Model R-squared: {r2_best}\n")
    print("Model performance metrics saved to 'model_performance.txt'.")

if __name__ == "__main__":
    main()

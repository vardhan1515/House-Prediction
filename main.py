# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib

# Load the dataset
data = pd.read_csv('house_train.csv')

# Drop irrelevant columns
data.drop(['Id'], axis=1, inplace=True)

# Handle missing values
# Separate numerical and categorical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Impute missing values for numerical columns with median
numerical_imputer = SimpleImputer(strategy='median')
data[numerical_cols] = numerical_imputer.fit_transform(data[numerical_cols])

# Impute missing values for categorical columns with the most frequent value
categorical_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

# Verify no missing values remain
assert data.isnull().sum().sum() == 0, "Missing values still present!"

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Save feature names for deployment
feature_names = data.drop('SalePrice', axis=1).columns.tolist()
with open('feature_names.pkl', 'wb') as f:
    joblib.dump(feature_names, f)
print("Feature names saved.")

# Separate features and target
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Regressor
base_model = GradientBoostingRegressor(random_state=42)

# Train the base model
base_model.fit(X_train, y_train)
print("Base model training completed.")

# Evaluate the base model
y_pred_base = base_model.predict(X_test)
rmse_base = sqrt(mean_squared_error(y_test, y_pred_base))
print(f"Base Model RMSE: {rmse_base}")

# Save the base model
joblib.dump(base_model, 'base_house_price_model.pkl')
print("Base model saved as 'base_house_price_model.pkl'")

# Optimize the model using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42), param_grid, 
    cv=5, scoring='neg_root_mean_squared_error', verbose=1
)
grid_search.fit(X_train, y_train)
print("Optimization completed.")
print("Best Parameters:", grid_search.best_params_)

# Train the best model
best_model = grid_search.best_estimator_

# Evaluate the optimized model
y_pred_best = best_model.predict(X_test)
rmse_best = sqrt(mean_squared_error(y_test, y_pred_best))
print(f"Optimized Model RMSE: {rmse_best}")

# Save the optimized model
joblib.dump(best_model, 'optimized_house_price_model.pkl')
print("Optimized model saved as 'optimized_house_price_model.pkl'")

# Test the best model with a sample input
# Reindex the sample input to match feature names
sample_input_df = X_test.iloc[0:1].reindex(columns=feature_names, fill_value=0)
sample_prediction = best_model.predict(sample_input_df)
print(f"Sample input: {sample_input_df}")
print(f"Predicted house price: ${sample_prediction[0]:,.2f}")

# Feature importance
feature_importances = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nTop 10 Features by Importance:")
print(feature_importances.head(10))

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import os

# --- Model Training Script ---
# This script trains a model to suggest discount percentages for perishable goods.

# Load the dataset
try:
    df = pd.read_csv('perishable_goods_pricing_dataset.csv')
except FileNotFoundError:
    print("Error: 'perishable_goods_pricing_dataset.csv' not found.")
    print("Please make sure you have the dataset in the same directory.")
    # As a fallback for demonstration, creating a dummy dataframe
    data = {
        'product_type': ['Fruit', 'Vegetable', 'Dairy', 'Fruit', 'Bakery'],
        'days_to_expiry': [3, 2, 5, 1, 1],
        'stock_quantity': [100, 150, 200, 50, 80],
        'average_daily_sales': [20, 30, 15, 10, 40],
        'cost_price': [1.0, 0.5, 2.0, 1.2, 1.5],
        'selling_price': [1.5, 0.8, 2.5, 1.8, 2.0],
        'season': ['Summer', 'Summer', 'All-Season', 'Summer', 'All-Season'],
        'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'profit_margin': [0.5, 0.3, 0.5, 0.6, 0.5],
        'historical_discount_given': [0, 1, 0, 1, 0],
        'units_expired_last_week': [5, 2, 1, 8, 3],
        'suggested_discount_percentage': [10, 15, 5, 25, 20]
    }
    df = pd.DataFrame(data)
    df.to_csv('perishable_goods_pricing_dataset.csv', index=False)
    print("Created a dummy 'perishable_goods_pricing_dataset.csv'.")


# Handle categorical features
categorical_cols = ['product_type', 'season', 'day_of_week']
label_encoders = {}

print("Encoding categorical features...")
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
print("Encoding complete.")

# Feature and target selection
features = [
    'product_type', 'days_to_expiry', 'stock_quantity',
    'average_daily_sales', 'cost_price', 'selling_price', 'season',
    'day_of_week', 'profit_margin', 'historical_discount_given',
    'units_expired_last_week'
]
target = 'suggested_discount_percentage'

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost Regressor model
print("Training the XGBoost Regressor model...")
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# Evaluate the model (optional)
score = model.score(X_test, y_test)
print(f"Model R^2 score: {score:.2f}")


# Save the trained model and the label encoders
print("Saving model and encoders...")
joblib.dump(model, 'ml_model.pkl')
joblib.dump(label_encoders, 'encoders.pkl')
print("Model and encoders saved successfully as 'ml_model.pkl' and 'encoders.pkl'.")


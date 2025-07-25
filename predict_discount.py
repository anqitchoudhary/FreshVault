import sys
import json
import joblib
import numpy as np
import pandas as pd


# --- Prediction Script ---
# This script loads the trained model and encoders to predict discount percentages.

def predict_discount(input_data):
    """
    Loads the trained model and encoders to predict the discount for given item data.
    """
    # Load the saved model and encoders
    try:
        model = joblib.load("ml_model.pkl")
        encoders = joblib.load("encoders.pkl")
    except FileNotFoundError:
        return {"error": "Model or encoders not found. Please run model.py first."}

    # Convert input JSON to a pandas DataFrame
    input_df = pd.DataFrame(input_data, index=[0])

    # Encode categorical features using the loaded encoders
    for col, encoder in encoders.items():
        if col in input_df.columns:
            # Use a default value for unseen labels
            input_df[col] = input_df[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

    # Ensure the order of columns matches the training data
    features_order = [
        'product_type', 'days_to_expiry', 'stock_quantity',
        'average_daily_sales', 'cost_price', 'selling_price', 'season',
        'day_of_week', 'profit_margin', 'historical_discount_given',
        'units_expired_last_week'
    ]

    # Reorder and handle missing columns
    X = input_df.reindex(columns=features_order).fillna(0)

    # Make a prediction
    predicted_discount = model.predict(X)[0]

    # Output the result as a JSON object
    result = {'suggested_discount_percentage': round(float(predicted_discount), 2)}
    return result


if __name__ == '__main__':
    # This part allows running the script from the command line,
    # reading input from stdin.
    try:
        input_data_str = sys.stdin.read()
        if not input_data_str:
            print(json.dumps({"error": "No input data received."}))
        else:
            input_data = json.loads(input_data_str)
            prediction = predict_discount(input_data)
            print(json.dumps(prediction))
    except json.JSONDecodeError:
        print(json.dumps({"error": "Invalid JSON format received."}))
    except Exception as e:
        print(json.dumps({"error": f"An error occurred: {str(e)}"}))

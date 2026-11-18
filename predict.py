import joblib
import pandas as pd

def predict_house_price(house_details):
    # 1. Load the saved model and column names
    model = joblib.load("models/house_price_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

    # 2. Convert input dictionary into a DataFrame matching training columns
    input_df = pd.DataFrame([house_details])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # 3. Predict the price
    predicted_price = model.predict(input_df)[0]
    return predicted_price
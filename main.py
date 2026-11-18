from preprocessing import load_and_clean_data
from train_model import train_and_save
from predict import predict_house_price

def main():
    data_path = "dataset/house-prices-advanced-regression-techniques/train.csv"

    print("--- Step 1: Preprocessing Data ---")
    X_train, X_test, y_train, y_test, feature_cols = load_and_clean_data(data_path)

    print("\n--- Step 2: Training Model ---")
    train_and_save(X_train, X_test, y_train, y_test, feature_cols)

    print("\n--- Step 3: Testing Sample Prediction ---")
    sample_house = {
        'LotArea': 8500,
        'OverallQual': 7,
        'OverallCond': 5,
        'YearBuilt': 2005,
        'GrLivArea': 1800,
        'FullBath': 2,
        'BedroomAbvGr': 3,
        'TotRmsAbvGrd': 7,
        'GarageCars': 2,
        'Neighborhood': 'CollgCr'
    }
    price = predict_house_price(sample_house)
    print(f"Predicted House Price: ${price:,.2f}")

if __name__ == "__main__":
    main()
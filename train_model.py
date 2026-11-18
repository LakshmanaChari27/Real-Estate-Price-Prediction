import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_and_save(X_train, X_test, y_train, y_test, feature_columns):
    # 1. Create and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 2. Check accuracy on the test set
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Model Trained Successfully!")
    print(f"Average Error (MAE): ${mae:,.2f}")
    print(f"Accuracy Score (R2): {r2:.2%}")

    # 3. Save the model and column list into the 'models/' folder
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/house_price_model.pkl")
    joblib.dump(feature_columns, "models/feature_columns.pkl")
    print("Model saved to models/house_price_model.pkl")
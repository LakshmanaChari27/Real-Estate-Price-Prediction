import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath):
    # 1. Load the dataset
    df = pd.read_csv(filepath)

    # 2. Pick key real-estate features + target ('SalePrice')
    features = [
        'LotArea',       # Lot size in square feet
        'OverallQual',   # Overall material and finish quality (1-10)
        'OverallCond',   # Overall condition rating (1-10)
        'YearBuilt',     # Original construction date
        'GrLivArea',     # Above grade (ground) living area sq ft
        'FullBath',      # Full bathrooms above grade
        'BedroomAbvGr',  # Bedrooms above grade
        'TotRmsAbvGrd',  # Total rooms above grade
        'GarageCars',    # Size of garage in car capacity
        'Neighborhood'   # Physical locations within city limits
    ]
    target = 'SalePrice'

    df = df[features + [target]].copy()

    # 3. Fill any missing numbers with the median
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # 4. Convert text ('Neighborhood') into numbers (One-Hot Encoding)
    df = pd.get_dummies(df, columns=['Neighborhood'], drop_first=True)

    # 5. Split into features (X) and target price (y)
    X = df.drop(columns=[target])
    y = df[target]

    # 6. Split into 80% training data and 20% testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, list(X.columns)
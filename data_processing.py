import pandas as pd


def preprocess_data(df):
    # Print the columns to verify available data
    print("Columns in DataFrame:", df.columns.tolist())

    # Change this to your actual target column name
    target_column = 'Income'  # Update as needed based on your prediction goal
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in DataFrame.")

    # Drop the Name column and target column for features
    X = df.drop(columns=[target_column, 'Name'])

    # Convert categorical variables to one-hot encoding
    X = pd.get_dummies(X, drop_first=True)  # drop_first to avoid multicollinearity

    y = df[target_column]  # Correct target column

    return X, y

import pandas as pd
from data_processing import preprocess_data
from linear_regression import train_and_evaluate_model

def main():
    # Load the data
    df = pd.read_csv('C:\\Users\\Eli\\PycharmProjects\\DPML\\depression_data.csv')  # Update with the correct path
    print("Initial DataFrame:")
    print(df.head())

    # Preprocess the data
    X, y = preprocess_data(df)

    # Train and evaluate the model
    train_and_evaluate_model(X, y)

if __name__ == "__main__":
    main()

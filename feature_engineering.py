import pandas as pd

def create_features(df):
    """Create additional features that might improve model performance."""
    # Example: Create an interaction feature
    df['Income_per_Child'] = df['Income'] / (df['Number of Children'] + 1)  # Avoid division by zero
    return df

def main():
    df = pd.read_csv('../data/processed/depression_data_processed.csv')
    enhanced_df = create_features(df)
    enhanced_df.to_csv('../data/processed/depression_data_enhanced.csv', index=False)

if __name__ == "__main__":
    main()

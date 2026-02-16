import pandas as pd

def load_phiusiil_dataset():
    # Load CSV directly
    df = pd.read_csv('data/PhiUSIIL.csv')  # adjust path if needed

    # Optional: inspect the dataset
    print(df.head())
    print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
    print("Label distribution:\n", df['status'].value_counts())  # check phishing vs legitimate

    return df

if __name__ == "__main__":
    df = load_phiusiil_dataset()

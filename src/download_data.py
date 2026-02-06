import pandas as pd
import os

def download_telco_data():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    os.makedirs('data', exist_ok=True)
    print("Downloading Telco Customer Churn dataset...")
    df = pd.read_csv(url)
    output_path = 'data/telco_churn.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    return df

if __name__ == "__main__":
    download_telco_data()

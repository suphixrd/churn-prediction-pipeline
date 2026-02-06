import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

class ChurnDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records")
        return df
    
    def clean_data(self, df):
        print("Cleaning data...")
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        print(f"Data cleaned. Shape: {df.shape}")
        return df
    
    def encode_features(self, df, fit=True):
        print("Encoding categorical features...")
        if 'Churn' in df.columns:
            if fit:
                self.label_encoders['Churn'] = LabelEncoder()
                df['Churn'] = self.label_encoders['Churn'].fit_transform(df['Churn'])
            else:
                df['Churn'] = self.label_encoders['Churn'].transform(df['Churn'])
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        return df
    
    def scale_features(self, X, fit=True):
        print("Scaling features...")
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def prepare_data(self, df, test_size=0.2, random_state=42):
        print("Preparing data for training...")
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train = self.scale_features(X_train, fit=True)
        X_test = self.scale_features(X_test, fit=False)
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, filepath='models/preprocessor.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='models/preprocessor.pkl'):
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        self.label_encoders = preprocessor_data['label_encoders']
        self.scaler = preprocessor_data['scaler']
        print(f"Preprocessor loaded from {filepath}")

def main():
    preprocessor = ChurnDataPreprocessor()
    df = preprocessor.load_data('data/telco_churn.csv')
    df = preprocessor.clean_data(df)
    df = preprocessor.encode_features(df, fit=True)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    preprocessor.save_preprocessor()
    print("\n✅ Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import os
import json

class ChurnModel:
    def __init__(self, model_params=None):
        if model_params is None:
            model_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
        self.model = RandomForestClassifier(**model_params)
        self.model_params = model_params
        self.metrics = {}
    
    def train(self, X_train, y_train):
        print("Training model...")
        self.model.fit(X_train, y_train)
        print("✅ Model training completed!")
    
    def evaluate(self, X_test, y_test):
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print("\n📊 Model Performance Metrics:")
        for metric_name, metric_value in self.metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        print("\n🔍 Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        return self.metrics
    
    def save_model(self, filepath='models/churn_model.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'model': self.model,
            'params': self.model_params,
            'metrics': self.metrics
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n💾 Model saved to {filepath}")
    
    def load_model(self, filepath='models/churn_model.pkl'):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.model_params = model_data['params']
        self.metrics = model_data['metrics']
        print(f"✅ Model loaded from {filepath}")
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

def main():
    print("Loading processed data...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    model = ChurnModel()
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    model.save_model()
    
    os.makedirs('models', exist_ok=True)
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\n✅ Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
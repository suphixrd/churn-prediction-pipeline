import pandas as pd
import json
import os
from datetime import datetime

class DriftDetector:
    def __init__(self, reference_data_path='data/processed/X_train.csv'):
        self.reference_data = pd.read_csv(reference_data_path)
        print(f" Reference data loaded: {self.reference_data.shape}")
    
    def load_current_data(self, predictions_log_path='data/predictions/predictions_log.json'):
        if not os.path.exists(predictions_log_path):
            print(" No predictions log found!")
            return None
        
        with open(predictions_log_path, 'r') as f:
            predictions = json.load(f)
        
        if len(predictions) == 0:
            print(" Predictions log is empty!")
            return None
        
        # Convert to DataFrame (remove prediction columns)
        current_data = pd.DataFrame(predictions)
        current_data = current_data.drop(['prediction', 'probability'], axis=1, errors='ignore')
        
        print(f"Current data loaded: {current_data.shape}")
        return current_data
    
    def detect_drift(self):
        print("\n Detecting data drift...")
        
        current_data = self.load_current_data()
        
        if current_data is None or len(current_data) < 30:
            print(" Not enough data for drift detection (minimum 30 samples needed)")
            return None
        
        # Simple drift detection using statistical comparison
        print("\n Comparing distributions...")
        
        # Compare ONLY numeric columns that exist in both datasets
        ref_numeric = self.reference_data.select_dtypes(include=['int64', 'float64']).columns
        curr_numeric = current_data.select_dtypes(include=['int64', 'float64']).columns
        
        # Get common numeric columns
        numeric_cols = list(set(ref_numeric) & set(curr_numeric))
        
        if len(numeric_cols) == 0:
            print(" No numeric columns found for comparison")
            return False
        
        drift_detected = False
        drift_score = 0
        
        for col in numeric_cols:
            ref_mean = self.reference_data[col].mean()
            curr_mean = current_data[col].mean()
            diff = abs(ref_mean - curr_mean) / (ref_mean + 1e-10)
            
            if diff > 0.1:  # 10% threshold
                print(f" Drift detected in {col}: {diff:.2%} change")
                drift_detected = True
                drift_score += diff
        
        drift_score = drift_score / len(numeric_cols) if len(numeric_cols) > 0 else 0
        
        # Log drift detection
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if drift_detected:
            print(f"\n DRIFT DETECTED! Drift score: {drift_score:.2%}")
            print(" Retraining recommended!")
            
            drift_log = {
                'timestamp': timestamp,
                'drift_detected': True,
                'drift_score': drift_score,
                'sample_size': len(current_data)
            }
            
            os.makedirs('logs', exist_ok=True)
            with open('logs/drift_log.json', 'a') as f:
                f.write(json.dumps(drift_log) + '\n')
            
            return True
        else:
            print(f" No significant drift detected. Drift score: {drift_score:.2%}")
            return False

def main():
    detector = DriftDetector()
    drift_detected = detector.detect_drift()
    
    if drift_detected:
        print("\n" + "="*50)
        print("ACTION REQUIRED: Model retraining triggered!")
        print("="*50)

if __name__ == "__main__":
    main()
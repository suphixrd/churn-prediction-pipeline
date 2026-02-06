python
import schedule
import time
from drift_detection import DriftDetector
import subprocess
import os
from datetime import datetime

def check_and_retrain():
    """Check for drift and trigger retraining if needed"""
    print(f"\n[{datetime.now()}] Checking for drift...")
    
    detector = DriftDetector()
    drift_detected = detector.detect_drift()
    
    if drift_detected:
        print("\n DRIFT DETECTED - Triggering automatic retraining...")

        try:
            result = subprocess.run(
                ['python', 'src/zenml_pipeline_complete.py'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(" Retraining completed successfully!")
                print(result.stdout)
            else:
                print(" Retraining failed!")
                print(result.stderr)
        
        except Exception as e:
            print(f" Error during retraining: {e}")
    else:
        print(" No drift detected - model is healthy")

def main():
    print("="*60)
    print("AUTOMATED DRIFT MONITORING & RETRAINING SERVICE")
    print("="*60)
    print("Monitoring interval: Every 1 hour")
    print("Press Ctrl+C to stop")
    print("="*60)

    schedule.every(1).hours.do(check_and_retrain)
    
    check_and_retrain()
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
python
from zenml import pipeline, step
import pandas as pd
from data_preprocessing import ChurnDataPreprocessor
from train_model import ChurnModel
from drift_detection import DriftDetector
import mlflow
import mlflow.sklearn
import os

@step
def load_data_step(filepath: str = 'data/telco_churn.csv') -> pd.DataFrame:
    """Load raw data"""
    preprocessor = ChurnDataPreprocessor()
    df = preprocessor.load_data(filepath)
    return df

@step
def preprocess_data_step(df: pd.DataFrame) -> tuple:
    """Preprocess and split data"""
    preprocessor = ChurnDataPreprocessor()
    df = preprocessor.clean_data(df)
    df = preprocessor.encode_features(df, fit=True)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    preprocessor.save_preprocessor()
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    return X_train, X_test, y_train, y_test

@step
def train_model_step(X_train: pd.DataFrame, y_train: pd.Series) -> ChurnModel:
    """Train model with MLflow tracking"""
    mlflow.set_experiment("churn_prediction_pipeline")
    
    with mlflow.start_run():
        model = ChurnModel()
        model.train(X_train, y_train.values.ravel())
        model.save_model()
        
        # Log to MLflow
        mlflow.log_params(model.model_params)
        mlflow.sklearn.log_model(model.model, "model")
        
        print(" Model trained and logged to MLflow")
        return model

@step
def evaluate_model_step(
    model: ChurnModel,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """Evaluate model and log metrics"""
    metrics = model.evaluate(X_test, y_test.values.ravel())
    
    with mlflow.start_run():
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
    
    return metrics

@step
def detect_drift_step() -> bool:
    """Detect data drift"""
    detector = DriftDetector()
    drift_detected = detector.detect_drift()
    return drift_detected if drift_detected is not None else False

@pipeline
def churn_training_pipeline():
    """Complete churn prediction pipeline"""
    df = load_data_step()
    X_train, X_test, y_train, y_test = preprocess_data_step(df)
    model = train_model_step(X_train, y_train)
    metrics = evaluate_model_step(model, X_test, y_test)

@pipeline
def churn_monitoring_pipeline():
    """Monitoring and drift detection pipeline"""
    drift_detected = detect_drift_step()

if __name__ == "__main__":
    print("="*60)
    print("RUNNING CHURN PREDICTION PIPELINE")
    print("="*60)
    
    # Run training pipeline
    training_pipeline = churn_training_pipeline()
    training_pipeline.run()
    
    print("\n" + "="*60)
    print("RUNNING DRIFT DETECTION PIPELINE")
    print("="*60)
    
    # Run monitoring pipeline
    monitoring_pipeline = churn_monitoring_pipeline()
    monitoring_pipeline.run()
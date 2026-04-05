import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import mlflow
import os
import joblib

def train_xgboost_baseline(csv_path="data/processed/processed_pan_india_properties.csv", target_col='Actual_3Yr_ROI_Pct'):
    """
    Trains an XGBoost regressor as a tabular baseline model.
    Logs parameters and metrics to MLflow.
    """
    print(f"Loading data from {csv_path} for XGBoost Baseline...")
    df = pd.read_csv(csv_path)
    
    # 1. Select numeric features
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Remove the target variable and any ID fields
    features = [c for c in numeric_cols if c not in ['Actual_3Yr_ROI_Pct', 'Actual_5Yr_ROI_Pct', 'Price_INR_Cr', 'Price_INR']]
    
    print(f"Features used for baseline: {features}")
    
    df = df.dropna(subset=features + [target_col])
    X = df[features]
    y = df[target_col]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # MLflow Setup
    mlflow_dir = os.path.join(os.getcwd(), "mlruns")
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    mlflow.set_experiment("BharatPropAdvisor_Baseline")
    
    with mlflow.start_run(run_name="XGBoost_Baseline_3Yr-ROI"):
        # Model hyperparameters
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        mlflow.log_params(params)
        
        # 2. Train Model
        print("Training XGBoost...")
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # 3. Evaluate Model
        y_pred = model.predict(X_test)
        
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"--- Baseline XGBoost Results ({target_col}) ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R2:   {r2:.4f}")
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Save Model and Scaler locally
        os.makedirs("models/baselines", exist_ok=True)
        model_path = "models/baselines/xgboost_baseline.json"
        scaler_path = "models/baselines/scaler.pkl"
        
        model.save_model(model_path)
        joblib.dump(scaler, scaler_path)
        
        mlflow.log_artifact(model_path)
        
        print(f"Saved baseline model to {model_path}")

if __name__ == "__main__":
    train_xgboost_baseline()

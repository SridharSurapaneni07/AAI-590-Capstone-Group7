import mlflow
import matplotlib.pyplot as plt
import pandas as pd
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def plot_mlflow_training_curves():
    # Setup MLflow tracking
    mlflow_dir = os.path.join(os.getcwd(), "mlruns")
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    mlflow.set_experiment("BharatPropAdvisor_Vision")
    
    # Get the latest run ID for ViT_Regression_Production_Scale
    runs = mlflow.search_runs(experiment_names=["BharatPropAdvisor_Vision"], 
                              filter_string="tags.mlflow.runName = 'ViT_Regression_Production_Scale'")
    if runs.empty:
        print("No production vision run found in MLflow. Looking for Baseline.")
        runs = mlflow.search_runs(experiment_names=["BharatPropAdvisor_Vision"])
        if runs.empty:
            print("No MLflow runs found.")
            return
            
    # Take the most recent run
    runs = runs.sort_values(by="start_time", ascending=False)
    latest_run_id = runs.iloc[0].run_id
    print(f"Fetching MLflow metrics for Run_ID: {latest_run_id}")
    
    # Fetch metrics
    client = mlflow.tracking.MlflowClient()
    try:
        train_loss = client.get_metric_history(latest_run_id, "train_loss")
        val_loss = client.get_metric_history(latest_run_id, "val_loss")
    except Exception as e:
        print(f"Error fetching metrics (might be incomplete): {e}")
        return
        
    epochs = [m.step for m in train_loss]
    train_values = [m.value for m in train_loss]
    
    # Val loss might not exist if it failed, but let's try
    val_epochs = [m.step for m in val_loss]
    val_values = [m.value for m in val_loss]
    
    if not epochs:
        print("No epochs completed yet/logged directly.")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_values, label='Training MSE Loss', color='#0984e3', linewidth=2.5)
    if val_epochs:
        plt.plot(val_epochs, val_values, label='Validation MSE Loss', color='#fdcb6e', linewidth=2.5, linestyle='--')
        
    plt.title('ViT-B/16 Regression Head Training Curve (Fresh Scaled Run)', fontsize=14, pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    os.makedirs("src/visualization/fresh_results", exist_ok=True)
    out_path = "src/visualization/fresh_results/vit_loss_curve.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved ViT training curve to {out_path}")

def plot_fresh_xgboost_scatter():
    # Load the exact data baseline used previously to map exactly to the 8.74 RMSE
    csv_path="data/processed/processed_pan_india_properties.csv"
    if not os.path.exists(csv_path):
        print("Processed data not found for XGBoost fresh scatter.")
        return
        
    df = pd.read_csv(csv_path)
    target_col = 'Actual_3Yr_ROI_Pct'
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    features = [c for c in numeric_cols if c not in ['Actual_3Yr_ROI_Pct', 'Actual_5Yr_ROI_Pct', 'Price_INR_Cr', 'Price_INR']]
    
    df = df.dropna(subset=features + [target_col])
    X = df[features]
    y = df[target_col]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train fresh baseline exactly as train_baseline.py did
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='#d63031', edgecolor='white', linewidth=0.5, 
                label=f"XGBoost Predictions\nR² = {r2:.3f}\nRMSE = {rmse:.2f}")
    
    # Perfect fit line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#2d3436', linestyle='--', linewidth=2, label='Perfect Target Trend')
    
    plt.title('Fresh XGBoost Tabular Baseline (3-Year ROI %)', fontsize=14, pad=15)
    plt.xlabel('Actual ROI %', fontsize=12)
    plt.ylabel('Predicted ROI %', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    out_path = "src/visualization/fresh_results/xgboost_scatter_fresh.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved fresh XGBoost scatter to {out_path}")

if __name__ == "__main__":
    os.makedirs("src/visualization/fresh_results", exist_ok=True)
    plot_fresh_xgboost_scatter()
    plot_mlflow_training_curves()

import mlflow
import optuna
import os

def setup_mlflow(experiment_name="BharatPropAdvisor_Fusion"):
    """
    Sets up the MLflow tracking URI to a local directory or remote server.
    """
    # Create a local mlruns directory if it doesn't exist
    mlflow_dir = os.path.join(os.getcwd(), "mlruns")
    os.makedirs(mlflow_dir, exist_ok=True)
    
    tracking_uri = f"file://{mlflow_dir}"
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
        
    mlflow.set_experiment(experiment_name)
    return experiment_id

def setup_optuna_study(study_name="hyperparam_optimization"):
    """
    Sets up a basic Optuna study to track hyperparameter sweeps.
    """
    db_path = os.path.join(os.getcwd(), "models", "optuna_study.db")
    storage_url = f"sqlite:///{db_path}"
    
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_url, 
        load_if_exists=True,
        direction="minimize" # Usually minimizing loss (e.g., RMSE for ROI prediction)
    )
    
    print(f"Optuna study created/loaded at: {storage_url}")
    return study

if __name__ == "__main__":
    print("Initializing MLOps Configuration...")
    setup_mlflow()
    setup_optuna_study()
    print("✅ MLOps configuration complete. Ready for training loops in Module 5.")

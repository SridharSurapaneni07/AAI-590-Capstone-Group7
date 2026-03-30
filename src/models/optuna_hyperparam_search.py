"""
Optuna Hyperparameter Exploration for BharatPropAdvisor ViT Pipeline.

This script systematically evaluates combinations of:
  - Learning Rate:     {0.01, 0.001, 0.0001}
  - Dropout Rate:      {0.1, 0.2, 0.3, 0.5}
  - Fusion d_model:    {64, 128, 256}
  - Attention Heads:   {2, 4, 8}

Each trial trains on a representative subset of the REI-Dataset for 5 fast 
epochs and logs the final validation MSE. Results are stored in optuna_study.db
and simultaneously logged to MLflow for full traceability.

Total trials: 20 (Optuna TPE sampler explores the space intelligently).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
import mlflow
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.train_vision import PremiumScoreImageDataset, get_vision_model
from torchvision import transforms


def create_objective(train_dataset, val_dataset, device, mlflow_dir):
    """Factory that returns the Optuna objective function with dataset closure."""

    def objective(trial):
        # === Hyperparameter Search Space ===
        lr = trial.suggest_categorical("learning_rate", [0.01, 0.001, 0.0001])
        dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.5])
        d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])

        # Constraint: nhead must evenly divide d_model
        if d_model % nhead != 0:
            raise optuna.exceptions.TrialPruned(
                f"d_model={d_model} not divisible by nhead={nhead}"
            )

        batch_size = 32
        epochs = 5  # Fast trials — enough to observe convergence trends

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Build ViT model with custom head incorporating trial's dropout
        model = get_vision_model()
        in_features = 768  # ViT-B/16 hidden dim
        model.heads.head = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        model = model.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.heads.head.parameters(), lr=lr)

        # Log this trial to MLflow
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        mlflow.set_experiment("BharatPropAdvisor_Optuna_Search")

        with mlflow.start_run(run_name=f"optuna_trial_{trial.number}"):
            mlflow.log_params({
                "learning_rate": lr,
                "dropout": dropout,
                "d_model": d_model,
                "nhead": nhead,
                "batch_size": batch_size,
                "epochs": epochs
            })

            # Training loop
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                n_train = 0
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    n_train += inputs.size(0)

                train_loss = running_loss / n_train

                # Validation
                model.eval()
                val_loss_sum = 0.0
                n_val = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss_sum += loss.item() * inputs.size(0)
                        n_val += inputs.size(0)

                val_loss = val_loss_sum / n_val

                mlflow.log_metric("train_mse", train_loss, step=epoch)
                mlflow.log_metric("val_mse", val_loss, step=epoch)

                # Report intermediate value for pruning
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    mlflow.log_metric("final_val_mse", val_loss)
                    raise optuna.exceptions.TrialPruned()

                print(f"  Trial {trial.number} | Epoch {epoch+1}/{epochs} | "
                      f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

            mlflow.log_metric("final_val_mse", val_loss)

        return val_loss  # Minimize validation MSE

    return objective


def run_search():
    print("=" * 70)
    print("  OPTUNA HYPERPARAMETER EXPLORATION — BharatPropAdvisor ViT Pipeline")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Prepare dataset (same as production training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_dir = "data/raw/REI-Dataset"
    full_dataset = PremiumScoreImageDataset(root_dir=data_dir, transform=transform)
    total = len(full_dataset)

    # Use a representative subset (500 images) for speed
    subset_size = min(500, total)
    remaining = total - subset_size
    subset, _ = torch.utils.data.random_split(
        full_dataset, [subset_size, remaining],
        generator=torch.Generator().manual_seed(42)
    )

    train_size = int(0.8 * subset_size)
    val_size = subset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        subset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Subset: {subset_size} images ({train_size} train / {val_size} val)")

    # MLflow tracking
    mlflow_dir = os.path.join(os.getcwd(), "mlruns")

    # Create or load the Optuna study
    db_path = os.path.join(os.getcwd(), "optuna_study.db")
    study = optuna.create_study(
        study_name="hyperparam_optimization",
        storage=f"sqlite:///{db_path}",
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    )

    objective = create_objective(train_dataset, val_dataset, device, mlflow_dir)

    start = time.time()
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    elapsed = time.time() - start

    # Print results
    print("\n" + "=" * 70)
    print("  EXPLORATION COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Trials completed: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best Val MSE: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")

    # Detailed results table
    print("\n--- ALL TRIAL RESULTS ---")
    print(f"{'Trial':>6} | {'LR':>8} | {'Dropout':>8} | {'d_model':>8} | {'nhead':>6} | {'Val MSE':>10} | {'State':>10}")
    print("-" * 75)
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            p = t.params
            print(f"{t.number:>6} | {p['learning_rate']:>8} | {p['dropout']:>8} | "
                  f"{p['d_model']:>8} | {p['nhead']:>6} | {t.value:>10.4f} | {'COMPLETE':>10}")
        elif t.state == optuna.trial.TrialState.PRUNED:
            print(f"{t.number:>6} | {'—':>8} | {'—':>8} | {'—':>8} | {'—':>6} | {'—':>10} | {'PRUNED':>10}")


if __name__ == "__main__":
    run_search()

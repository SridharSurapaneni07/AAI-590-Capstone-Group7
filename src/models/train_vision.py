import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import mlflow
import os
import random

class PremiumScoreImageDataset(Dataset):
    """
    Wraps ImageFolder to return a mock 'Premium Score' (Regression Target)
    instead of a categorical class label, since the raw REI dataset does not 
    come with pre-scored aesthetic values.
    """
    def __init__(self, root_dir, transform=None):
        self.image_folder = ImageFolder(root=root_dir, transform=transform)
        # Mocking a static target per image path for consistency across epochs
        random.seed(42)
        self.mock_targets = [random.uniform(40.0, 95.0) for _ in range(len(self.image_folder))]

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        img, _ = self.image_folder[idx]
        target = torch.tensor([self.mock_targets[idx]], dtype=torch.float32)
        return img, target

def get_vision_model():
    """
    Loads a pretrained ViT-Base and swaps the classification head
    for a single-node linear layer for Regression (Premium Score 0-100).
    """
    # Load pretrained ViT from torchvision
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    
    # Freeze all parameters initially for transfer learning
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the head
    # ViT_b_16 in torchvision uses `model.heads.head` as the final linear layer
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, 1) # Output a single continuous value
    
    return model

def train_vision_model(data_dir="data/raw/REI-Dataset", epochs=1, batch_size=32):
    print("Setting up Vision Transformer (ViT-Base)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        dataset = PremiumScoreImageDataset(root_dir=data_dir, transform=transform)
        # Limit dataset to 100 samples for swift execution during testing/baseline
        subset_indices = list(range(100))
        dataset = torch.utils.data.Subset(dataset, subset_indices)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except FileNotFoundError:
        print(f"Directory {data_dir} not found. Skipping ViT training.")
        return
        
    model = get_vision_model().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.heads.head.parameters(), lr=1e-3)
    
    # MLflow tracking
    mlflow_dir = os.path.join(os.getcwd(), "mlruns")
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    mlflow.set_experiment("BharatPropAdvisor_Vision")
    
    with mlflow.start_run(run_name="ViT_Regression_Baseline"):
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("backbone", "vit_b_16")
        
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                print(f"Epoch {epoch+1}/{epochs} | Batch {i+1} | Loss: {loss.item():.4f}")
                
            epoch_loss = running_loss / len(dataset)
            print(f"Epoch {epoch+1} Average Loss: {epoch_loss:.4f}")
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            
        # Save Vision model
        os.makedirs("models/vision", exist_ok=True)
        model_path = "models/vision/vit_premium_scorer.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Vision model trained and saved to {model_path}")
        mlflow.log_artifact(model_path)

if __name__ == "__main__":
    train_vision_model()

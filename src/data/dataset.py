import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torchvision import transforms
from PIL import Image
import os

class PropTabularDataset(Dataset):
    """
    A basic PyTorch Dataset for Tabular housing features to predict ROI.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class MultiModalPropDataset(Dataset):
    """
    Placeholder for the full Multimodal Dataset (Images + Text + Tabular).
    Will be fully implemented when fusion architecture is assembled, but 
    set up here to outline the dataloader structure for the Vision/Text branches.
    """
    def __init__(self, dataframe, image_dir, image_transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Tabular 
        tabular_features = torch.tensor([row['BuiltUpArea_sqft'], row['Bathrooms'], row['AmenitiesCount']], dtype=torch.float32)
        
        # 2. Text (Placeholder for BERT inputs)
        text_desc = str(row.get('Locality', 'Unknown'))
        
        # 3. Vision
        # Since the Kaggle dataset is not perfectly 1:1 mapped to the image dataset out of the box,
        # we randomly sample an image from the provided REI-dataset categories during training to simulate fusion
        img_category = np.random.choice(['bedroom', 'kitchen', 'bathroom', 'livingRoom'])
        cat_path = os.path.join(self.image_dir, img_category)
        
        image_tensor = torch.zeros((3, 224, 224)) # Default blank image
        if os.path.exists(cat_path):
            images = os.listdir(cat_path)
            if len(images) > 0:
                img_name = np.random.choice(images)
                img_path = os.path.join(cat_path, img_name)
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = self.transform(image)
                except Exception:
                    pass

        # Target Label
        target = torch.tensor(row['Actual_3Yr_ROI_Pct'], dtype=torch.float32)

        return tabular_features, text_desc, image_tensor, target

def get_tabular_dataloaders(csv_path, batch_size=32, target_col='Actual_3Yr_ROI_Pct'):
    """
    Prepares train/val data loaders for tabular modeling (e.g., TabNet or Baseline MLP).
    """
    df = pd.read_csv(csv_path)
    
    # Simple feature selection for MVP
    features = ['Price_INR_Cr', 'BuiltUpArea_sqft', 'Bathrooms', 'Balconies', 'AmenitiesCount']
    for f in features:
        df[f] = df[f].fillna(0)
    
    X = df[features].values
    y = df[target_col].values
    
    # Scale Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    train_dataset = PropTabularDataset(X_train, y_train)
    val_dataset = PropTabularDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler

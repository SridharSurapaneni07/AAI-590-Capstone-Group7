import torch
import torch.nn as nn
import os

class PropTabularBranch(nn.Module):
    """
    Feed-Forward Neural Network acting as the Tabular embedding branch.
    Replaces traditional ML algorithms (like XGBoost) so numerical/categorical 
    features can be fine-tuned via backpropagation during multimodal fusion.
    """
    def __init__(self, input_dim=13, embed_dim=128):
        super(PropTabularBranch, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, embed_dim)
        )
        
    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    print("Testing Tabular Branch Initializer...")
    
    # 13 features used in XGBoost baseline (BHK, Bathrooms, Area, etc.)
    dummy_input = torch.randn(32, 13) 
    
    model = PropTabularBranch(input_dim=13, embed_dim=128)
    embeddings = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output embedding shape: {embeddings.shape}")
    print("Tabular branch ready for Multimodal Fusion.")

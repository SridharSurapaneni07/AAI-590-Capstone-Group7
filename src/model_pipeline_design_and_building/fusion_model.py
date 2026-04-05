import torch
import torch.nn as nn
from src.model_training.train_tabular import PropTabularBranch
from src.model_training.train_text import VastuTextBranch
from src.model_training.train_vision import get_vision_model

class MultimodalFusionModel(nn.Module):
    """
    Final Custom Multimodal Fusion Architecture for BharatPropAdvisor.
    Takes inputs from 3 modalities:
    1. Tabular (Area, Bathrooms, Amenities) -> PropTabularBranch
    2. Text (Property description strings) -> VastuTextBranch (BERT)
    3. Vision (Images arrays) -> ViT-Base branch (Fine-tuned)
    """
    def __init__(self, tabular_dim=13, text_model_name="bert-base-multilingual-cased", embed_dim=128):
        super(MultimodalFusionModel, self).__init__()
        
        # 1. Modality Branches
        self.tabular_branch = PropTabularBranch(input_dim=tabular_dim, embed_dim=embed_dim)
        self.text_branch = VastuTextBranch(model_name=text_model_name, embed_dim=embed_dim)
        
        # We load the ViT but remove its final layer so it outputs an embedding (dim=768) 
        # instead of the 1D regression score. 
        # For the real implementation, we would load the fine-tuned weights here.
        self.vision_branch = get_vision_model()
        self.vision_branch.heads.head = nn.Identity() # Output 768-dim vector instead
        
        # Reduce vision output to match our `embed_dim` (128)
        self.vision_proj = nn.Linear(768, embed_dim)
        
        # 2. Cross-Attention Fusion Layer
        # We treat the 3 modalities as a sequence of length 3 for the Transformer Encoder
        self.fusion_encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=4, 
            dim_feedforward=256,
            dropout=0.2,
            batch_first=True
        )
        
        # 3. Final Prediction Heads
        self.roi_predictor = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1) # Predicts 3Yr/5Yr ROI
        )
        
        self.vastu_predictor = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Predicts Vastu Compliance Score (0.0 to 1.0)
        )
        
    def forward(self, tab_features, texts, images, device):
        # Embed each modality
        tab_emb = self.tabular_branch(tab_features) # (batch, embed_dim)
        
        text_emb = self.text_branch(texts, device) # (batch, embed_dim)
        
        vis_raw = self.vision_branch(images) # (batch, 768)
        vis_emb = self.vision_proj(vis_raw) # (batch, embed_dim)
        
        # Stack modalities to form a sequence -> Shape: (batch, 3, embed_dim)
        fused_sequence = torch.stack([tab_emb, text_emb, vis_emb], dim=1)
        
        # Apply Cross-Attention
        attended_sequence = self.fusion_encoder(fused_sequence) # (batch, 3, embed_dim)
        
        # Flatten the attended sequence -> Shape: (batch, 3 * embed_dim)
        flat_fusion = attended_sequence.reshape(attended_sequence.size(0), -1)
        
        # Multitask Predictions
        roi_pred = self.roi_predictor(flat_fusion)
        vastu_score = self.vastu_predictor(flat_fusion)
        
        return roi_pred, vastu_score

if __name__ == "__main__":
    print("Testing Multimodal Fusion Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Dummy inputs for a batch size of 2
    dummy_tab = torch.randn(2, 13).to(device)
    dummy_texts = ["East facing apartment", "South entrance plot"]
    dummy_imgs = torch.randn(2, 3, 224, 224).to(device)
    
    model = MultimodalFusionModel().to(device)
    roi, vastu = model(dummy_tab, dummy_texts, dummy_imgs, device)
    
    print(f"ROI Prediction Shape: {roi.shape}")
    print(f"Vastu Compliance Shape: {vastu.shape}")
    print("Fusion Model Ready!")

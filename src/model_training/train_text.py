import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os

class VastuTextBranch(nn.Module):
    """
    A text embedding branch using a pre-trained multilingual BERT (mBERT) 
    or IndicBERT. It processes property descriptions/amenities to output
    a specialized embedding vector that captures Vastu compliance context.
    """
    def __init__(self, model_name="bert-base-multilingual-cased", embed_dim=128):
        super(VastuTextBranch, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT for baseline fusion to save memory
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim)
        )
        
    def forward(self, texts, device):
        # Tokenize batch of texts
        encoded = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors='pt'
        ).to(device)
        
        # Extract [CLS] token representation
        outputs = self.bert(**encoded)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Project to target dimension
        return self.fc(cls_embedding)

if __name__ == "__main__":
    print("Testing Text Branch Initializer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = VastuTextBranch().to(device)
    
    sample_texts = ["East facing, marble floors", "South entrance, poor lighting"]
    embeddings = model(sample_texts, device)
    
    print(f"Generated Text Embedding Shape: {embeddings.shape}")
    print("Text branch ready for Multimodal Fusion.")

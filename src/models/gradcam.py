import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from src.models.train_vision import get_vision_model
import os

class ViTGradCAM:
    """
    Implements Grad-CAM specifically for Vision Transformers to provide 
    visual explainability (XAI) for the "Premium Aesthetic Score".
    """
    def __init__(self, model, target_layer_name='encoder.layers.encoder_layer_11.ln_1'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        # Register hooks to save gradients and activations from the target layer
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Find the target layer in torchvision ViT structure
        target_layer = dict([*self.model.named_modules()])[self.target_layer_name]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor):
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        # Since the model weights are frozen (transfer learning), we must explicitly
        # tell PyTorch to track gradients back to the image pixels
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        # Backward pass from the predicted continuous score
        output.sum().backward()
        
        # Gradients are shape (batch, sequence_length, hidden_dim) e.g., (1, 197, 768)
        gradients = self.gradients[0] # Take first item in batch -> (197, 768)
        activations = self.activations[0] # Take first item in batch -> (197, 768)
        
        # Global Average Pooling across the sequence dimension (ignoring class token at index 0)
        weights = torch.mean(gradients[1:, :], dim=0) # Shape: (768,)
        
        # Weight the activations (excluding class token)
        cam = torch.zeros(activations.shape[0] - 1, dtype=torch.float32) # Shape: (196,)
        for i, w in enumerate(weights):
            cam += w * activations[1:, i]
            
        cam = F.relu(cam) # ReLU to pass only positive influence
        
        # Reshape the 196 patches back into a 14x14 grid
        cam = cam.view(14, 14).detach().numpy()
        
        # Normalize between 0 and 1
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        return cam

def apply_heatmap(image_path, heatmap, output_path):
    # Read original image
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (224, 224))
    
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (224, 224))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superimpose
    superimposed_img = heatmap_colored * 0.4 + original_img * 0.6
    superimposed_img = np.uint8(superimposed_img)
    
    # Save
    plt.imsave(output_path, superimposed_img)
    print(f"Saved Grad-CAM rendering to {output_path}")
    return superimposed_img

if __name__ == "__main__":
    print("Initializing Grad-CAM module for Visual XAI...")
    device = torch.device("cpu") # Render on CPU
    
    # Load model architecture
    model = get_vision_model()
    # In production, load trained weights here. Using default random weights for structural test.
    
    # Find a sample image
    from glob import glob
    test_images = glob("data/raw/REI-Dataset/bedroom/*.jpg")
    if len(test_images) > 0:
        test_img_path = test_images[0]
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(Image.open(test_img_path).convert('RGB')).unsqueeze(0)
        
        cam = ViTGradCAM(model)
        heatmap = cam.generate_heatmap(input_tensor)
        
        os.makedirs("src/visualization", exist_ok=True)
        apply_heatmap(test_img_path, heatmap, "src/visualization/gradcam_sample.png")
    else:
        print("No test images found in bedroom folder. Skipping render test.")

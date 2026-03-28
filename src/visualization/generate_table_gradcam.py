import torch
import os
from torchvision import transforms
from PIL import Image
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.train_vision import get_vision_model
from src.models.gradcam import ViTGradCAM, apply_heatmap

def generate_table_gradcams():
    print("Loading trained ViT model weights...")
    model = get_vision_model()
    weight_path = "models/vision/vit_premium_scorer.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    else:
        print("Model weights not found. Stopping.")
        return
        
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    cam_generator = ViTGradCAM(model)
    
    for i in range(1, 4):
        img_path = f"src/visualization/fresh_results/sample_case_{i}.jpg"
        if not os.path.exists(img_path):
            print(f"Skipping {img_path}, doesn't exist.")
            continue
            
        print(f"Generating Grad-CAM for {img_path}")
        try:
            input_tensor = transform(Image.open(img_path).convert('RGB')).unsqueeze(0)
            heatmap = cam_generator.generate_heatmap(input_tensor)
            
            out_path = f"src/visualization/fresh_results/sample_case_{i}_gradcam.jpg"
            apply_heatmap(img_path, heatmap, out_path)
        except Exception as e:
            print(f"Error rendering {img_path}: {e}")

if __name__ == "__main__":
    generate_table_gradcams()

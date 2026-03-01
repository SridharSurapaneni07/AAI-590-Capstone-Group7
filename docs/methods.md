# Methodology

This section outlines the progressive modeling approach used to construct the BharatPropAdvisor framework, beginning with baselines and culminating in a multimodal architecture.

## 1. Baseline Tabular Model (XGBoost)
To establish a performance floor, an Extreme Gradient Boosting (XGBoost) regressor was trained solely on the structured tabular features (e.g., Square Footage, Bathrooms, Amenities Count, Age of Property). 

**Data Preparation:** Continuous features were normalized using Standard Scaling to ensure uniform gradient updates. Categorical variables were label-encoded. 

**Hyperparameters & Tracking:** Early experimentation was tracked locally using MLflow. The model utilized `n_estimators=100`, `max_depth=6`, and a `learning_rate=0.1`. The baseline model predicted the mocked 3-year ROI percentage.

## 2. Vision Transformer (ViT) Branch
Unlike standard real estate models, BharatPropAdvisor extracts latent aesthetic value directly from property listings.

**Architecture:** We employed a pre-trained Vision Transformer (`ViT-Base-Patch16-224`) from Hugging Face/Torchvision. Transformers, originally designed for NLP, treat images as sequences of flattened 16x16 pixel patches.

**Transfer Learning:** The core weights of the ViT backbone were frozen to retain the low-level visual filters learned from ImageNet. We replaced the terminal classification head with a single continuous-output Linear layer. 

**Training Strategy:** This branch was fine-tuned using the REI-Dataset. Each image was aggressively resized to `224x224` and normalized using mean-standard deviation normalization `[0.485, 0.456, 0.406]`. The model optimized a Mean Squared Error (MSE) loss against mocked "Premium Aesthetic Scores", demonstrating the pipeline's capability to regress visual desirability mathematically. 

## 3. Data Loaders & Multimodal Pipeline
A custom PyTorch Dataset (`MultiModalPropDataset`) was engineered to handle the simultaneous processing of diverse data streams. For any given listing, the loader streams:
1. An embedded vector representing normalized Tabular Data.
2. Tokenized sub-word strings representing Textual amenities (for Vastu validation).
3. A `3x224x224` image tensor passing through the frozen ViT branch.

Subsequent modules document the final concatenation of these branches into a single cross-attention mechanism for holistic ROI prediction.

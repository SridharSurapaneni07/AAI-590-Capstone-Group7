# 🏙️ BharatPropAdvisor AI
**A Master's Capstone GenAI Architecture for Multimodal Real Estate Analytics**

## Overview
BharatPropAdvisor is an end-to-end Machine Learning orchestration system designed to evaluate real-estate ROI and cultural compliance (Vastu) using a **Late-Fusion Multimodal Architecture**. 

Instead of relying solely on disparate tabular data, this system ingests:
1. **Raw Images** (passed through a Vision Transformer for aesthetic premium scoring & Grad-CAM visual heat-mapping).
2. **Text & Metadata** (passed through an `mBERT` Agent orchestrator to determine Vastu geometries).
3. **Tabular/Spatial Data** (passed through an MLP & XGBoost Baseline for ROI forecasting and geographic PCA mapping).

This repository contains the complete code for Data Ingestion, EDA bias analysis, Model Training, Cross-Attention Fusion, and the Streamlit frontend. It satisfies all complexity mandates for a Capstone AI deliverable.

---

## Primary Dataset Link:
https://www.kaggle.com/datasets/pratyushpuri/pan-india-property-listings-2025-real-estate-data

(Pan-India 2025 synthetic but realistic listings,
~25k–30k rows, columns: City,Locality, PropertyType (Apartment/Villa/Independent House/Plot/Row House),Price_INR, Facing, AmenitiesCount, Lat/Long, YearBuilt, BHK, etc.)

## Secondary Dataset (vision branch only – subset used) Link:
https://www.kaggle.com/datasets/nikhilkushwaha2529/real-estate-with-image (~5,000
real room/exterior photos in 6 folders: Bedroom, Kitchen, Bathr

---

## 🚀 Quickstart & Installation

**Prerequisites:** Python 3.9+, Pip, and Virtualenv.

**1. Clone the repository**
```bash
git clone https://github.com/SridharSurapaneni07/BharatPropAdvisorAI.git
cd BharatPropAdvisorAI
```

**2. Setup Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Streamlit Application (Local UI)**
```bash
streamlit run app/main.py
```
*The app will be available at `http://localhost:8501`. Ensure the port is free.*

---

## 📂 Repository Structure & Code Navigation

Per the University Capstone rubric, all code elements are separated and labeled clearly.

```text
BharatPropAdvisorAI/
├── README.md                   # This master documentation file
├── REPORT.md                   # The comprehensive 20-Page Final Technical Report
├── requirements.txt            # Python dependencies (Streamlit, Torch, Shap, transformers)
├── app/
│   ├── main.py                 # The Streamlit Frontend Dashboard & Integration logic
│   └── assets/                 # UI Mock images (Villas, Penthouses)
├── src/
│   ├── data/                   # Data Ingestion & Bias Exploration (EDA)
│   │   ├── dataset.py          # Script for Pandas EDA, Outlier IQR clipping, & PCA
│   ├── models/                 # Model Definition & Training Pipelines
│   │   ├── train_baseline.py   # Code for Baseline XGBoost tabular training
│   │   ├── train_tabular.py    # Code for MLP Tabular Embeddings
│   │   ├── train_vision.py     # Code for Vision Transformer fine-tuning
│   │   ├── train_text.py       # Code for mBERT textual embeddings
│   │   ├── fusion_model.py     # The Custom Cross-Attention Layer fusing all embeddings
│   │   ├── gradcam.py          # PyTorch Hook logic for Visual Explainability heatmaps
│   ├── agents/                 # LLM Orchestration
│   │   ├── supervisor.py       # A2A LangChain supervisor managing Vastu queries
│   ├── mcp_server/             # Model Context Protocol
│   │   ├── vastu_server.py     # Tool-calling server exposing spatial geometrical checks
│   ├── visualization/          # ML Validation Artifacts 
│       ├── shap_summary.png    # SHAP Game-theoretic feature importance plots
│       ├── xgboost_scatter.png # Baseline validation scatter charts
│       ├── bias_wealth...png   # Seaborn EDA bias distributions
├── data/                       # Local volume for Dataset files (Pan-India Properties)
└── models/                     # Compiled PyTorch & XGBoost parameter weights (.pth, .json)
```

---

## 🧠 Training the Models Locally
If you wish to re-train the models from scratch (rather than just running the Streamlit inferencing app), execute the pipelines in the following specific order:

**1. Baseline Metric Evaluation**
```bash
python src/models/train_baseline.py
```

**2. Vision Transformer & Feature Encoders**
```bash
python src/models/train_vision.py
python src/models/train_tabular.py
python src/models/train_text.py
```

**3. Late Fusion Multi-modal Architecture**
```bash
python src/models/fusion_model.py
```

---

## 📊 Evaluation & Interpretability
This project explicitly fulfills the "Sufficiently Complex architecture" mandate by implementing Explainable AI (XAI).

- **SHAP (SHapley Additive exPlanations):** We use Shap TreeExplainers to evaluate directional tabular drift (e.g. visualizing *why* a high BHK drives positive ROI).
- **Grad-CAM (Gradient-weighted Class Activation Mapping):** We calculate the partial derivatives of the ViT's final classification layer backwards against the PyTorch input image tensor, allowing the Streamlit UI to display visual heatmaps of architectural areas that commanded a 'premium' price. 

*For the complete evaluation breakdown, refer to the included [REPORT.md](./REPORT.md).*

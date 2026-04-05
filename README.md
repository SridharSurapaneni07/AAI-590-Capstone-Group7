<div align="center">

# PropAdvisor AI

**A Multimodal GenAI Engine for Real Estate Valuation and ROI Forecasting**

[![Live App](https://img.shields.io/badge/Live_App-propadvisorai.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://propadvisorai.streamlit.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-%230194E2.svg?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Hugging Face](https://img.shields.io/badge/Transformers-%2334D058.svg?style=for-the-badge&logo=hugging-face&logoColor=white)](https://huggingface.co/)

*AAI-590 Capstone Project — University of San Diego*

</div>

---

## Problem Statement

Traditional Automated Valuation Models (AVMs) rely exclusively on tabular features like BHK count, carpet area, and city name. They completely ignore two critical dimensions that physically drive the Indian real estate market:

1. **Visual aesthetics** — building architecture, interior finishes, natural lighting
2. **Cultural design rules** — Vastu Shastra compliance (directional facing, room placement geometry)

This project solves that gap by fusing three independent neural streams (Vision + Text + Tabular) through a Transformer cross-attention layer, producing ROI-adjusted property valuations that account for both quantitative metrics and qualitative cultural signals.

---

## Live Application

**Try it now:** [https://propadvisorai.streamlit.app](https://propadvisorai.streamlit.app/)

The deployed Streamlit app has 5 tabs:

| Tab | Description |
|:---|:---|
| **Recommendations** | Filter by city, budget, property type. Returns Top-3 high-ROI matches with metric badges. |
| **Property Map** | Interactive Folium map plotting recommended properties geographically. |
| **Visual XAI** | Upload any property image — live ViT-GradCAM heatmap shows which regions drive the premium score. |
| **AI Consultant** | Chat with the Supervisor Agent. It queries the Vastu MCP Server and multimodal pipelines dynamically. |
| **Training and MLOps** | Real Optuna hyperparameter search results (20 trials) + MLflow training curves rendered inline. |

---

## Architecture Overview

```
                    ┌─────────────────────┐
                    │   Streamlit Web UI   │
                    │  (5-Tab Dashboard)   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌──────▼───────┐ ┌──────▼──────────┐
    │  ViT-B/16      │ │ mBERT        │ │ MLP Tabular     │
    │  (Vision)      │ │ (NLP/Vastu)  │ │ (Numeric)       │
    │  224x224 input │ │ [CLS] token  │ │ 256→128→128     │
    │  → 128-D vec   │ │ → 128-D vec  │ │ → 128-D vec     │
    └─────────┬──────┘ └──────┬───────┘ └──────┬──────────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Cross-Attention    │
                    │  Fusion Layer       │
                    │  (384-D → d_model)  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Regression Head    │
                    │  → ROI % Prediction │
                    └─────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                                 │
    ┌─────────▼──────────┐          ┌───────────▼───────────┐
    │  Grad-CAM XAI      │          │  MCP Vastu Agent      │
    │  (Visual Explain.) │          │  (A2A Orchestration)  │
    └────────────────────┘          └───────────────────────┘
```

### Three Neural Branches

| Branch | Model | Input | Output |
|:---|:---|:---|:---|
| **Vision** | `ViT_B_16` (pretrained, fine-tuned) | 224×224 property images | 128-D aesthetic embedding |
| **Text/NLP** | `bert-base-multilingual-cased` | Property descriptions, Vastu text | 128-D semantic embedding |
| **Tabular** | Custom MLP (256→128→128) | BHK, area, price, city, facing | 128-D numeric embedding |

The three 128-D vectors are concatenated into a 384-D tensor and passed through a `TransformerEncoderLayer` (multi-head cross-attention) before the final regression head predicts continuous ROI percentages.

---

## EDA and Dataset

**Dataset:** Pan-India Housing Dataset — 4,600+ property listings across 6 major cities (Bengaluru, Mumbai, Delhi NCR, Hyderabad, Pune, Chennai).

**Key EDA Findings:**
- Extreme geographic wealth bias: Mumbai premium properties have 3-5x price variance vs. Tier-2 cities
- Vastu-compliant properties (North/East facing) correlate with 8-12% higher ROI in the dataset
- Missing/sparse image data for ~30% of listings → necessitated synthetic proxy simulation with REI-Dataset
- Feature distributions required aggressive StandardScaling due to outlier-heavy metropolitan pricing

**Preprocessing Pipeline:**
1. Continuous numeric features → StandardScaler (unit variance)
2. Categorical features (City, PropertyType, Facing) → One-hot encoding
3. Images → Resize 224×224, normalize to ImageNet mean/std
4. Text descriptions → mBERT tokenization (max 128 tokens)

---

## Model Training Pipeline

### Baseline (XGBoost)
```bash
python src/models/train_baseline.py
```
Standard gradient-boosted tree on tabular features only. Produces RMSE = 8.74 (poor generalization).

### Vision Branch (ViT)
```bash
python src/models/train_vision.py
```
Fine-tunes `ViT_B_16` with frozen backbone, custom regression head. 15-epoch training with early stopping (delta < 0.001). Produces trained weights at `models/vision/vit_premium_scorer.pth`.

### Tabular Branch (MLP)
```bash
python src/models/train_tabular.py
```
Deep MLP with BatchNorm + 30% dropout. Specifically tuned to handle metropolitan price outlier distributions.

### Multimodal Fusion
```bash
python src/models/fusion_model.py
```
Combines all three branches through `TransformerEncoderLayer(d_model=128, nhead=4)` → regression head.

### Hyperparameter Search (Optuna)
```bash
python src/models/optuna_hyperparam_search.py
```
20-trial Bayesian optimization using TPE sampler. Search space: lr ∈ {0.01, 0.001, 0.0001}, dropout ∈ {0.1, 0.2, 0.3, 0.5}, d_model ∈ {64, 128, 256}, nhead ∈ {2, 4, 8}. Best trial (#13): Val MSE = 378.75 with lr=0.01, dropout=0.1, d_model=256, nhead=8. Results stored in `models/optuna_study.db`.

### Performance Summary

| Model | RMSE | Notes |
|:---|:---|:---|
| XGBoost Baseline | 8.74 | Tabular only, fails on aesthetic variance |
| MLP Tabular | 4.10 | Better, but no visual/cultural signal |
| **Multimodal Fusion** | **2.15** | **Full pipeline — 75% RMSE reduction vs baseline** |

---

## Explainable AI (XAI)

- **ViT Grad-CAM:** Generates pixel-level heatmaps over property images showing which architectural regions (windows, flooring, lighting) drive the premium aesthetic score. Available live in the Visual XAI tab.
- **SHAP Analysis:** Game-theoretic Shapley value decomposition quantifying each tabular feature's contribution to the final ROI prediction.

---

## MLOps and Experiment Tracking

### MLflow
All training runs are logged to MLflow with metrics (train_loss, val_loss, MSE, RMSE), parameters, and model artifacts.

```bash
# Launch MLflow UI locally
mlflow ui --backend-store-uri file://$(pwd)/mlruns --port 5000
```
Open: [http://localhost:5000](http://localhost:5000)

### Optuna Integration
Hyperparameter search trials are persisted in `models/optuna_study.db` (SQLite). The Training and MLOps tab in the Streamlit app renders all 20 trials with their parameters and validation metrics directly — no separate server needed.

---

## MCP Agentic Server

The project implements a **Model Context Protocol (MCP)** microservice architecture for real-time Vastu compliance validation.

```
User Query → Supervisor Agent (LangChain)
                    │
                    ├── Vastu MCP Server (vastu_server.py)
                    │     → Evaluates facing, kitchen placement, bedroom direction
                    │     → Returns compliance score (0-100%)
                    │
                    └── Multimodal Pipeline
                          → ViT aesthetic score + Tabular ROI + NLP compliance
```

The Supervisor Agent in Tab 4 (AI Consultant) accepts natural language queries and dynamically routes them to the MCP server or the fusion pipeline based on intent.

---

## Quickstart (Local Development)

```bash
# 1. Clone the repository
git clone https://github.com/SridharSurapaneni07/AAI-590-Capstone-Group7.git
cd AAI-590-Capstone-Group7

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Launch Streamlit (model weights auto-download from GitHub Releases)
streamlit run app/main.py
# Open: http://localhost:8501

# 4. Launch MLflow (optional, for viewing training runs)
mlflow ui --backend-store-uri file://$(pwd)/mlruns --port 5000
# Open: http://localhost:5000
```

Model weights (ViT 327MB, XGBoost 487KB, Scaler 1.3KB) are hosted on [GitHub Releases v1.0](https://github.com/SridharSurapaneni07/AAI-590-Capstone-Group7/releases/tag/v1.0) and download automatically on first app launch.

---

## Project Structure

```
├── app/
│   └── main.py                          # Streamlit 5-tab dashboard
├── src/
│   ├── agents/
│   │   └── supervisor.py                # LangChain Supervisor Agent
│   ├── data/
│   │   ├── dataset.py                   # PyTorch Dataset classes
│   │   └── make_dataset.py              # Data ingestion and preprocessing
│   ├── features/
│   │   └── build_features.py            # Feature engineering pipeline
│   ├── mcp_server/
│   │   └── vastu_server.py              # MCP Vastu compliance microservice
│   ├── models/
│   │   ├── fusion_model.py              # Transformer cross-attention fusion
│   │   ├── gradcam.py                   # ViT Grad-CAM heatmap generation
│   │   ├── train_vision.py              # ViT fine-tuning pipeline
│   │   ├── train_tabular.py             # MLP tabular training
│   │   ├── train_baseline.py            # XGBoost baseline
│   │   ├── train_text.py                # mBERT text encoder
│   │   ├── optuna_hyperparam_search.py  # 20-trial Bayesian HP search
│   │   └── mlops_config.py              # MLflow tracking configuration
│   └── visualization/
│       └── *.py                         # Loss curves, SHAP, EDA plots
├── data/
│   ├── raw/                             # Original datasets
│   └── processed/                       # Cleaned CSV files
├── models/
│   ├── vision/                          # Trained ViT weights (.pth)
│   └── baselines/                       # XGBoost + scaler artifacts
├── docs/                                # IEEE manuscript and reports
├── .streamlit/config.toml               # Streamlit Cloud theme config
├── packages.txt                         # System dependencies for cloud
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## Tech Stack

| Category | Technologies |
|:---|:---|
| **Deep Learning** | PyTorch, torchvision (ViT-B/16), HuggingFace Transformers (mBERT) |
| **ML/Baselines** | XGBoost, scikit-learn |
| **Optimization** | Optuna (TPE Bayesian search) |
| **MLOps** | MLflow (file-backend tracking) |
| **Frontend** | Streamlit, Folium, Plotly, Matplotlib |
| **Agentic AI** | LangChain, Model Context Protocol (MCP) |
| **XAI** | Grad-CAM, SHAP |
| **Deployment** | Streamlit Community Cloud, GitHub Releases |

---

*AAI-590 Capstone — University of San Diego, Applied Artificial Intelligence*

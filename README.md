<div align="center">

**An Enterprise-Grade Multimodal GenAI Engine for Real Estate Valuation**

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-%230194E2.svg?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%2334D058.svg?style=for-the-badge&logo=hugging-face&logoColor=white)](https://huggingface.co/)

*Bridging the gap between empirical ROI forecasting and qualitative cultural aesthetics.*
</div>

---
Traditional Automated Valuation Models (AVMs) operate exclusively on tabular data (BHK, Square Footage, City), entirely ignoring the subjective visual aesthetics and cultural design rules (like *Vastu Shastra*) that physically drive the Indian real estate market.

**BharatPropAdvisor** solves this by orchestrating a massive **Multimodal Neural Fusion Architecture**. We simultaneously evaluate unlinked datasets across three continuous streams:

1. **The Visual Engine (ViT):** A Vision Transformer (`ViT_B_16`) ingests architectural photography to map structural room aesthetics, generating spatial Grad-CAM heatmaps to prove its visual logic.
2. **The NLP Engine (mBERT):** A Multilingual BERT model evaluating localized property text descriptions against cultural geometries.
3. **The Empirical Engine (MLP):** A deep Multi-Layer Perceptron evaluating Pan-India Tabular structural data.

These three neural pathways collide in a mathematical **Cross-Attention Fusion Layer**, adjusting standard ROI predictions against visual premiums, while an external **Agentic MCP (Model Context Protocol)** validates real-time property rules via LLM coordination.

---

## 🚀 Quickstart: Running the Dashboards

This architecture includes a localized **MLflow Telemetry Server** to view neural training iterations and a unified **Streamlit User Interface** representing the consumer-facing frontend.

**1. Clone and Install**
```bash
git clone https://github.com/SridharSurapaneni07/BharatPropAdvisorAI.git
cd BharatPropAdvisorAI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Launch the MLflow Tracking Server (MLOps)**
Monitor the continuous early-stopping algorithms, RMSE metrics, and Adam optimization curves visually.
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
```
👉 *Open your browser to: [http://localhost:5000](http://localhost:5000)*

**3. Launch the Streamlit Consumer User Interface**
The 4-Tab Web Application demonstrating the Agentic workflow, Live ViT-GradCAM inference, and the Final Fusion Predictions (Sub-2.0s latency).
```bash
streamlit run app/main.py
```
👉 *Open your browser to: [http://localhost:8501](http://localhost:8501)*

---

## 🧠 Core System Capabilities

### 1. Explainable AI (XAI)
Deep Neural Networks are inherently opaque. BharatPropAdvisor breaks the "Black Box" using:
*   **ViT-GradCAM:** Generates glowing pixel heatmaps over structural images to explicitly prove to the consumer *what* the visual model valued (e.g., highlighting a bay window).
*   **Predictive SHAP:** Game-theoretic matrix calculations physically breaking down why a specific algorithmic prediction shifted positively or negatively compared to baseline medians.

### 2. Live Agentic Orchestration (MCP)
Integrated into Tab 4 of the Streamlit App, a custom LangChain **Supervisor Agent** converses naturally with users, dynamically routing queries to our `vastu_server.py` microservice via the **Model Context Protocol (MCP)** to mathematically validate South/East facing geometric rules dynamically against the user's queries.

### 3. Continuous Integration
The neural network implements an Enterprise-scale CI/CD PyTorch sequence. During the 15-Epoch training loops, custom callbacks monitor the `val_loss` array. If the model achieves continuous optimization (<0.001 Delta), it safely aborts and serializes the state-dictionary (`.pth`) directly to the SQLite MLflow registry, preventing "Curse of Dimensionality" overfitting.

---

## 📂 Master Architecture Tree
```text
/
├── README.md                   # This master documentation file
├── requirements.txt            # Python environments (PyTorch, Streamlit, MLflow)
├── app/
│   ├── main.py                 # The Monolithic Streamlit Frontend Dashboard
│   └── assets/                 # UI/UX graphic design assets
├── src/
│   ├── mcp_server/             # Model Context Protocol microservices
│   │   └── vastu_server.py     # Agentic tool bridging Vastu rules to the LLM
│   ├── models/                 # Deep Learning Optimization Pipelines
│   │   ├── train_vision.py     # ViT Feature Extraction Loop
│   │   ├── train_tabular.py    # MLP Tabular Scaling Architecture
│   │   ├── fusion_model.py     # Central Transformer Cross-Attention Layer
│   │   └── gradcam.py          # Spatial visualization activation matrix
│   ├── visualization/          # Algorithmic output rendering scripts
├── data/                       # Local volume array for property structures
└── mlruns/                     # Active MLOps SQLite tracking database
```

---

*This architecture was engineered specifically to adhere to the elite academic standards required for the AAI-590 Capstone Publication.*

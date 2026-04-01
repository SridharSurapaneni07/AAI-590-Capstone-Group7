import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
from PIL import Image
import sys
import torch
import sqlite3
import json
import urllib.request

# Add root folder to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ───────────────────────────────────────────────────────────────
# AUTO-DOWNLOAD MODEL WEIGHTS FROM GITHUB RELEASES
# ───────────────────────────────────────────────────────────────
RELEASE_BASE = "https://github.com/SridharSurapaneni07/AAI-590-Capstone-Group7/releases/download/v1.0"
MODEL_FILES = {
    "models/vision/vit_premium_scorer.pth": f"{RELEASE_BASE}/vit_premium_scorer.pth",
    "models/baselines/xgboost_baseline.json": f"{RELEASE_BASE}/xgboost_baseline.json",
    "models/baselines/scaler.pkl": f"{RELEASE_BASE}/scaler.pkl",
}

def download_models():
    """Download model weights from GitHub Releases if not present locally."""
    for local_path, url in MODEL_FILES.items():
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                print(f"Downloading {os.path.basename(local_path)}...")
                urllib.request.urlretrieve(url, local_path)
                print(f"  Saved to {local_path}")
            except Exception as e:
                print(f"  Download failed: {e}")

download_models()

from src.agents.supervisor import SupervisorAgent

# ───────────────────────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PropAdvisor AI",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────────────────────────────────────
# PREMIUM CSS — Light, Clean, Mobile-Responsive
# ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global */
    .stApp {
        background-color: #f5f7fb;
        font-family: 'Inter', sans-serif;
        color: #1a1a2e;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}

    /* Title */
    .app-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1a1a2e;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    .app-subtitle {
        color: #6c757d;
        font-size: 1rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    
    /* Cards */
    .prop-card {
        background: #ffffff;
        border: 1px solid #e8ecf1;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 1rem;
        transition: all 0.25s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .prop-card:hover {
        border-color: #667eea;
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.12);
    }
    
    /* Property title */
    .prop-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
    }
    .prop-loc {
        color: #6c757d;
        font-size: 0.85rem;
        margin-bottom: 0.8rem;
    }
    .prop-price {
        font-size: 1.8rem;
        font-weight: 800;
        color: #00b894;
        margin: 0.5rem 0;
    }
    
    /* Metric badges */
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 0.8rem;
    }
    .m-badge {
        background: #eef1f8;
        border: 1px solid #dce1ec;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #4a5568;
    }
    .m-badge.vision {
        background: #fff9e6;
        border-color: #ffd966;
        color: #b8860b;
    }
    
    /* Stats row */
    .stats-row {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin: 1.5rem 0;
    }
    .stat-card {
        flex: 1;
        min-width: 140px;
        background: #ffffff;
        border: 1px solid #e8ecf1;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .stat-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #667eea;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #6c757d;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Optuna table */
    .optuna-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        margin-top: 1rem;
    }
    .optuna-table th {
        background: #eef1f8;
        color: #1a1a2e;
        padding: 10px 14px;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid #dce1ec;
    }
    .optuna-table td {
        padding: 8px 14px;
        border-bottom: 1px solid #f0f0f5;
        color: #4a5568;
    }
    .optuna-table tr:hover td {
        background: #f5f7fb;
    }
    .best-row td {
        color: #00b894 !important;
        font-weight: 700;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e8ecf1;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #eef1f8;
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #6c757d;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #667eea !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    
    /* Best config card */
    .best-card {
        background: #f0faf5;
        border: 1px solid #b2dfdb;
        border-radius: 14px;
        padding: 20px;
        margin-top: 1rem;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .app-title { font-size: 1.6rem; }
        .prop-price { font-size: 1.4rem; }
        .stats-row { flex-direction: column; }
        .stat-card { min-width: 100%; }
        .metric-row { flex-direction: column; }
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_agent():
    return SupervisorAgent()

agent = get_agent()

@st.cache_resource
def get_vision_xai():
    import importlib
    import src.models.gradcam
    importlib.reload(src.models.gradcam)
    from src.models.train_vision import get_vision_model
    from src.models.gradcam import ViTGradCAM
    device = torch.device("cpu")
    model = get_vision_model().to(device)
    model.eval()
    weight_path = "models/vision/vit_premium_scorer.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
    cam = ViTGradCAM(model, target_layer_name='encoder.layers.encoder_layer_11.ln_1')
    return cam

try:
    gradcam = get_vision_xai()
    vision_ready = True
except Exception as e:
    vision_ready = False
    st.sidebar.error(f"Vision init error: {e}")

@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/processed/processed_pan_india_properties.csv")
    except Exception as e:
        st.error(f"Data load error: {e}")
        return pd.DataFrame()

df_properties = load_data()

def load_optuna_trials():
    """Load real Optuna trial data from the SQLite study database."""
    db_path = os.path.join(os.path.dirname(__file__), '..', 'optuna_study.db')
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        trials = pd.read_sql("SELECT * FROM trials", conn)
        params = pd.read_sql("SELECT * FROM trial_params", conn)
        values = pd.read_sql("SELECT * FROM trial_values", conn)
        conn.close()
        if trials.empty:
            return None
        return {"trials": trials, "params": params, "values": values}
    except Exception:
        return None

def load_mlflow_metrics():
    """Load training metrics from MLflow local file store."""
    mlruns_dir = os.path.join(os.path.dirname(__file__), '..', 'mlruns')
    metrics_data = []
    if not os.path.exists(mlruns_dir):
        return None
    for exp_id in os.listdir(mlruns_dir):
        exp_path = os.path.join(mlruns_dir, exp_id)
        if not os.path.isdir(exp_path) or exp_id in ['models', '.trash']:
            continue
        for run_id in os.listdir(exp_path):
            run_path = os.path.join(exp_path, run_id)
            metrics_path = os.path.join(run_path, 'metrics')
            params_path = os.path.join(run_path, 'params')
            tags_path = os.path.join(run_path, 'tags')
            if not os.path.isdir(metrics_path):
                continue
            run_data = {"run_id": run_id[:8], "experiment": exp_id}
            # Read run name
            run_name_file = os.path.join(tags_path, 'mlflow.runName')
            if os.path.exists(run_name_file):
                with open(run_name_file) as f:
                    run_data["name"] = f.read().strip()
            # Read params
            if os.path.isdir(params_path):
                for pf in os.listdir(params_path):
                    with open(os.path.join(params_path, pf)) as f:
                        run_data[f"param_{pf}"] = f.read().strip()
            # Read metrics (last value)
            for mf in os.listdir(metrics_path):
                with open(os.path.join(metrics_path, mf)) as f:
                    lines = f.readlines()
                    if lines:
                        last = lines[-1].strip().split()
                        run_data[f"metric_{mf}"] = float(last[1]) if len(last) > 1 else None
                        # Also collect time series for training curves
                        if mf in ['train_loss', 'val_loss', 'train_mse', 'val_mse']:
                            series = []
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) >= 3:
                                    series.append({"step": int(parts[2]), "value": float(parts[1])})
                            run_data[f"series_{mf}"] = series
            metrics_data.append(run_data)
    return metrics_data if metrics_data else None


# ───────────────────────────────────────────────────────────────
# SIDEBAR
# ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### PropAdvisor")
    st.markdown("---")
    st.markdown("#### Investor Profile")
    
    with st.form("investment_form"):
        budget_range = st.slider("Budget Range (Cr)", min_value=0.5, max_value=20.0, value=(1.0, 5.0), step=0.5)
        time_horizon = st.selectbox("Investment Horizon", ["3 Years", "5 Years", "10 Years"])
        property_type = st.multiselect("Property Type", ["Apartment", "Villa", "Penthouse", "Studio", "Independent House"], default=["Apartment"])
        city = st.selectbox("Target City", ["Any", "Bengaluru", "MMR (Mumbai)", "Delhi NCR", "Hyderabad", "Pune", "Chennai"])
        
        st.markdown("---")
        st.markdown("#### Upload Property Image")
        st.caption("Upload a real estate image for live visual scoring and Grad-CAM analysis.")
        uploaded_file = st.file_uploader("Select Image", type=["jpg", "png", "jpeg"])
        
        analyze_btn = st.form_submit_button("Generate Recommendations", type="primary", use_container_width=True)

# ───────────────────────────────────────────────────────────────
# HEADER
# ───────────────────────────────────────────────────────────────
st.html("<div class='app-title'>PropAdvisor AI</div>")
st.html("<div class='app-subtitle'>Multimodal GenAI Engine for Real Estate Valuation — Tabular + Vision + NLP Fusion</div>")

# Quick stats bar
if not df_properties.empty:
    n_props = len(df_properties)
    n_cities = df_properties['City'].nunique() if 'City' in df_properties.columns else 0
    avg_roi = df_properties['Actual_3Yr_ROI_Pct'].mean() if 'Actual_3Yr_ROI_Pct' in df_properties.columns else 0
    st.html(f"""
    <div class="stats-row">
        <div class="stat-card"><div class="stat-value">{n_props:,}</div><div class="stat-label">Properties Indexed</div></div>
        <div class="stat-card"><div class="stat-value">{n_cities}</div><div class="stat-label">Cities Covered</div></div>
        <div class="stat-card"><div class="stat-value">{avg_roi:.1f}%</div><div class="stat-label">Avg 3-Year ROI</div></div>
        <div class="stat-card"><div class="stat-value">3</div><div class="stat-label">Neural Branches</div></div>
    </div>
    """)

# ───────────────────────────────────────────────────────────────
# MAIN CONTENT
# ───────────────────────────────────────────────────────────────
if analyze_btn:
    st.session_state['analyzed'] = True

if st.session_state.get('analyzed', False):
    with st.spinner("Running multimodal analysis pipeline..."):
        filtered_df = df_properties.copy()
        if not filtered_df.empty:
            if city != "Any":
                search_city = "MMR" if "MMR" in city else city
                filtered_df = filtered_df[filtered_df['City'] == search_city]
            if property_type:
                filtered_df = filtered_df[filtered_df['PropertyType'].isin(property_type)]
            filtered_df = filtered_df[(filtered_df['Price_INR_Cr'] >= budget_range[0]) & (filtered_df['Price_INR_Cr'] <= budget_range[1])]
            
            # Live Vision Score
            global_premium_score = 0
            if uploaded_file is not None and vision_ready:
                from torchvision import transforms
                img = Image.open(uploaded_file).convert('RGB')
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                input_tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    global_premium_score = float(gradcam.model(input_tensor)[0][0].item())
            
            roi_col = 'Actual_5Yr_ROI_Pct' if '5' in time_horizon else 'Actual_3Yr_ROI_Pct'
            
            if global_premium_score > 0:
                roi_normalized = (filtered_df[roi_col] / 40.0) * 100 
                filtered_df['Total_Investment_Score'] = (roi_normalized * 0.7) + (global_premium_score * 0.3)
                filtered_df = filtered_df.sort_values(by='Total_Investment_Score', ascending=False)
            else:
                filtered_df = filtered_df.sort_values(by=roi_col, ascending=False)
            
            top_3 = filtered_df.head(3).copy()
            if global_premium_score > 0:
                top_3['Premium_Score'] = global_premium_score
            else:
                top_3['Premium_Score'] = None
            top_3 = top_3.to_dict('records')
        else:
            top_3 = []
        
        # ── 5 TABS ──
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Recommendations", "Property Map", "Visual XAI", "AI Consultant", "Training and MLOps"
        ])

        # ── TAB 1: Recommendations ──
        with tab1:
            st.subheader("Top Matches")
            if not top_3:
                st.warning("No properties match your criteria. Try broadening your budget or city selection.")
            else:
                cols = st.columns(len(top_3))
                for idx, col in enumerate(cols):
                    prop = top_3[idx]
                    with col:
                        bhk_val = prop.get('BHK', '')
                        bhk_str = f"{bhk_val} BHK " if pd.notna(bhk_val) and bhk_val else ""
                        prop_name = f"{bhk_str}{prop.get('PropertyType', 'Property')}"
                        loc = f"{prop.get('Locality', 'Unknown')}, {prop.get('City', city)}"
                        price = f"Rs. {prop.get('Price_INR_Cr', 0):.2f} Cr"
                        roi3 = f"{prop.get('Actual_3Yr_ROI_Pct', 0):.1f}%"
                        roi5 = f"{prop.get('Actual_5Yr_ROI_Pct', 0):.1f}%"
                        
                        vision_badge = f'<div class="m-badge vision">Visual Score: {prop.get("Premium_Score"):.1f}/100</div>' if prop.get('Premium_Score') is not None else ''
                        
                        st.html(f"""
                        <div class="prop-card">
                            <div class="prop-title">{prop_name}</div>
                            <div class="prop-loc">{loc}</div>
                            <div class="prop-price">{price}</div>
                            <div class="metric-row">
                                <div class="m-badge">3Y ROI: {roi3}</div>
                                <div class="m-badge">5Y ROI: {roi5}</div>
                                {vision_badge}
                            </div>
                        </div>
                        """)
                        
                        with st.expander("Request AI Vastu Analysis"):
                            from src.mcp_server.vastu_server import VastuMCPServer
                            import json as _json
                            _vastu = VastuMCPServer()
                            _facing = str(prop.get('Facing', 'Unknown'))
                            _result = _json.loads(_vastu.evaluate_property(facing=_facing))
                            st.success(f"Vastu Match Score: {_result['vastu_compliance_percentage']}%")
                            if _result['detailed_report']:
                                for _detail in _result['detailed_report']:
                                    st.caption(f"- {_detail}")
                            else:
                                st.caption(f"Facing: {_facing}. No detailed Vastu rules matched. Provide kitchen/bedroom placement for a complete assessment.")

        # ── TAB 2: Map ──
        with tab2:
            st.subheader("Geographic Distribution")
            if len(top_3) > 0:
                first_prop = top_3[0]
                m = folium.Map(location=[first_prop.get('Latitude', 20.5937), first_prop.get('Longitude', 78.9629)], zoom_start=11)
                for prop in top_3:
                    if pd.notna(prop.get('Latitude')) and pd.notna(prop.get('Longitude')):
                        popup = f"Rs.{prop.get('Price_INR_Cr', 0):.2f}Cr | {prop.get('Actual_3Yr_ROI_Pct', 0)}% ROI"
                        folium.Marker(
                            [prop['Latitude'], prop['Longitude']], 
                            popup=popup,
                            tooltip=f"{prop.get('Locality', 'Property')}",
                            icon=folium.Icon(color="purple", icon="info-sign")
                        ).add_to(m)
            else:
                m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
            st_folium(m, width=900, height=450, returned_objects=[])

        # ── TAB 3: Visual XAI ──
        with tab3:
            st.subheader("Vision Transformer Explainability")
            if uploaded_file is not None:
                colA, colB = st.columns(2)
                img = Image.open(uploaded_file).convert('RGB')
                colA.image(img, caption="Uploaded Image", use_container_width=True)
                
                if vision_ready:
                    with st.spinner("Running ViT-GradCAM inference..."):
                        from torchvision import transforms
                        from src.models.gradcam import apply_heatmap
                        
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        input_tensor = transform(img).unsqueeze(0)
                        heatmap_data = gradcam.generate_heatmap(input_tensor)
                        
                        temp_in = "temp_in.jpg"
                        temp_out = "temp_heatmap.jpg"
                        img.save(temp_in)
                        apply_heatmap(temp_in, heatmap_data, temp_out)
                        
                        res_img = Image.open(temp_out)
                        colB.image(res_img, caption="Grad-CAM Activation Map", use_container_width=True)
                        st.success("Transformer attention extracted. Red regions indicate areas commanding premium aesthetic value.")
                else:
                    st.error("Vision model not initialized. Cannot render XAI output.")
            else:
                st.info("Upload a property image in the sidebar to activate the Visual Analysis pipeline.")

        # ── TAB 4: AI Consultant ──
        with tab4:
            st.subheader("Agent-to-Agent Orchestrator")
            st.caption("Ask about investment potential, Vastu compliance, or property analysis. The agent uses real data from your recommended properties.")
            
            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Display existing chat history
            for msg in st.session_state.chat_history:
                st.chat_message(msg['role']).write(msg['content'])
            
            user_query = st.chat_input("Ask: Is this East-facing apartment a good investment?")
            if user_query:
                st.chat_message("user").write(user_query)
                st.session_state.chat_history.append({'role': 'user', 'content': user_query})
                
                # Build real property features from top recommendation
                if top_3:
                    best_prop = top_3[0]
                    real_features = {
                        'Facing': str(best_prop.get('Facing', 'Unknown')),
                        'City': str(best_prop.get('City', 'Unknown')),
                        'Locality': str(best_prop.get('Locality', 'Unknown')),
                        'PropertyType': str(best_prop.get('PropertyType', 'Property')),
                        'BHK': best_prop.get('BHK', 'N/A'),
                        'Price_INR_Cr': best_prop.get('Price_INR_Cr', 0),
                        'Actual_3Yr_ROI_Pct': best_prop.get('Actual_3Yr_ROI_Pct', 0),
                        'Actual_5Yr_ROI_Pct': best_prop.get('Actual_5Yr_ROI_Pct', 0),
                    }
                else:
                    real_features = {}
                
                response = agent.process_query(user_query, real_features)
                st.chat_message("assistant").write(response)
                st.session_state.chat_history.append({'role': 'assistant', 'content': response})

        # ── TAB 5: Training and MLOps ──
        with tab5:
            st.subheader("Training Pipeline and Hyperparameter Search")
            
            # ── Optuna Results ──
            st.markdown("#### Optuna Bayesian Hyperparameter Search")
            optuna_data = load_optuna_trials()
            if optuna_data and not optuna_data["trials"].empty:
                trials_df = optuna_data["trials"]
                params_df = optuna_data["params"]
                values_df = optuna_data["values"]
                
                completed = trials_df[trials_df['state'] == 'COMPLETE']
                pruned = trials_df[trials_df['state'] == 'FAIL_OR_PRUNED'] if 'FAIL_OR_PRUNED' in trials_df['state'].values else trials_df[trials_df['state'] != 'COMPLETE']
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Trials", len(trials_df))
                c2.metric("Completed", len(completed))
                c3.metric("Pruned", len(pruned))
                
                # Build results table from params and values
                table_html = '<table class="optuna-table"><tr><th>Trial</th><th>Learning Rate</th><th>Dropout</th><th>d_model</th><th>nhead</th><th>Val MSE</th><th>State</th></tr>'
                
                for _, trial_row in trials_df.iterrows():
                    tid = trial_row['trial_id']
                    state = trial_row['state']
                    
                    # Get params for this trial
                    t_params = params_df[params_df['trial_id'] == tid]
                    lr = t_params[t_params['param_name'] == 'learning_rate']['param_value'].values
                    dr = t_params[t_params['param_name'] == 'dropout']['param_value'].values
                    dm = t_params[t_params['param_name'] == 'd_model']['param_value'].values
                    nh = t_params[t_params['param_name'] == 'nhead']['param_value'].values
                    
                    lr_val = lr[0] if len(lr) > 0 else "—"
                    dr_val = dr[0] if len(dr) > 0 else "—"
                    dm_val = dm[0] if len(dm) > 0 else "—"
                    nh_val = nh[0] if len(nh) > 0 else "—"
                    
                    # Get value
                    t_val = values_df[values_df['trial_id'] == tid]['value'].values
                    val_str = f"{float(t_val[0]):.2f}" if len(t_val) > 0 else "—"
                    
                    state_label = "Complete" if state == "COMPLETE" else "Pruned"
                    row_class = ' class="best-row"' if val_str != "—" and len(t_val) > 0 and float(t_val[0]) == values_df['value'].min() else ""
                    
                    table_html += f'<tr{row_class}><td>{tid}</td><td>{lr_val}</td><td>{dr_val}</td><td>{dm_val}</td><td>{nh_val}</td><td>{val_str}</td><td>{state_label}</td></tr>'
                
                table_html += '</table>'
                st.html(table_html)
                
                # Best trial summary
                if not values_df.empty:
                    best_idx = values_df['value'].idxmin()
                    best_trial_id = values_df.loc[best_idx, 'trial_id']
                    best_val = values_df.loc[best_idx, 'value']
                    best_params = params_df[params_df['trial_id'] == best_trial_id]
                    
                    st.html(f"""
                    <div class="best-card">
                        <div class="prop-title" style="color: #00b894;">Best Configuration — Trial #{best_trial_id}</div>
                        <div style="color: #4a5568; margin-top: 8px;">
                            Validation MSE: <strong style="color: #00b894;">{best_val:.4f}</strong>
                        </div>
                        <div class="metric-row" style="margin-top: 12px;">
                            {''.join(f'<div class="m-badge">{row["param_name"]}: {row["param_value"]}</div>' for _, row in best_params.iterrows())}
                        </div>
                    </div>
                    """)
            else:
                st.info("No Optuna trials found. Run src/models/optuna_hyperparam_search.py to generate hyperparameter exploration data.")
            
            st.markdown("---")
            
            # ── MLflow Metrics ──
            st.markdown("#### MLflow Experiment Tracking")
            mlflow_data = load_mlflow_metrics()
            if mlflow_data:
                for run in mlflow_data:
                    run_name = run.get("name", run.get("run_id", "Unknown"))
                    
                    # Collect visible metrics
                    display_metrics = {}
                    for k, v in run.items():
                        if k.startswith("metric_") and not k.startswith("metric_series_") and v is not None:
                            clean_name = k.replace("metric_", "").replace("_", " ").title()
                            display_metrics[clean_name] = v
                    
                    if display_metrics:
                        with st.expander(f"Run: {run_name}"):
                            metric_cols = st.columns(min(len(display_metrics), 4))
                            for i, (name, val) in enumerate(display_metrics.items()):
                                metric_cols[i % len(metric_cols)].metric(name, f"{val:.4f}")
                            
                            # Training curves
                            train_series = run.get("series_train_loss") or run.get("series_train_mse")
                            val_series = run.get("series_val_loss") or run.get("series_val_mse")
                            
                            if train_series and val_series:
                                import matplotlib.pyplot as plt
                                fig, ax = plt.subplots(figsize=(8, 3))
                                fig.patch.set_facecolor('#f5f7fb')
                                ax.set_facecolor('#ffffff')
                                
                                train_steps = [p["step"] for p in train_series]
                                train_vals = [p["value"] for p in train_series]
                                val_steps = [p["step"] for p in val_series]
                                val_vals = [p["value"] for p in val_series]
                                
                                ax.plot(train_steps, train_vals, color='#667eea', linewidth=2, label='Train Loss')
                                ax.plot(val_steps, val_vals, color='#00b894', linewidth=2, label='Val Loss')
                                ax.set_xlabel("Epoch", color='#4a5568', fontsize=9)
                                ax.set_ylabel("MSE", color='#4a5568', fontsize=9)
                                ax.tick_params(colors='#4a5568', labelsize=8)
                                ax.legend(fontsize=8, facecolor='#ffffff', edgecolor='#e8ecf1', labelcolor='#1a1a2e')
                                for spine in ax.spines.values():
                                    spine.set_color('#e8ecf1')
                                ax.grid(True, alpha=0.3, color='#dce1ec')
                                
                                st.pyplot(fig)
                                plt.close(fig)
            else:
                st.info("No MLflow data found. Run training scripts and ensure mlruns/ directory is accessible.")
else:
    st.info("Set your investment constraints in the sidebar and click Generate Recommendations to begin analysis.")

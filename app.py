"""
Streamlit Deployment App
==========================
Upload a dyed sample image → predict concentration (ppm) → display results.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import extract_color_features, crop_roi, resize_image, load_image
from src.cnn_model import DyeConcentrationCNN
from src.baseline_models import FEATURE_COLUMNS, COLOR_FEATURE_COLUMNS, DYE_TYPE_COLUMNS
from src.report_generator import generate_report_template, generate_report_groq
from src.utils import DYE_CONFIG

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Dye Concentration Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 16px; margin-bottom: 2rem;
        color: white; text-align: center;
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .main-header p { font-size: 1rem; opacity: 0.9; margin-top: 0.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem; border-radius: 12px; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h3 { font-size: 2rem; color: #4a00e0; margin: 0; }
    .metric-card p { color: #666; margin: 0.3rem 0 0 0; font-size: 0.9rem; }
    .report-box {
        background: #1e1e2e; color: #cdd6f4; padding: 1.5rem;
        border-radius: 12px; font-family: 'Courier New', monospace;
        font-size: 0.85rem; white-space: pre-wrap;
    }
    .color-swatch {
        width: 60px; height: 60px; border-radius: 50%;
        display: inline-block; border: 3px solid white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────
@st.cache_resource
def load_baseline_model(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_resource
def load_cnn_model(model_path):
    if not os.path.exists(model_path):
        return None
    model = DyeConcentrationCNN(pretrained=False, freeze_backbone=False)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def predict_baseline(model, cropped_image_array, dye_type="methylene_blue"):
    features = extract_color_features(cropped_image_array)
    # Add dye_type one-hot encoding
    features["dye_methylene_blue"] = 1.0 if dye_type == "methylene_blue" else 0.0
    features["dye_congo_red"] = 1.0 if dye_type == "congo_red" else 0.0
    features["dye_crystal_violet"] = 1.0 if dye_type == "crystal_violet" else 0.0
    X = np.array([[features[col] for col in FEATURE_COLUMNS]])
    return float(model.predict(X)[0]), features

def predict_cnn(model, cropped_pil_image):
    # Must match training transforms in dataset.py (ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(cropped_pil_image).unsqueeze(0)
    with torch.no_grad():
        pred = model(tensor)
    # Model outputs raw ppm directly (no scaling needed)
    image_array = np.array(cropped_pil_image.resize((224, 224)))
    features = extract_color_features(image_array)
    return float(pred.item()), features

def create_gauge_chart(ppm, max_ppm=200):
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={"projection": "polar"})
    ratio = min(ppm / max_ppm, 1.0)
    theta = np.linspace(0, np.pi, 100)
    ax.plot(theta, [1]*100, color="#e0e0e0", linewidth=20, alpha=0.3)
    theta_fill = np.linspace(0, np.pi * ratio, 100)
    colors = ["#00b894", "#fdcb6e", "#e17055", "#d63031"]
    color = colors[min(int(ratio * len(colors)), len(colors) - 1)]
    ax.plot(theta_fill, [1]*100, color=color, linewidth=20)
    ax.set_ylim(0, 1.5)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0); ax.set_thetamax(180)
    ax.set_rticks([]); ax.set_thetagrids([0, 45, 90, 135, 180],
        ["0", str(max_ppm//4), str(max_ppm//2), str(3*max_ppm//4), str(max_ppm)])
    ax.set_title(f"{ppm:.1f} ppm", fontsize=22, fontweight="bold", pad=20, color=color)
    plt.tight_layout()
    return fig

def create_rgb_bar(features):
    fig, ax = plt.subplots(figsize=(6, 2))
    vals = [features.get("r_mean", 0), features.get("g_mean", 0), features.get("b_mean", 0)]
    colors = ["#e74c3c", "#27ae60", "#3498db"]
    labels = ["Red", "Green", "Blue"]
    bars = ax.barh(labels, vals, color=colors, height=0.6, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(val + 2, bar.get_y() + bar.get_height()/2, f"{val:.0f}",
                va="center", fontweight="bold", fontsize=11)
    ax.set_xlim(0, 280); ax.set_xlabel("Mean Channel Value")
    ax.set_title("RGB Channel Analysis", fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ── Main App ──────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧪 Dye Concentration Predictor</h1>
        <p>Upload a smartphone image of a dyed sample to predict its concentration (ppm)</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        model_choice = st.selectbox("Prediction Model", ["Random Forest (Baseline)", "CNN (MobileNetV2)"],
                                      help="Select the ML model for prediction")
        dye_type = st.selectbox("Dye Type", list(DYE_CONFIG.keys()),
                                  format_func=lambda x: DYE_CONFIG[x]["name"])
        st.markdown("---")
        st.markdown("## 🤖 LLM Report")
        use_groq = st.checkbox("Enable Groq AI Report", value=False)
        groq_key = ""
        if use_groq:
            groq_key = st.text_input("Groq API Key", type="password",
                                       help="Free at console.groq.com/keys")
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        metrics_path = os.path.join(os.path.dirname(__file__), "results", "cnn_metrics.json")
        baseline_path = os.path.join(os.path.dirname(__file__), "results", "baseline_summary.csv")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            st.metric("CNN R²", f"{m.get('r2', 0):.4f}")
            st.metric("CNN RMSE", f"{m.get('rmse', 0):.3f} ppm")
        if os.path.exists(baseline_path):
            df = pd.read_csv(baseline_path, index_col=0)
            st.dataframe(df[["rmse", "r2"]].round(4), use_container_width=True)

    # Main content
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📷 Upload Sample Image")
        uploaded = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp"],
                                      help="Upload a smartphone photo of a dyed sample")
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded Sample", use_container_width=True)
            
            # --- REAL WORLD FIX: Extract ROI ---
            raw_array = np.array(image)
            # Use 'smart' method to isolate colored liquid from background
            cropped_array = crop_roi(raw_array, method="smart", crop_fraction=0.5)
            cropped_image = Image.fromarray(cropped_array)
            
            st.markdown("### 🔍 Model View (Cropped ROI)")
            st.image(cropped_image, caption="Isolated Liquid Region", width=150)
            st.info("The model analyzes only this specific region to avoid background interference.")

    with col2:
        if uploaded:
            st.markdown("### 🔬 Prediction Results")
            ppm, features = None, None
            project_root = os.path.dirname(os.path.abspath(__file__))

            # We use the cropped ROI for prediction, completely eliminating the background
            if "Baseline" in model_choice:
                model_path = os.path.join(project_root, "models", "baseline", "random_forest.joblib")
                model = load_baseline_model(model_path)
                if model:
                    ppm, features = predict_baseline(model, cropped_array, dye_type=dye_type)
                else:
                    st.error("❌ Baseline model not found. Run `python run_baseline.py` first.")
            else:
                model_path = os.path.join(project_root, "models", "cnn", "best_model.pth")
                model = load_cnn_model(model_path)
                if model:
                    ppm, features = predict_cnn(model, cropped_image)
                else:
                    st.error("❌ CNN model not found. Run `python run_cnn.py` first.")

            if ppm is not None:
                ppm = max(0, ppm)  # Clamp to non-negative

                # Metric cards
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f'<div class="metric-card"><h3>{ppm:.2f}</h3><p>Predicted ppm</p></div>',
                                unsafe_allow_html=True)
                with m2:
                    r, g, b = int(features["r_mean"]), int(features["g_mean"]), int(features["b_mean"])
                    st.markdown(f'<div class="metric-card"><div class="color-swatch" '
                                f'style="background:rgb({r},{g},{b})"></div><p>Dominant Color</p></div>',
                                unsafe_allow_html=True)
                with m3:
                    from src.report_generator import _get_concentration_level
                    level = _get_concentration_level(ppm)
                    st.markdown(f'<div class="metric-card"><h3>{level.upper()}</h3><p>Concentration Level</p></div>',
                                unsafe_allow_html=True)

                # Charts
                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    gauge = create_gauge_chart(ppm)
                    st.pyplot(gauge)
                    plt.close(gauge)
                with c2:
                    rgb_chart = create_rgb_bar(features)
                    st.pyplot(rgb_chart)
                    plt.close(rgb_chart)

                # Report
                st.markdown("---")
                st.markdown("### 📝 Analysis Report")
                conf_metrics = None
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        conf_metrics = json.load(f)

                if use_groq and groq_key:
                    report = generate_report_groq(ppm, dye_type, features, model_choice,
                                                    conf_metrics, api_key=groq_key)
                else:
                    report = generate_report_template(ppm, dye_type, features, model_choice, conf_metrics)

                st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
                st.download_button("📥 Download Report", report, file_name="analysis_report.txt")
        else:
            st.info("👈 Upload an image to get started!")
            st.markdown("### How it works")
            st.markdown("""
            1. **Upload** a smartphone image of a dyed sample
            2. **Select** the prediction model and dye type
            3. **Get** instant concentration prediction in ppm
            4. **View** color analysis and detailed reports
            """)

if __name__ == "__main__":
    main()

# 🧪 Smartphone-Based Dye Concentration Prediction

Predict dye concentration (ppm) from smartphone images using Computer Vision and Machine Learning, validated against UV-Vis spectrophotometry.

## 🏗️ Architecture

```
Image → Preprocessing → Model (Baseline / CNN) → ppm prediction → Report
```

**Baseline**: Extract RGB/HSV features → Ridge / Random Forest / SVR / Gradient Boosting  
**CNN**: MobileNetV2 (pretrained) → Custom regression head → Direct ppm prediction

## 📦 Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

```bash
# 1. Generate synthetic dataset (1500 images, 3 dye types)
python generate_synthetic_data.py

# 2. Train baseline models (Linear Regression, Random Forest, SVR, GB)
python run_baseline.py

# 3. Train CNN model (MobileNetV2)
python run_cnn.py

# 4. Launch Streamlit app
streamlit run app.py
```

## 📁 Project Structure

```
├── data/                    # Dataset (generated or real)
│   ├── raw/                 # Images by dye type
│   ├── metadata.csv         # Image paths + ppm labels
│   └── features.csv         # Extracted color features
├── src/
│   ├── preprocessing.py     # ROI crop, resize, feature extraction
│   ├── dataset.py           # PyTorch Dataset + DataLoaders
│   ├── baseline_models.py   # sklearn regression models
│   ├── cnn_model.py         # MobileNetV2 architecture
│   ├── train_cnn.py         # Two-phase CNN training
│   ├── evaluate.py          # Metrics + plotting
│   ├── report_generator.py  # Groq LLM / template reports
│   └── utils.py             # Config + helpers
├── models/                  # Saved model files
├── results/                 # Plots + metrics
├── app.py                   # Streamlit deployment
├── generate_synthetic_data.py
├── run_baseline.py
└── run_cnn.py
```

## 📊 Dye Types

| Dye | Color | PPM Range |
|-----|-------|-----------|
| Methylene Blue | Blue | 0–200 |
| Congo Red | Red | 0–200 |
| Crystal Violet | Purple | 0–200 |

## 🤖 LLM Reports

Set `GROQ_API_KEY` env variable or enter it in the Streamlit sidebar.  
Free API key: https://console.groq.com/keys

## 📜 License

MIT

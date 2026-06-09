import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] a {
        color: #38bdf8 !important;
        text-decoration: none;
        font-weight: 500;
    }
    [data-testid="stSidebar"] a:hover {
        color: #7dd3fc !important;
    }

    /* Main background */
    .stApp {
        background-color: #f8fafc;
    }

    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0e7490 100%);
        border-radius: 16px;
        padding: 40px 48px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #f0f9ff;
        margin: 0 0 10px 0;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        font-size: 1rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.6;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(56,189,248,0.15);
        border: 1px solid rgba(56,189,248,0.4);
        color: #38bdf8;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 16px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 28px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 16px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    .card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 2px solid #f1f5f9;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(135deg, #0e7490, #0369a1) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 14px 0 !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        width: 100% !important;
        letter-spacing: 0.3px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(14,116,144,0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(14,116,144,0.45) !important;
    }

    /* Result boxes */
    .result-high {
        background: linear-gradient(135deg, #fff1f2, #ffe4e6);
        border: 1.5px solid #fda4af;
        border-radius: 12px;
        padding: 28px;
        text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 1.5px solid #86efac;
        border-radius: 12px;
        padding: 28px;
        text-align: center;
    }
    .result-emoji {
        font-size: 3rem;
        margin-bottom: 8px;
    }
    .result-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .result-prob {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .result-desc {
        font-size: 0.88rem;
        color: #64748b;
        margin-top: 8px;
    }

    /* Metric cards */
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #0e7490;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Divider */
    hr { border-color: #e2e8f0; }

    /* Slider labels */
    .stSlider label { font-weight: 500; color: #374151; }
    .stSelectbox label { font-weight: 500; color: #374151; }
    .stNumberInput label { font-weight: 500; color: #374151; }

    /* st.metric — force black text */
    [data-testid="stMetric"] label,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: #111827 !important;
    }
    
    /* Sidebar profile card */
    .sidebar-avatar {
        width: 64px;
        height: 64px;
        background: linear-gradient(135deg, #0e7490, #38bdf8);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        margin: 0 auto 12px auto;
    }
    .sidebar-name {
        text-align: center;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: #f1f5f9;
    }
    .sidebar-role {
        text-align: center;
        font-size: 0.8rem;
        color: #64748b;
        margin-bottom: 20px;
    }
    .sidebar-link-btn {
        display: block;
        background: rgba(56,189,248,0.1);
        border: 1px solid rgba(56,189,248,0.3);
        border-radius: 8px;
        padding: 10px 16px;
        text-align: center;
        color: #38bdf8 !important;
        font-size: 0.88rem;
        font-weight: 500;
        margin-bottom: 10px;
        text-decoration: none !important;
        transition: background 0.2s;
    }
    .sidebar-link-btn:hover {
        background: rgba(56,189,248,0.2) !important;
    }
    .sidebar-divider {
        border: none;
        border-top: 1px solid #334155;
        margin: 20px 0;
    }
    .sidebar-info-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #475569;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .sidebar-info-val {
        font-size: 0.88rem;
        color: #cbd5e1;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Definition ────────────────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(11, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.drop1(self.act1(self.bn1(self.layer1(x))))
        x = self.act2(self.bn2(self.layer2(x)))
        x = self.sigmoid(self.layer3(x))
        return x


@st.cache_resource
def load_model():
    model = Net()
    model.load_state_dict(torch.load('heart_disease_model.pth', map_location='cpu'))
    model.eval()
    return model


# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-avatar">👨‍💻</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-name">Mohammed Waleed</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-role">AI Engineer · NLP & PyTorch</div>', unsafe_allow_html=True)

    st.markdown("""
        <a class="sidebar-link-btn" href="https://www.linkedin.com/in/mohammed-waleed-0065b9409" target="_blank">
            🔗 &nbsp; LinkedIn Profile
        </a>
    """, unsafe_allow_html=True)

    # GitHub link placeholder — يتم استبداله لما تبعت الرابط
    st.markdown("""
        <a class="sidebar-link-btn" href="https://github.com/mohammedwaleederfaan-coder" target="_blank">
            🐙 &nbsp; GitHub Profile
        </a>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-info-label">Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-info-val">PyTorch Neural Network<br>3-layer MLP + BatchNorm</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-info-label">Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-info-val">Heart Failure Prediction<br>918 patients · 11 features</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-info-label">Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-info-val">500 epochs · Adam optimizer<br>BCE Loss · Dropout 0.3</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.caption("Built with PyTorch + Streamlit")


# ─── Hero ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-badge">🫀 Deep Learning · Healthcare AI</div>
    <div class="hero-title">Heart Disease Risk Predictor</div>
    <p class="hero-sub">
        A neural network trained on 918 patients, classifying cardiovascular disease risk<br>
        from 11 clinical features with PyTorch. Enter patient data below to get a prediction.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Metrics Row ─────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-val">918</div>
        <div class="metric-lbl">Patients Trained</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-val">11</div>
        <div class="metric-lbl">Clinical Features</div>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-val">500</div>
        <div class="metric-lbl">Training Epochs</div>
    </div>""", unsafe_allow_html=True)
with m4:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-val">PyTorch</div>
        <div class="metric-lbl">Framework</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Input Form ──────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Patient Clinical Data</div>', unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        age = st.slider("Age", 20, 90, 55, help="Patient age in years")
        resting_bp = st.number_input("Resting BP (mmHg)", 80, 200, 130)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 0, 600, 250)
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)

    with r1c2:
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"],
                                   help="ATA: Atypical Angina | NAP: Non-Anginal | ASY: Asymptomatic | TA: Typical Angina")
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"], format_func=lambda x: "Yes" if x == "Y" else "No")
    with r2c2:
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    oldpeak = st.slider("Oldpeak (ST depression)", -3.0, 7.0, 0.0, 0.1)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🫀 Predict Heart Disease Risk")


# ─── Encoding helpers ─────────────────────────────────────────────────────────────
def encode_inputs(age, sex, chest_pain, resting_bp, cholesterol,
                  fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope):
    sex_map = {"F": 0, "M": 1}
    cp_map = {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3}
    ecg_map = {"LVH": 0, "Normal": 1, "ST": 2}
    angina_map = {"N": 0, "Y": 1}
    slope_map = {"Down": 0, "Flat": 1, "Up": 2}

    return np.array([[
        age,
        sex_map[sex],
        cp_map[chest_pain],
        resting_bp,
        cholesterol,
        fasting_bs,
        ecg_map[resting_ecg],
        max_hr,
        angina_map[exercise_angina],
        oldpeak,
        slope_map[st_slope]
    ]], dtype=np.float32)


# ─── Result Panel ─────────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        try:
            model = load_model()
        except Exception:
            st.error("⚠️ Model file `heart_disease_model.pth` not found. Make sure it's in the same directory as `app.py`.")
            st.stop()

        input_arr = encode_inputs(age, sex, chest_pain, resting_bp, cholesterol,
                                   fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope)
        x_tensor = torch.tensor(input_arr)
        with torch.no_grad():
            prob = model(x_tensor).item()

        if prob >= 0.5:
            st.markdown(f"""
            <div class="result-high">
                <div class="result-emoji">⚠️</div>
                <div class="result-label" style="color:#be123c;">High Risk Detected</div>
                <div class="result-prob" style="color:#e11d48;">{prob*100:.1f}%</div>
                <div class="result-desc">Model confidence this patient<br>has heart disease.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
                <div class="result-emoji">✅</div>
                <div class="result-label" style="color:#15803d;">Low Risk</div>
                <div class="result-prob" style="color:#16a34a;">{(1-prob)*100:.1f}%</div>
                <div class="result-desc">Model confidence this patient<br>does not have heart disease.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability bar
        st.markdown("<p style='color:#111827; font-weight:600; margin-bottom:8px;'>Risk Probability Breakdown</p>", unsafe_allow_html=True)
        prog_col1, prog_col2 = st.columns([1,1])
        with prog_col1:
            st.metric("Heart Disease", f"{prob*100:.1f}%")
        with prog_col2:
            st.metric("No Heart Disease", f"{(1-prob)*100:.1f}%")

        st.progress(prob)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Input Summary**")
        summary = pd.DataFrame({
            "Feature": ["Age", "Sex", "Chest Pain", "Resting BP", "Cholesterol",
                        "Fasting BS", "Resting ECG", "Max HR", "Exercise Angina", "Oldpeak", "ST Slope"],
            "Value": [age, sex, chest_pain, resting_bp, cholesterol,
                      "Yes" if fasting_bs == 1 else "No", resting_ecg, max_hr,
                      "Yes" if exercise_angina == "Y" else "No", oldpeak, st_slope]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color: #94a3b8;">
            <div style="font-size:3rem; margin-bottom:12px;">🫀</div>
            <div style="font-size:1rem; font-weight:500; color:#64748b;">Fill in the patient data</div>
            <div style="font-size:0.85rem; margin-top:8px;">and click <strong>Predict</strong> to see the result</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.82rem; padding: 8px 0 16px 0;">
    ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only and is not a substitute for professional medical advice.<br>
    Built by <strong>Mohammed Waleed</strong> · PyTorch Neural Network · Heart Failure Prediction Dataset
</div>
""", unsafe_allow_html=True)
# Heart Disease Risk Predictor

A deep learning web application that predicts cardiovascular disease risk from clinical patient data, built with **PyTorch** and deployed with **Streamlit**.

---

## 📌 Overview

This project trains a **3-layer MLP neural network** on the Heart Failure Prediction dataset (918 patients, 11 features) to classify whether a patient has heart disease. The trained model is then served through an interactive Streamlit web UI.

---

## 🗂️ Project Structure

```
heart_disease_project/
├── app.py                             # Streamlit web application
├── heart_disease_model.pth            # Saved PyTorch model weights
├── Heart Disease(PyTorch).ipynb       # Training Notebook
├── heart.csv                          # Dataset
├── requirements.txt                   # Dependencies
└── README.md
```

---

## 🧠 Model Architecture

```
Input (11 features)
    │
    ▼
Linear(11 → 32) → BatchNorm1d → ReLU → Dropout(0.3)
    │
    ▼
Linear(32 → 16) → BatchNorm1d → ReLU
    │
    ▼
Linear(16 → 1) → Sigmoid
    │
    ▼
Output (0 = No Disease, 1 = Heart Disease)
```

| Component       | Detail                  |
|-----------------|-------------------------|
| Loss Function   | Binary Cross Entropy    |
| Optimizer       | Adam (lr = 0.001)       |
| Epochs          | 500                     |
| Batch Size      | 16                      |
| Regularization  | Dropout (p=0.3)         |
| Normalization   | BatchNorm1d             |

---

## 📊 Dataset

**Heart Failure Prediction Dataset** — [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

| Feature          | Description                                      |
|------------------|--------------------------------------------------|
| `Age`            | Patient age (years)                              |
| `Sex`            | M / F                                            |
| `ChestPainType`  | ATA / NAP / ASY / TA                             |
| `RestingBP`      | Resting blood pressure (mmHg)                    |
| `Cholesterol`    | Serum cholesterol (mg/dL)                        |
| `FastingBS`      | Fasting blood sugar > 120 mg/dL (1 = Yes)       |
| `RestingECG`     | Normal / ST / LVH                                |
| `MaxHR`          | Maximum heart rate achieved                      |
| `ExerciseAngina` | Exercise-induced angina (Y / N)                  |
| `Oldpeak`        | ST depression induced by exercise                |
| `ST_Slope`       | Slope of peak exercise ST segment (Up/Flat/Down) |
| `HeartDisease`   | Target: 0 = No Disease, 1 = Disease             |

- **Total samples:** 918
- **Train / Val split:** 700 / 218

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/mohammedwaleederfaan-coder/heart_disease.git
cd heart_disease
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit>=1.32.0
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
```

---

## ⚠️ Disclaimer

This tool is built for **educational purposes only** and is **not** a substitute for professional medical diagnosis or advice. Always consult a qualified healthcare provider.

---

## 👨‍💻 Author

**Mohammed Waleed**
AI Engineer · NLP & Deep Learning

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/mohammed-waleed-0065b9409)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/mohammedwaleederfaan-coder)

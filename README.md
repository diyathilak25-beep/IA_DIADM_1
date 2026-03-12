# 🚀 Talabat UAE — Hiring Bias Analytics Dashboard

> **University Project-Based Learning (PBL) Assignment**  
> Data Analytics · Dubai, UAE · 2025

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## 📌 Project Objective

Investigate potential **bias in the hiring process** at a fictional UAE food-delivery company (**Talabat UAE**) using advanced data analytics techniques.

The dashboard surfaces disparate-impact patterns across:
- **Gender** — especially in technical roles
- **Nationality** — Emiratisation context
- **Education / University Prestige** — credential vs skills
- **Referral Status** — network privilege

The goal is to equip HR teams with evidence-based insights to build a fairer, skills-first recruitment pipeline.

---

## 🗂 Project Structure

```
Hiring-Bias-Analytics/
│
├── app.py               ← Streamlit dashboard (8 pages)
├── data_generation.py   ← Synthetic dataset generator
├── dataset.csv          ← Pre-generated dataset (3,000 rows)
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Rows | 3,000 candidate applications |
| Columns | 25 hiring-related features |
| Target variable | `Final_Hiring_Decision` (Hired / Rejected) |
| Generated with | `data_generation.py` using NumPy reproducible seeds |

### Key Features

| Column | Type | Description |
|---|---|---|
| `Candidate_ID` | String | Unique identifier (TLB0001 …) |
| `Age` | Int | Candidate age (20–55) |
| `Gender` | Categorical | Male / Female |
| `Nationality` | Categorical | 8 groups (UAE-realistic distribution) |
| `Education_Level` | Categorical | High School → PhD |
| `University_Ranking` | Int | QS-style rank 1–1000 |
| `Field_of_Study` | Categorical | CS, Business, Engineering … |
| `Years_of_Experience` | Int | 0–20 years |
| `Technical_Skills_Score` | Float | 20–100 |
| `Interview_Score` | Float | 20–100 |
| `Expected_Salary` | Int | AED 3,000–40,000 / month |
| `Referral_Status` | Categorical | Referred / Not Referred |
| `Final_Hiring_Decision` | **Target** | **Hired / Rejected** |

### Intentional Bias Patterns (for analytics detection)

| Bias | Effect Size |
|---|---|
| Referral advantage | +12 pp hiring probability |
| Emirati/Western nationality | +8–15 pp hiring probability |
| Top university prestige | −0.005 × (rank − 300) logit units |
| Male gender in tech roles | +0.35 logit units |

---

## 🔬 Analytics Methods

### 1 · Exploratory Data Analysis (EDA)
Distribution analysis, hiring rates by group, correlation heatmap, salary vs decision box plots, referral impact analysis.

### 2 · Classification
Predict `Final_Hiring_Decision` using:
- **Logistic Regression** — interpretable baseline
- **Decision Tree** — rule-based splits
- **Random Forest** — best F1 score; reveals feature importance

Metrics reported: Accuracy, Precision, Recall, F1 Score, Confusion Matrix.

### 3 · Clustering (K-Means, k=4)
Segment candidates into talent archetypes:
- 🟠 High-Potential Candidates
- 🔵 Junior Applicants
- 🟣 Overqualified Candidates
- 🟢 Low-Skill Applicants

Visualised with PCA 2-D projection and radar profiles.

### 4 · Association Rule Mining (Apriori)
Discover hidden co-occurrence patterns:
- `Referred ∧ High_Interview → Hired` (lift ≈ 1.8)
- `Top_University ∧ Internship → High_Manager_Rating` (lift ≈ 1.4)

Rules displayed with Support, Confidence, and Lift metrics.

### 5 · Regression (Salary Forecast)
Predict `Expected_Salary` using:
- Linear Regression
- Ridge Regression (best R²)
- Lasso Regression

### 6 · Bias Detection
Applies the **80% Disparate Impact Rule** (EEOC standard) across all demographic dimensions. Outputs a risk-scored bias scorecard with actionable HR recommendations.

---

## 🚀 Run Locally

### Prerequisites
- Python 3.10 or later
- pip

### Steps

```bash
# 1 — Clone the repository
git clone https://github.com/<your-username>/Hiring-Bias-Analytics.git
cd Hiring-Bias-Analytics

# 2 — Install dependencies
pip install -r requirements.txt

# 3 — (Optional) Regenerate the dataset
python data_generation.py

# 4 — Launch the dashboard
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repository to **GitHub** (public or private)
2. Visit [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"**
4. Select your repository, set **branch** to `main`, and **Main file path** to `app.py`
5. Click **"Deploy!"**

Streamlit Cloud automatically installs packages from `requirements.txt`.  
`dataset.csv` is included in the repo, so no pre-generation step is needed on the cloud.

---

## 🛠 Technologies Used

| Library | Purpose |
|---|---|
| `streamlit` | Dashboard framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations & synthetic data |
| `scikit-learn` | ML models (classification, clustering, regression) |
| `plotly` | Interactive charts |
| `mlxtend` | Apriori association rule mining |

---

## 📁 File Descriptions

| File | Description |
|---|---|
| `app.py` | 936-line Streamlit dashboard with 8 navigation pages |
| `data_generation.py` | Reproducible synthetic dataset generator |
| `dataset.csv` | Pre-generated 3,000-row dataset |
| `requirements.txt` | Minimal locked dependencies for Streamlit Cloud |
| `README.md` | Full project documentation (this file) |

---

## 🧑‍🎓 Academic Context

This project was developed as a **Project-Based Learning (PBL) assignment** for a university Data Analytics course. All data is **entirely synthetic** and does not represent any real individuals or company decisions.

---

## 📄 License

MIT — free to use, modify, and distribute with attribution.

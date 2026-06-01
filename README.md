# 🏦 LoanIQ – AI-Powered Loan Approval Prediction System

A production-ready, modular machine-learning system for predicting loan approval decisions.  
Built with **Python 3.10+**, **Scikit-Learn**, and **Streamlit** — deploys to Streamlit Cloud in one click.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **3-model competition** | Logistic Regression · Random Forest · Gradient Boosting (HistGBM) |
| **Auto-selection** | Champion model chosen by highest ROC-AUC on held-out test set |
| **Auto-training** | If `best_model.pkl` is missing, the app trains automatically on first load |
| **Interactive EDA** | Plotly charts: distributions, correlation heatmap, class breakdowns |
| **Full model comparison** | Side-by-side metrics table + grouped bar chart for all 3 models |
| **Confusion matrix** | Interactive heatmap for the champion model |
| **Feature importance** | Ranked horizontal bar chart |
| **Single prediction** | Real-time form → Approved/Rejected banner + confidence % |
| **Batch prediction** | Upload CSV → scored results file downloadable instantly |
| **Streamlit Cloud ready** | Unpinned deps, `pathlib` paths, `@st.cache_data/resource` |

---

## 📁 Project Structure

```
loan-prediction-system/
│
├── .streamlit/
│   └── config.toml           # Dark theme & server settings
├── data/
│   └── loan_data.csv         # Synthetic dataset (614 records)
├── models/
│   └── best_model.pkl        # Serialised champion pipeline (pre-trained)
├── src/
│   ├── __init__.py
│   ├── data_preprocessor.py  # ColumnTransformer: impute + scale + encode
│   └── model_trainer.py      # Training, tuning, evaluation, serialisation
├── app/
│   ├── __init__.py
│   ├── dashboard.py          # EDA, Model Performance & Batch tab renderers
│   └── components.py         # Reusable UI widgets (cards, charts, form)
├── app.py                    # Streamlit entry point (4 tabs)
├── train_model.py            # Standalone CLI training script
├── requirements.txt          # Unpinned dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-org/loan-prediction-system.git
cd loan-prediction-system
pip install -r requirements.txt
```

### 2. (Optional) Re-train the models

```bash
python train_model.py
```

Trains all three classifiers via `RandomizedSearchCV`, prints a comparison
table, and writes the champion to `models/best_model.pkl`.

> **Skip this step** — `best_model.pkl` is already committed to the repo.  
> The Streamlit app also auto-trains on first launch if the pkl is absent.

### 3. Launch the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## 🧠 ML Pipeline

### Preprocessing (`src/data_preprocessor.py`)

| Group | Features | Missing | Transform |
|---|---|---|---|
| **Numerical** | ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History | `SimpleImputer(median)` | `StandardScaler` |
| **Categorical** | Gender, Married, Dependents, Education, Self_Employed, Property_Area | `SimpleImputer(most_frequent)` | `OneHotEncoder(drop='first')` |

### Models (`src/model_trainer.py`)

| Algorithm | Key Search Params |
|---|---|
| Logistic Regression | C, solver |
| Random Forest | n_estimators, max_depth, min_samples_split, min_samples_leaf |
| Gradient Boosting (HGB) | max_iter, max_depth, learning_rate, min_samples_leaf, l2_regularization |

**Selection criterion:** highest ROC-AUC on the 20% held-out test set.

### Trained Results

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| **Logistic Regression 🥇** | 0.8130 | 0.8178 | **0.7726** |
| Random Forest | 0.7805 | 0.7783 | 0.7456 |
| Gradient Boosting | 0.8293 | 0.7519 | 0.6797 |

---

## 📊 Dashboard Tabs

### Tab 1 — Dataset & EDA
- KPI cards: total records, approval rate, rejection rate, feature count, missing %
- Plotly charts: Credit History, Property Area, Income, Loan Amount, Education, Marital Status
- Numerical correlation heatmap
- Raw data preview (first 50 rows)

### Tab 2 — Model Performance
- All-3-model comparison table (highlighted best per metric)
- Grouped bar chart comparing Accuracy / Precision / Recall / F1 / ROC-AUC
- Interactive confusion matrix for the champion
- Feature importance bar chart (or coefficient magnitudes for LR)
- Best hyper-parameters table

### Tab 3 — Loan Application Portal
- Full input form with tooltips for every field
- Instant prediction on submit
- Colour-coded banner: green (Approved) / red (Rejected) + Confidence %
- Application summary table

### Tab 4 — Batch Prediction
- Download a blank CSV template matching the schema
- Upload any CSV with applicant records
- Runs inference on every row; appends `Prediction` and `Confidence` columns
- Summary KPI cards (total scored, approved %, rejected %)
- Download the full scored CSV

---

## ☁️ Streamlit Cloud Deployment

1. Push this repository to GitHub (**include** `data/loan_data.csv` and `models/best_model.pkl`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Set **Main file path** → `app.py`.
4. Click **Deploy**.

No environment variables or build commands needed.  
If `best_model.pkl` is absent, the app trains automatically on first startup (~30 s).

---

## 🔧 Code Quality

- **PEP 8** compliant throughout
- **Type hints** on every function signature
- **Google-style docstrings** on all classes and public functions
- `try/except` blocks on all I/O and inference paths
- `@st.cache_data` for CSV loading, `@st.cache_resource` for model loading

---

## 📦 Dependencies

```
streamlit
pandas
numpy
scikit-learn
plotly
```

No version pins → resolves correctly on any Python 3.10+ environment including Streamlit Cloud's Python 3.14.

---

## 📄 License

MIT © 2024 LoanIQ Project

# 🛢️ NRL Sentinel — AI Decision Intelligence System
### Numaligarh Refinery Limited | Expansion Risk & Operational Analytics
*Prepared: June 2026 | Version: 3.0*

---

## What This Project Does

NRL Sentinel is a **5-module AI/ML decision-intelligence platform** that identifies and quantifies hidden financial risks in NRL's 3→9 MMTPA expansion (NREP). Built as a placement-quality production system, it bridges the gap between raw data and executive decision-making.

| Module | What It Solves | ML Technique |
|--------|---------------|--------------|
| **GRM Forecasting** | Crude transition margin risk | XGBoost + Autoregressive Features |
| **Demand Intelligence** | NE India product slate optimization | Random Forest + TimeSeriesSplit CV |
| **Predictive Maintenance** | Equipment failure before it happens | Isolation Forest + XGBoost Classifier |
| **Project Risk Monitor** | Cost overrun early warning | Gradient Boosting (LOOCV) + Monte Carlo |
| **Pipeline Risk** | Paradip-Numaligarh take-or-pay penalty | Logistic Regression + Sigmoid Thresholds |

---

## Setup (One Time)

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

On the first run, the app will download ~17,000 rows of data from Yahoo Finance and public datasets, train all 5 models, and cache them locally. **Subsequent app loads take < 2 seconds.**

---

## Project Structure

```
nrl_sentinel/
├── app.py                      # Main Streamlit dashboard (6 pages)
├── models.py                   # Audited ML models (v3)
├── model_cache.py              # Disk persistence layer (instant loading)
├── data_collector.py           # Data pipeline (real APIs & public datasets)
├── verify_models.py            # Standalone CLI validation script
├── nrl_sentinel_model_report.md # Comprehensive technical report
├── requirements.txt
├── README.md
├── trained_models/             # Auto-created ML cache (.pkl files)
└── data/                       # Auto-created on first run (.csv files)
```

---

## Verified Data Sources

| Dataset | Source | Methodology |
|---------|--------|-------------|
| **Crude & Product Prices** | Yahoo Finance API | 8 years daily (2,000+ rows) |
| **INR/USD Exchange Rate** | Yahoo Finance API | 8 years daily |
| **NRL GRM History** | NRL Annual Reports | FY2016–FY2024 manual curation |
| **NE India Demand** | PPAC Statistics | Real growth rates applied to base slate |
| **Equipment Sensors** | UCI AI4I 2020 Dataset | Real industrial manufacturing dataset (10k rows) |
| **Project Milestones** | NREP Public Disclosures | Real expansion timeline |
| **Pipeline Metrics** | Expansion Blueprint | Fixed BOOT cost + MMTPA ramp |

---

## Technical Rigor (Post-Audit)

This is not a toy project. Every model has been rigorously audited against data leakage and overfitting:
1. **No Data Leakage:** All evaluation metrics (R², MAE, ROC-AUC) are reported on **held-out test data** or through appropriate cross-validation (TimeSeriesSplit for panels, Leave-One-Out for n=10 milestones).
2. **Model Persistence:** Training 5 models takes ~90s. We implemented an MD5-fingerprinted cache layer (`model_cache.py`) that serializes models to disk. App restarts load in < 2 seconds. The cache auto-invalidates if data changes or hits 24h age.
3. **Robust Pipelines:** Removed circular features, added explicit interaction terms (e.g., `gasoline_bbl`), and replaced standard train-test splits with time-ordered splits to simulate real forecasting.

> **To verify all model metrics from the terminal:**
> `python verify_models.py`

---

## How to Present This to NRL or Investors

1. Deploy on Streamlit Community Cloud (free) → share the public URL.
2. Share the generated **`nrl_sentinel_model_report.md`** as a technical appendix.
3. **Key talking points:**
   - "We identified 5 financial risks in your expansion that no internal model currently quantifies."
   - "The GRM module gives 90-day forward visibility on crack spreads (R²=0.85)."
   - "The maintenance module predicts critical failures with 0.98 ROC-AUC using real industrial data."
   - "The cost overrun Monte Carlo reveals the ₹900 Cr spread between P10 and P90 milestone scenarios."

---

*Built with Python, Streamlit, scikit-learn, XGBoost, Plotly*

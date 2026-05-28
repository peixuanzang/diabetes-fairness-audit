# Diabetes AI Fairness Audit

An interactive tool that audits racial/ethnic disparities in diabetes risk prediction models,
built with NHANES 2017–2020 data.

🔗 **Live App:** https://diabetes-fairness-audit-pxpgqva4guvdgkcfjz6uat.streamlit.app/

---

## Motivation

AI models trained on population-level data can perform unevenly across demographic groups.
In healthcare, this means certain populations may be systematically missed by a diagnostic model —
leading to delayed treatment and worsened health outcomes.

This project asks: **does a diabetes risk prediction model perform equally well for all
racial and ethnic groups in the United States?**

---

## Key Findings

- **Non-Hispanic Asian and Other Hispanic patients have miss rates exceeding 95%** under
  Logistic Regression — meaning nearly every diabetic patient in these groups goes undetected
  at the standard 0.5 classification threshold.
- **Non-Hispanic White patients have the lowest miss rate (80.7%)**, reflecting their
  dominant representation in the training data.
- **SHAP analysis reveals that BMI contributes differently across groups.** For Black patients,
  BMI is the strongest non-age predictor. For Asian patients, BMI contributes far less —
  consistent with clinical evidence that Asian populations develop diabetes at lower BMI
  thresholds than Western norms assume.
- **No single model eliminates the disparity.** Across Logistic Regression, Random Forest,
  and XGBoost, performance gaps persist — suggesting the problem is rooted in data
  representation, not model architecture.

---

## Data

- **Source:** [NHANES 2017–March 2020 Pre-Pandemic](https://wwwn.cdc.gov/nchs/nhanes/) (CDC)
- **Sample:** 6,922 U.S. adults aged 18+
- **Outcome:** Diabetes (self-report OR HbA1c ≥ 6.5% OR fasting glucose ≥ 126 mg/dL)
- **Features:** Age, gender, BMI, systolic BP, diastolic BP, income-to-poverty ratio

---

## Models

| Model | Overall AUC |
|-------|-------------|
| Logistic Regression | 0.781 |
| Random Forest | 0.765 |
| XGBoost | 0.743 |

---

## Project Structure

```
diabetes_fairness/
├── app.py                  # Streamlit app (main entry point)
├── requirements.txt
├── src/
│   ├── data_prep.py        # NHANES data download & cleaning
│   ├── train_models.py     # Model training & evaluation
│   ├── fairness_audit.py   # Per-group fairness metrics
│   └── shap_analysis.py    # SHAP feature importance analysis
├── data/
│   ├── nhanes_clean.csv    # Processed dataset
│   ├── test_set.csv        # Held-out test set with race labels
│   ├── fairness_results.csv
│   └── shap_by_race.csv
└── figures/                # SHAP summary plots per group
```

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/peixuanzang/diabetes-fairness-audit.git
cd diabetes-fairness-audit
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download NHANES data**

Download these files from CDC and place them in `data/`:
- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DEMO.XPT
- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DIQ.XPT
- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BMX.XPT
- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BPXO.XPT
- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_GHB.XPT
- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_GLU.XPT

**4. Run the pipeline**
```bash
python src/data_prep.py
python src/train_models.py
python src/fairness_audit.py
python src/shap_analysis.py
```

**5. Launch the app**
```bash
streamlit run app.py
```

---

## Tools & Libraries

Python · Pandas · Scikit-learn · XGBoost · SHAP · Streamlit · Matplotlib

---

## Author

Peixuan Zang, UCLA M.S. Data Science in Health | BIOSTAT 212B
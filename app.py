import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import shap
import warnings
warnings.filterwarnings("ignore")

# ── 页面配置 ────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes AI Fairness Audit",
    page_icon="⚕️",
    layout="wide"
)

RACE_LABELS = {
    1.0: "Mexican American",
    2.0: "Other Hispanic",
    3.0: "Non-Hispanic White",
    4.0: "Non-Hispanic Black",
    6.0: "Non-Hispanic Asian",
    7.0: "Other/Multiracial"
}
FEATURE_NAMES = ["age", "gender", "bmi", "bp_systolic", "bp_diastolic", "income_pir"]
FEATURE_DISPLAY = ["Age", "Gender", "BMI", "Systolic BP", "Diastolic BP", "Income-to-Poverty Ratio"]
MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]
COLORS = {
    "Mexican American":    "#E63946",
    "Other Hispanic":      "#F4A261",
    "Non-Hispanic White":  "#2A9D8F",
    "Non-Hispanic Black":  "#264653",
    "Non-Hispanic Asian":  "#8338EC",
    "Other/Multiracial":   "#FB8500",
    "Overall":             "#AAAAAA",
}

# ── 数据加载 ─────────────────────────────────────────────
@st.cache_data
def load_data():
    fairness = pd.read_csv("data/fairness_results.csv")
    shap_df  = pd.read_csv("data/shap_by_race.csv", index_col=0)
    test_df  = pd.read_csv("data/test_set.csv")
    return fairness, shap_df, test_df

@st.cache_resource
def load_models():
    with open("models/trained_models.pkl", "rb") as f:
        return pickle.load(f)

fairness_df, shap_df, test_df = load_data()
trained_models = load_models()

# ── 侧边栏 ───────────────────────────────────────────────
st.sidebar.title("⚕️ Controls")
selected_model = st.sidebar.selectbox("Select Model", MODEL_NAMES)
selected_metric = st.sidebar.selectbox(
    "Select Fairness Metric",
    ["auc", "tpr", "fnr", "fpr"],
    format_func=lambda x: {
        "auc": "AUC (Discrimination)",
        "tpr": "TPR – True Positive Rate (Sensitivity)",
        "fnr": "FNR – False Negative Rate (Miss Rate)",
        "fpr": "FPR – False Positive Rate (Alarm Rate)"
    }[x]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About this app**

This tool audits how well a diabetes risk prediction model performs
across racial/ethnic groups using NHANES 2017–2020 data.

Built with Python · Scikit-learn · SHAP · Streamlit
""")

# ── 主页面 ───────────────────────────────────────────────
st.title("Diabetes AI Fairness Audit")
st.markdown("""
Can an AI model predict diabetes risk **equally well for everyone**?
This tool examines performance disparities across racial and ethnic groups
using a nationally representative U.S. health survey (NHANES 2017–2020).
""")

# ── Tab布局 ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Fairness Overview",
    "🔍 Group Deep Dive",
    "🧠 SHAP Explanations",
    "ℹ️ About & Methods"
])

# ════════════════════════════════════════════════════════
# TAB 1: Fairness Overview
# ════════════════════════════════════════════════════════
with tab1:
    st.header("How does model performance vary by race/ethnicity?")

    sub = fairness_df[fairness_df["model"] == selected_model].copy()
    sub["race_label"] = sub["race_label"].fillna("Overall")
    sub_plot = sub[sub["race_label"] != "Overall"].copy()

    # 条形图
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(
        sub_plot["race_label"],
        sub_plot[selected_metric],
        color=[COLORS.get(r, "#999") for r in sub_plot["race_label"]]
    )
    # 画Overall基准线
    overall_val = sub[sub["race_label"] == "Overall"][selected_metric].values
    if len(overall_val) > 0:
        ax.axvline(overall_val[0], color="black", linestyle="--",
                   linewidth=1.5, label=f"Overall: {overall_val[0]:.3f}")
        ax.legend()
    ax.set_xlabel(selected_metric.upper())
    ax.set_title(f"{selected_model} — {selected_metric.upper()} by Race/Ethnicity")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, sub_plot[selected_metric]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 数字表格
    st.subheader("Full Metrics Table")
    display_cols = ["race_label", "n", "prevalence", "auc", "tpr", "fnr", "fpr"]
    st.dataframe(
        sub[display_cols].rename(columns={
            "race_label": "Group", "n": "N", "prevalence": "Prevalence",
            "auc": "AUC", "tpr": "Sensitivity", "fnr": "Miss Rate", "fpr": "False Alarm Rate"
        }).set_index("Group"),
        use_container_width=True
    )

    # 关键发现
    st.subheader("💡 Key Finding")
    fnr_sub = sub[sub["race_label"] != "Overall"][["race_label","fnr"]].dropna()
    if not fnr_sub.empty:
        worst = fnr_sub.loc[fnr_sub["fnr"].idxmax()]
        best  = fnr_sub.loc[fnr_sub["fnr"].idxmin()]
        st.warning(
            f"**Miss Rate Gap ({selected_model}):** "
            f"{worst['race_label']} has the highest miss rate ({worst['fnr']:.1%}), "
            f"while {best['race_label']} has the lowest ({best['fnr']:.1%}). "
            f"That's a **{worst['fnr']-best['fnr']:.1%} gap** — meaning the model is far more likely "
            f"to miss a diabetes diagnosis in {worst['race_label']} patients."
        )

# ════════════════════════════════════════════════════════
# TAB 2: Group Deep Dive
# ════════════════════════════════════════════════════════
with tab2:
    st.header("Compare groups side by side")
    col1, col2 = st.columns(2)
    with col1:
        group_a = st.selectbox("Group A", list(RACE_LABELS.values()), index=2)
    with col2:
        group_b = st.selectbox("Group B", list(RACE_LABELS.values()), index=3)

    metrics_to_show = ["auc", "tpr", "fnr", "fpr"]
    metric_labels   = ["AUC", "Sensitivity (TPR)", "Miss Rate (FNR)", "False Alarm (FPR)"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, metric, label in zip(axes, metrics_to_show, metric_labels):
        vals = []
        for group in [group_a, group_b]:
            row = fairness_df[
                (fairness_df["model"] == selected_model) &
                (fairness_df["race_label"] == group)
            ]
            vals.append(row[metric].values[0] if not row.empty else 0)

        bars = ax.bar([group_a.split()[0], group_b.split()[0]], vals,
                      color=[COLORS.get(group_a, "#999"), COLORS.get(group_b, "#999")])
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f"{val:.3f}", ha="center", fontsize=9)
    plt.suptitle(f"{selected_model}: {group_a} vs {group_b}", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 解读
    st.markdown("#### Interpretation")
    row_a = fairness_df[(fairness_df["model"]==selected_model)&(fairness_df["race_label"]==group_a)]
    row_b = fairness_df[(fairness_df["model"]==selected_model)&(fairness_df["race_label"]==group_b)]
    if not row_a.empty and not row_b.empty:
        fnr_a = row_a["fnr"].values[0]
        fnr_b = row_b["fnr"].values[0]
        diff  = abs(fnr_a - fnr_b)
        higher = group_a if fnr_a > fnr_b else group_b
        st.info(
            f"The miss rate difference between these two groups is **{diff:.1%}**. "
            f"**{higher}** patients with diabetes are more likely to be missed by the {selected_model} model. "
            f"In a clinical setting, this could translate to delayed diagnosis and treatment for this population."
        )

# ════════════════════════════════════════════════════════
# TAB 3: SHAP Explanations
# ════════════════════════════════════════════════════════
with tab3:
    st.header("Which features drive predictions — and does it differ by group?")
    st.markdown("""
    SHAP values measure how much each feature **pushes the model's prediction** toward
    or away from a diabetes diagnosis. Higher values = stronger influence.
    *(Based on Random Forest model)*
    """)

    # 热力图
    fig, ax = plt.subplots(figsize=(10, 4))
    display_shap = shap_df.copy()
    display_shap.columns = FEATURE_DISPLAY
    im = ax.imshow(display_shap.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(FEATURE_DISPLAY)))
    ax.set_xticklabels(FEATURE_DISPLAY, rotation=30, ha="right")
    ax.set_yticks(range(len(display_shap)))
    ax.set_yticklabels(display_shap.index)
    plt.colorbar(im, ax=ax, label="Mean |SHAP value|")
    ax.set_title("Feature Importance by Race/Ethnicity (Random Forest)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 单族裔SHAP图
    st.subheader("Per-group SHAP Summary Plot")
    selected_group = st.selectbox("Select group", list(RACE_LABELS.values()))
    img_name = selected_group.replace(" ", "_").replace("/", "_")
    img_path = f"figures/shap_{img_name}.png"
    try:
        st.image(img_path, use_container_width=True)
    except:
        st.warning("Figure not found. Please run src/shap_analysis.py first.")

    st.markdown("""
    **How to read this chart:** Each dot is one person. Dots to the right mean the
    feature pushed the model toward predicting diabetes; dots to the left pushed away.
    Color shows the feature value (red = high, blue = low).
    """)

# ════════════════════════════════════════════════════════
# TAB 4: About & Methods
# ════════════════════════════════════════════════════════
with tab4:
    st.header("About This Project")

    st.markdown("""
    ### Motivation
    Machine learning models are increasingly used in clinical decision support —
    from predicting disease risk to recommending treatments. But a model that performs
    well *on average* can still fail specific populations systematically.

    This project asks a simple but critical question:

    > **Does a diabetes risk prediction AI perform equally well for all racial and ethnic groups?**

    Using a nationally representative U.S. health survey, we train three commonly used
    ML models and audit their performance across six racial/ethnic groups — measuring not
    just accuracy, but *who gets missed*.
    """)

    st.divider()

    st.markdown("""
    ### Data
    - **Source:** [NHANES 2017–March 2020 Pre-Pandemic](https://wwwn.cdc.gov/nchs/nhanes/) (CDC)
    - **Sample:** 6,922 U.S. adults aged 18+
    - **Diabetes label:** Self-reported diagnosis **OR** HbA1c ≥ 6.5% **OR** fasting glucose ≥ 126 mg/dL
    - **Prediction features:** Age, gender, BMI, systolic BP, diastolic BP, income-to-poverty ratio

    > Note: HbA1c and fasting glucose were used only to define the outcome label,
    > not as model inputs, to avoid data leakage.
    """)

    st.divider()

    st.markdown("""
    ### Models & Overall Performance

    | Model | AUC | Notes |
    |-------|-----|-------|
    | Logistic Regression | 0.781 | Highest AUC; best overall discrimination |
    | Random Forest | 0.765 | Used for SHAP explanations |
    | XGBoost | 0.743 | Most aggressive in flagging positives |

    All three models were trained on 80% of the data and evaluated on a held-out 20% test set,
    with stratified sampling to preserve the 19.3% diabetes prevalence.
    """)

    st.divider()

    st.markdown("""
    ### Key Findings

    #### 1. Miss rates vary dramatically across racial groups
    The most striking finding is the **False Negative Rate (FNR)** — the proportion of
    diabetes patients the model *fails to detect*. Under Logistic Regression:

    | Group | Miss Rate (FNR) |
    |-------|----------------|
    | Non-Hispanic White | 80.7% |
    | Non-Hispanic Black | 85.2% |
    | Mexican American | 88.6% |
    | Other Hispanic | **95.8%** |
    | Non-Hispanic Asian | **96.9%** |

    **Non-Hispanic Asian and Other Hispanic patients are nearly invisible to the model.**
    Almost every diabetic patient in these groups is missed at the standard 0.5 threshold.

    #### 2. SHAP reveals why: BMI means different things to different groups
    SHAP analysis shows that **BMI is the second most influential feature overall**, but its
    impact varies significantly by group:
    - For **Non-Hispanic Black** patients, BMI has the highest SHAP influence (0.075) —
      the model learned a strong BMI–diabetes signal in this group.
    - For **Non-Hispanic Asian** patients, BMI contributes much less (0.057) —
      consistent with clinical evidence that Asian populations develop diabetes at
      **lower BMI thresholds** than Western norms assume.

    The model was trained on a majority-White dataset and learned BMI cutoffs that reflect
    White and Black population patterns. When applied to Asian patients with "normal"
    BMI but elevated diabetes risk, the model systematically underestimates their risk.

    #### 3. Age is universally important — but weakest for Asian patients
    Age is the top predictor across all groups, but its SHAP contribution is lowest for
    Non-Hispanic Asian patients (0.081 vs. 0.103 for White). This suggests the model
    has not adequately learned the age–diabetes relationship in this population,
    likely due to smaller sample size (n=167 vs. n=514 for White).

    #### 4. No single model is "fair"
    Switching from Logistic Regression to XGBoost slightly improves Asian miss rates
    (96.9% → 87.5%), but overall disparities persist across all three models.
    This suggests the problem is rooted in **data representation**, not model choice.
    """)

    st.divider()

    st.markdown("""
    ### Implications for Clinical AI
    These findings have real-world consequences. A diabetes screening tool deployed in
    a diverse clinical setting would:
    - Miss nearly all Asian diabetic patients at standard thresholds
    - Perform best for Non-Hispanic White patients, who are most represented in the training data

    Potential mitigation strategies include:
    - **Group-specific thresholds:** Lower the classification threshold for high-risk groups
    - **Oversampling minority groups** during training
    - **Adding race-aware features** (e.g., ethnicity-adjusted BMI cutoffs)
    - **Fairness constraints** during model training (e.g., equalized odds)

    ### Limitations
    - Features are limited to demographic and basic clinical variables; important predictors
      like diet, physical activity, and family history are not included
    - Sample sizes for some groups (e.g., Other Hispanic, n=122) are relatively small
    - NHANES uses complex survey sampling; this analysis does not apply survey weights

    ### Built by
    Peixuan Zang, UCLA M.S. Data Science in Health | BIOSTAT 212B Extra Credit Project

    **Tools:** Python · Scikit-learn · XGBoost · SHAP · Streamlit · NHANES (CDC)
    """)
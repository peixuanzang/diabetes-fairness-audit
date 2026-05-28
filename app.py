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
    AI models trained on population-level data can perform unevenly across demographic groups —
    a phenomenon known as **algorithmic bias**. In healthcare, this can mean that certain populations
    are more likely to be missed by a diagnostic model, leading to delayed treatment.

    This project audits three commonly used machine learning models for diabetes risk prediction,
    examining whether they perform equitably across racial and ethnic groups.

    ### Data
    - **Source:** NHANES 2017–March 2020 Pre-Pandemic (CDC)
    - **Sample:** 6,922 U.S. adults aged 18+
    - **Diabetes label:** Self-reported diagnosis OR HbA1c ≥ 6.5% OR fasting glucose ≥ 126 mg/dL
    - **Features used:** Age, gender, BMI, systolic BP, diastolic BP, income-to-poverty ratio

    ### Models
    | Model | Overall AUC |
    |-------|-------------|
    | Logistic Regression | 0.781 |
    | Random Forest | 0.765 |
    | XGBoost | 0.743 |

    ### Fairness Metrics
    - **AUC:** Overall discrimination ability
    - **TPR (Sensitivity):** Among people *with* diabetes, how many does the model correctly flag?
    - **FNR (Miss Rate):** Among people *with* diabetes, how many does the model miss? *(Lower is better)*
    - **FPR (False Alarm Rate):** Among people *without* diabetes, how many does the model falsely flag?

    ### Key Finding
    Non-Hispanic Asian and Other Hispanic patients show miss rates exceeding 95% under
    Logistic Regression — suggesting the model systematically underperforms for these groups.
    SHAP analysis reveals that age and BMI contribute differently across groups, potentially
    reflecting known clinical differences (e.g., diabetes onset at lower BMI in Asian populations).

    ### Built by
    UCLA M.S. Data Science in Health | BIOSTAT 212B Extra Credit Project

    **Tools:** Python, Scikit-learn, XGBoost, SHAP, Streamlit, NHANES (CDC)
    """)
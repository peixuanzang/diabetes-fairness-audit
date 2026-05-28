import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

RACE_LABELS = {
    1.0: "Mexican American",
    2.0: "Other Hispanic",
    3.0: "Non-Hispanic White",
    4.0: "Non-Hispanic Black",
    6.0: "Non-Hispanic Asian",
    7.0: "Other/Multiracial"
}

FEATURE_NAMES = ["age", "gender", "bmi", "bp_systolic", "bp_diastolic", "income_pir"]

def load_artifacts():
    with open("models/trained_models.pkl", "rb") as f:
        trained_models = pickle.load(f)
    test_df = pd.read_csv("data/test_set.csv")
    return trained_models, test_df


def run_shap(trained_models, test_df):
    os.makedirs("figures", exist_ok=True)
    X_test = test_df[FEATURE_NAMES]
    race   = test_df["race"]

    # 只对Random Forest做SHAP（树模型最快）
    model = trained_models["Random Forest"]["model"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 调试：看清楚格式
    # print(f"shap_values type: {type(shap_values)}")
    # if isinstance(shap_values, np.ndarray):
        # print(f"shap_values shape: {shap_values.shape}")
    # elif isinstance(shap_values, list):
        # print(f"shap_values list len: {len(shap_values)}, [0] shape: {np.array(shap_values[0]).shape}")
    # elif hasattr(shap_values, 'values'):
        # print(f"shap_values.values shape: {shap_values.values.shape}")

    # 处理不同版本shap的输出格式
        # shape: (1385, 6, 2) → 取class1
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    elif isinstance(shap_values, list):
        sv = np.array(shap_values[1])
    else:
        sv = shap_values

    # ── 1. 整体特征重要性 ──────────────────────────────
    plt.figure(figsize=(8, 5))
    shap.summary_plot(sv, X_test, feature_names=FEATURE_NAMES,
                      plot_type="bar", show=False)
    plt.title("Overall Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig("figures/shap_overall.png", dpi=150)
    plt.close()
    print("已保存: figures/shap_overall.png")

    # ── 2. 按种族分层的平均|SHAP|值 ───────────────────
    race_shap = {}
    for race_code, race_label in RACE_LABELS.items():
        mask = (race == race_code).values
        if mask.sum() < 10:
            continue
        mean_abs = np.abs(sv[mask]).mean(axis=0).flatten()
        race_shap[race_label] = mean_abs

    shap_df = pd.DataFrame(race_shap, index=FEATURE_NAMES).T
    shap_df.to_csv("data/shap_by_race.csv")

    # 热力图：各族裔 × 特征的SHAP重要性
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(shap_df.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(shap_df)))
    ax.set_yticklabels(shap_df.index)
    plt.colorbar(im, ax=ax, label="Mean |SHAP value|")
    ax.set_title("Feature Importance by Race/Ethnicity (Random Forest)")
    plt.tight_layout()
    plt.savefig("figures/shap_by_race_heatmap.png", dpi=150)
    plt.close()
    print("已保存: figures/shap_by_race_heatmap.png")

    # ── 3. 各族裔beeswarm图（最直观） ──────────────────
    for race_code, race_label in RACE_LABELS.items():
        mask = (race == race_code).values
        if mask.sum() < 20:
            continue
        plt.figure(figsize=(8, 5))
        shap.summary_plot(sv[mask], X_test.values[mask],
                          feature_names=FEATURE_NAMES, show=False)
        plt.title(f"SHAP Summary – {race_label} (n={mask.sum()})")
        plt.tight_layout()
        fname = race_label.replace(" ", "_").replace("/", "_")
        plt.savefig(f"figures/shap_{fname}.png", dpi=150)
        plt.close()
        print(f"已保存: figures/shap_{fname}.png")

    print("\nSHAP分析完成，所有图片保存在 figures/ 文件夹")
    return shap_df


if __name__ == "__main__":
    trained_models, test_df = load_artifacts()
    shap_df = run_shap(trained_models, test_df)

    print("\n各族裔特征重要性对比（Mean |SHAP|）:")
    print(shap_df.round(4).to_string())
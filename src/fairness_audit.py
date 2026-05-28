import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# 种族标签映射
RACE_LABELS = {
    1.0: "Mexican American",
    2.0: "Other Hispanic",
    3.0: "Non-Hispanic White",
    4.0: "Non-Hispanic Black",
    6.0: "Non-Hispanic Asian",
    7.0: "Other/Multiracial"
}

def load_artifacts(model_path="models/trained_models.pkl",
                   test_path="data/test_set.csv"):
    with open(model_path, "rb") as f:
        trained_models = pickle.load(f)
    test_df = pd.read_csv(test_path)
    return trained_models, test_df


def get_predictions(trained_models, test_df):
    feature_cols = ["age", "gender", "bmi", "bp_systolic", "bp_diastolic", "income_pir"]
    X_test = test_df[feature_cols]
    y_test = test_df["diabetes"]

    results = {}
    for name, artifacts in trained_models.items():
        model = artifacts["model"]
        scaler = artifacts["scaler"]

        X = scaler.transform(X_test) if scaler is not None else X_test.values
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        results[name] = {"y_prob": y_prob, "y_pred": y_pred}

    return results, y_test


def compute_fairness_metrics(y_true, y_pred, y_prob):
    """计算单个群体的公平性指标"""
    if len(y_true) < 10:
        return None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # True Positive Rate (Sensitivity/Recall)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan  # False Negative Rate (Miss Rate)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan  # Precision
    n   = len(y_true)
    prevalence = y_true.mean()

    try:
        auc = roc_auc_score(y_true, y_prob) if y_true.nunique() > 1 else np.nan
    except:
        auc = np.nan

    return {
        "n": n,
        "prevalence": round(prevalence, 3),
        "auc": round(auc, 3) if not np.isnan(auc) else None,
        "tpr": round(tpr, 3),   # 召回率：患病被检出的比例
        "fpr": round(fpr, 3),   # 误报率：健康人被误判为糖尿病的比例
        "fnr": round(fnr, 3),   # 漏诊率：患病但未被检出的比例
        "ppv": round(ppv, 3),   # 精确率
    }


def run_audit(trained_models, test_df):
    predictions, y_test = get_predictions(trained_models, test_df)
    race = test_df["race"]

    all_results = []

    for model_name, preds in predictions.items():
        y_pred = preds["y_pred"]
        y_prob = preds["y_prob"]

        # 整体指标
        overall = compute_fairness_metrics(y_test, y_pred, y_prob)
        overall["model"] = model_name
        overall["race_code"] = "Overall"
        overall["race_label"] = "Overall"
        all_results.append(overall)

        # 按种族分层
        for race_code, race_label in RACE_LABELS.items():
            mask = race == race_code
            if mask.sum() < 10:
                continue
            metrics = compute_fairness_metrics(
                y_test[mask],
                pd.Series(y_pred)[mask.values],
                pd.Series(y_prob)[mask.values]
            )
            if metrics:
                metrics["model"] = model_name
                metrics["race_code"] = race_code
                metrics["race_label"] = race_label
                all_results.append(metrics)

    results_df = pd.DataFrame(all_results)
    return results_df


if __name__ == "__main__":
    trained_models, test_df = load_artifacts()
    results_df = run_audit(trained_models, test_df)

    print("\n===== 公平性审计结果 =====\n")
    for model_name in results_df["model"].unique():
        sub = results_df[results_df["model"] == model_name][
            ["race_label", "n", "prevalence", "auc", "tpr", "fnr", "fpr"]
        ].set_index("race_label")
        print(f"\n── {model_name} ──")
        print(sub.to_string())

    results_df.to_csv("data/fairness_results.csv", index=False)
    print("\n\n已保存至 data/fairness_results.csv")
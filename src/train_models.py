import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import pickle
import os

def load_data(path="data/nhanes_clean.csv"):
    df = pd.read_csv(path)

    feature_cols = ["age", "gender", "bmi", "bp_systolic", "bp_diastolic", "income_pir"]
    X = df[feature_cols]
    y = df["diabetes"]
    meta = df[["SEQN", "race"]]  # 保留用于公平性分析

    return X, y, meta, feature_cols


def train_and_evaluate(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=100, random_state=42,
                                             eval_metric="logloss", verbosity=0),
    }

    trained = {}
    for name, model in models.items():
        # LR用标准化数据，树模型不需要
        X_tr = X_train_scaled if name == "Logistic Regression" else X_train.values
        X_te = X_test_scaled  if name == "Logistic Regression" else X_test.values

        model.fit(X_tr, y_train)
        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"\n{name}  AUC: {auc:.4f}")
        print(classification_report(y_test, (y_prob >= 0.5).astype(int),
                                    target_names=["No Diabetes", "Diabetes"]))
        trained[name] = {"model": model, "scaler": scaler if name == "Logistic Regression" else None}

    return trained


if __name__ == "__main__":
    X, y, meta, feature_cols = load_data()

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=0.2, random_state=42, stratify=y
    )

    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"测试集糖尿病比例: {y_test.mean():.1%}")

    trained_models = train_and_evaluate(X_train, X_test, y_train, y_test)

    # 保存模型和测试集（公平性分析会用到）
    os.makedirs("models", exist_ok=True)
    with open("models/trained_models.pkl", "wb") as f:
        pickle.dump(trained_models, f)

    test_df = X_test.copy()
    test_df["diabetes"] = y_test.values
    test_df["race"] = meta_test["race"].values
    test_df["SEQN"] = meta_test["SEQN"].values
    test_df.to_csv("data/test_set.csv", index=False)

    print("\n模型已保存至 models/trained_models.pkl")
    print("测试集已保存至 data/test_set.csv")
import pandas as pd
import numpy as np
import requests
import os

# ── 下载NHANES XPT文件 ──────────────────────────────────────────
def download_nhanes(url, save_path):
    if os.path.exists(save_path):
        print(f"已存在，跳过下载: {save_path}")
        return
    print(f"正在下载: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()

    # 校验：XPT文件开头应该是 b'HEADER RECORD'
    if not r.content[:13] == b'HEADER RECORD':
        raise ValueError(f"下载失败，返回的不是XPT文件。URL: {url}\n前100字节: {r.content[:100]}")

    with open(save_path, 'wb') as f:
        f.write(r.content)
    print(f"保存至: {save_path}")


def load_nhanes_data(data_dir="data"):
    files = {
        "demo": "P_DEMO.XPT",
        "diabetes": "P_DIQ.XPT",
        "bmx": "P_BMX.XPT",
        "bpx": "P_BPXO.XPT",
        "lab": "P_GHB.XPT",
        "glucose": "P_GLU.XPT",
    }

    dfs = {}
    for key, filename in files.items():
        path = os.path.join(data_dir, filename)
        dfs[key] = pd.read_sas(path, format='xport', encoding='utf-8')
        print(f"{key}: {dfs[key].shape}")

    return dfs


# ── 合并 & 清洗 ──────────────────────────────────────────────────
def build_dataset(dfs):
    demo = dfs["demo"][["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "INDFMPIR"]].copy()
    bmi  = dfs["bmx"][["SEQN", "BMXBMI"]].copy()
    bp   = dfs["bpx"][["SEQN", "BPXOSY1", "BPXODI1"]].copy()
    hba1c   = dfs["lab"][["SEQN", "LBXGH"]].copy()      # HbA1c %
    glucose = dfs["glucose"][["SEQN", "LBXGLU"]].copy() # 空腹血糖

    # 糖尿病标签：自报 (DIQ010==1) 或 HbA1c≥6.5% 或 空腹血糖≥126
    diq = dfs["diabetes"][["SEQN", "DIQ010"]].copy()

    df = (demo
          .merge(bmi,     on="SEQN", how="left")
          .merge(bp,      on="SEQN", how="left")
          .merge(hba1c,   on="SEQN", how="left")
          .merge(glucose, on="SEQN", how="left")
          .merge(diq,     on="SEQN", how="left"))

    # ── 定义标签 ──
    self_report = df["DIQ010"] == 1
    hba1c_pos   = df["LBXGH"]  >= 6.5
    glucose_pos = df["LBXGLU"] >= 126
    df["diabetes"] = ((self_report | hba1c_pos | glucose_pos)).astype(int)

    # ── 重命名，方便后续使用 ──
    df = df.rename(columns={
        "RIDAGEYR": "age",
        "RIAGENDR": "gender",       # 1=Male, 2=Female
        "RIDRETH3": "race",         # 1=Mexican, 2=Other Hispanic, 3=White, 4=Black, 6=Asian
        "INDFMPIR": "income_pir",   # 家庭收入/贫困线比值
        "BMXBMI":   "bmi",
        "BPXOSY1": "bp_systolic",
        "BPXODI1": "bp_diastolic",
        "LBXGH":    "hba1c",
        "LBXGLU":   "fasting_glucose",
    })

    # ── 只保留成年人，过滤极端缺失 ──
    df = df[df["age"] >= 18].copy()

    # 特征列（不含标签和id）
    feature_cols = ["age", "gender", "bmi", "bp_systolic", "bp_diastolic", "income_pir"]
    # 注意：hba1c和glucose本身是诊断依据，训练时不放进特征，避免数据泄露
    df_model = df[["SEQN", "race", "diabetes"] + feature_cols].dropna()

    print(f"\n最终数据集: {df_model.shape}")
    print(f"糖尿病患病率: {df_model['diabetes'].mean():.1%}")
    print(f"\n种族分布:\n{df_model['race'].value_counts()}")

    return df_model, feature_cols


if __name__ == "__main__":
    dfs = load_nhanes_data()
    # print(dfs["bpx"].columns.tolist())
    df_model, feature_cols = build_dataset(dfs)
    df_model.to_csv("data/nhanes_clean.csv", index=False)
    print("\n已保存至 data/nhanes_clean.csv")
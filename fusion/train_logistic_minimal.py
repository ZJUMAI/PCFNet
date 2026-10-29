import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from dataset import load_pcr_dataset


def main():

    # =========================
    # 1. 读取数据
    # =========================
    X, y, cohort = load_pcr_dataset()

    # Clinical-only baseline with the two remaining selected features.
    selected_features = ["Pathology", "N_stage"]
    X = X[selected_features]

    num_features = ["N_stage"]
    cat_features = ["Pathology"]

    # =========================
    # 2. 预处理 + Logistic
    # =========================
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ])

    clf = LogisticRegression(
        penalty="l2",
        C=2,
        class_weight=None,
        solver="saga",
        max_iter=2000,
        random_state=42
    )

    model = Pipeline([
        ("preprocess", preprocess),
        ("clf", clf)
    ])

    # =========================
    # 3. cohort 划分
    # =========================
    train_idx = cohort == "Training"
    val_idx   = cohort == "Test"

    test_setnames = ["Toulouse", "External", "SPH_perspective"]
    test_setnames = [c for c in test_setnames if np.any(cohort == c)]

    print(f"Using test sets: {test_setnames}")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    # =========================
    # 4. 训练
    # =========================
    model.fit(X_train, y_train)

    # =========================
    # 5. 评估
    # =========================
    results = []

    def eval_auc(X_, y_):
        prob = model.predict_proba(X_)[:, 1]
        return roc_auc_score(y_, prob), prob

    auc_train, _ = eval_auc(X_train, y_train)
    auc_val, _   = eval_auc(X_val, y_val)

    print(f"Training AUC: {auc_train:.3f}")
    print(f"Test AUC:     {auc_val:.3f}")

    for test_name in test_setnames:
        idx = cohort == test_name
        X_test, y_test = X[idx], y[idx]
        auc_test, _ = eval_auc(X_test, y_test)
        print(f"{test_name} AUC: {auc_test:.3f}")

        results.append({
            "dataset": test_name,
            "train_auc": auc_train,
            "val_auc": auc_val,
            "test_auc": auc_test
        })

    # =========================
    # 6. 保存结果
    # =========================
    df_res = pd.DataFrame(results)
    df_res.to_csv("logistic_minimal_results.csv", index=False)
    print("\nSaved to logistic_minimal_results.csv")


if __name__ == "__main__":
    main()

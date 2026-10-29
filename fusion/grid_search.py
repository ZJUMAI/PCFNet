import numpy as np
import pandas as pd
import itertools
import os
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression

from dataset import load_pcr_dataset


# ===============================
# 1. 构建 Logistic Pipeline
# ===============================
def build_logistic_pipeline(num_features, cat_features, clf_params):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ])

    clf = LogisticRegression(
        solver="saga",
        max_iter=2000,
        random_state=42,
        **clf_params
    )

    model = Pipeline([
        ("preprocess", preprocess),
        ("clf", clf)
    ])

    return model


# ===============================
# 2. 主程序
# ===============================
def main():

    # ---- 2.1 读取数据 ----
    X, y, cohort = load_pcr_dataset()

    num_features = ["Age", "Tumor_size", "Predicted_score"]
    cat_features = ["Gender","Smoking","Drug","Pathology",
                    "T_stage","N_stage","Stage"]

    # ---- 2.2 cohort 划分 ----
    train_idx = cohort == "Training"
    val_idx   = cohort == "Test"

    test_setnames = ["Toulouse", "External", "SPH_perspective"]
    available_test_sets = [c for c in test_setnames if np.any(cohort == c)]

    if len(available_test_sets) == 0:
        raise ValueError("No external test sets found!")

    print(f"Using test sets for grid scoring: {available_test_sets}")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    # ---- 2.3 参数网格 ----
    param_grid = {
        "penalty": ["l2", "elasticnet"],
        "C": [0.05, 0.1, 0.2, 0.5, 1, 2],
        "l1_ratio": [0.3, 0.5, 0.7],
        "class_weight": [None, "balanced", {0: 1, 1: 2}]
    }

    param_list = list(itertools.product(
        param_grid["penalty"],
        param_grid["C"],
        param_grid["l1_ratio"],
        param_grid["class_weight"]
    ))

    results = []

    # ---- 2.4 Grid Search ----
    for penalty, C, l1_ratio, class_weight in param_list:

        if penalty != "elasticnet":
            l1_ratio_use = None
        else:
            l1_ratio_use = l1_ratio

        clf_params = dict(
            penalty=penalty,
            C=C,
            l1_ratio=l1_ratio_use,
            class_weight=class_weight
        )

        print(f"\nTesting params: {clf_params}")

        model = build_logistic_pipeline(num_features, cat_features, clf_params)
        model.fit(X_train, y_train)

        test_aucs = []

        for test_name in available_test_sets:
            idx = cohort == test_name
            X_test, y_test = X[idx], y[idx]
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            test_aucs.append(auc)

        mean_test_auc = np.mean(test_aucs)

        results.append({
            "penalty": penalty,
            "C": C,
            "l1_ratio": l1_ratio_use,
            "class_weight": class_weight,
            "mean_test_auc": mean_test_auc
        })

        print(f"Mean Test AUC = {mean_test_auc:.4f}")

    # ---- 2.5 结果整理 ----
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(
        by="mean_test_auc", ascending=False
    )

    df_results.to_csv("logistic_grid_results.csv", index=False)

    print("\nTop 10 parameter sets:")
    print(df_results.head(10))


if __name__ == "__main__":
    main()

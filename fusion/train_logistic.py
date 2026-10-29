# =========================
# SCIENTIFICALLY CORRECT SETUP (Bayesian Optimization)
# =========================
# Training cohort   : model fitting + training AUC
# Test cohort       : hyperparameter optimization (validation AUC)
# External cohorts  : final testing ONLY
#
# ✔ No data leakage
# ✔ Paper-safe design
# ✔ Bayesian Optimization (skopt)

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

from skopt import Optimizer
from skopt.space import Real, Integer, Categorical

from dataset import load_pcr_dataset

RANDOM_STATE = 712
N_CALLS = 40


# =========================
# Parameter name order
# =========================
def get_param_names(model_name):

    if model_name == "logistic":
        return ["C", "l1_ratio", "class_weight"]

    elif model_name == "rf":
        return [
            "n_estimators", "max_depth",
            "min_samples_leaf", "min_samples_split",
            "max_features", "class_weight"
        ]

    elif model_name == "lgbm":
        return [
            "n_estimators", "learning_rate", "max_depth",
            "num_leaves", "min_child_samples",
            "reg_alpha", "reg_lambda",
            "subsample", "colsample_bytree"
        ]

    elif model_name == "xgb":
        return [
            "n_estimators", "learning_rate", "max_depth",
            "min_child_weight", "gamma",
            "subsample", "colsample_bytree",
            "reg_lambda"
        ]

    else:
        raise ValueError(model_name)


# =========================
# 1. Build pipeline
# =========================
def build_pipeline(model_name, num_features, cat_features):

    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_features),

        ("cat", Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_features)
    ])

    if model_name == "logistic":
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=4000,
            random_state=RANDOM_STATE
        )

    elif model_name == "rf":
        clf = RandomForestClassifier(
            bootstrap=True,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    elif model_name == "lgbm":
        clf = lgb.LGBMClassifier(
            objective="binary",
            random_state=RANDOM_STATE
        )

    elif model_name == "xgb":
        clf = xgb.XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    else:
        raise ValueError(model_name)

    return Pipeline([
        ("preprocess", preprocess),
        ("clf", clf)
    ])


# =========================
# 2. Bayesian search space
# =========================
def get_search_space(model_name):

    if model_name == "logistic":
        return [
            Real(1e-3, 10, prior="log-uniform"),   # C（别到 30，太松）
            Real(0.1, 0.9),                        # l1_ratio
            Categorical([None, "balanced"])        # ✅ 正确
        ]

    elif model_name == "rf":
        return [
            Integer(300, 1200),                     # n_estimators
            Categorical([None, 2, 3, 4, 5, 6]),     # max_depth
            Integer(3, 20),                         # min_samples_leaf
            Integer(5, 40),                         # min_samples_split
            Categorical(["sqrt", "log2", None]),   # ✅ 兼容你环境
            Categorical([None, "balanced"])
        ]



    elif model_name == "lgbm":
        return [
            Integer(300, 1200),                    # n_estimators
            Categorical([0.005, 0.01, 0.02, 0.05]),
            Categorical([2, 3, 4, 5, 6]),
            Categorical([7, 15, 31, 63]),
            Categorical([5, 10, 20, 40]),
            Categorical([0.0, 0.1, 0.5, 1.0]),
            Categorical([0.5, 1, 3, 5, 10]),
            Categorical([0.6, 0.7, 0.8, 0.9]),
            Categorical([0.6, 0.7, 0.8, 0.9])
        ]


    elif model_name == "xgb":
        return [
            Integer(300, 800),
            Categorical([0.01, 0.02, 0.05]),
            Categorical([3, 4, 5]),
            Categorical([5, 8, 12]),
            Categorical([0.0, 0.3, 1.0]),
            Categorical([0.7, 0.8, 0.9]),
            Categorical([0.7, 0.8, 0.9]),
            Categorical([1, 3, 5])
        ]


    else:
        raise ValueError(model_name)


# =========================
# 3. AUC metrics
# =========================
def training_auc(model, X, y, cohort):
    idx = cohort == "Training"
    y_prob = model.predict_proba(X[idx])[:, 1]
    return roc_auc_score(y[idx], y_prob)


def validation_auc(model, X, y, cohort):
    idx = cohort == "Test"
    y_prob = model.predict_proba(X[idx])[:, 1]
    return roc_auc_score(y[idx], y_prob)


# =========================
# 4. Main
# =========================
def main():

    X, y, cohort = load_pcr_dataset()

    num_features = ["Age", "Tumor_size", "Predicted_score"]
    cat_features = [
        "Gender", "Smoking", "Drug", "Pathology",
        "T_stage", "N_stage", "Stage"
    ]

    X_train = X[cohort == "Training"]
    y_train = y[cohort == "Training"]

    external_test_sets = ["Toulouse", "External", "SPH_perspective"]
    model_names = ["lgbm", "logistic", "rf", "xgb"]

    results = []

    for model_name in model_names:
        print(f"\n===== Bayesian Optimization (Validation AUC): {model_name} =====")

        pipe = build_pipeline(model_name, num_features, cat_features)
        space = get_search_space(model_name)
        param_names = get_param_names(model_name)

        optimizer = Optimizer(
            dimensions=space,
            base_estimator="GP",
            random_state=RANDOM_STATE
        )

        best_val_auc = -np.inf
        best_model = None
        best_params = None

        for i in range(N_CALLS):
            params = optimizer.ask()

            param_dict = {}
            for k, v in zip(param_names, params):
                if k == "class_weight" and v == "none":
                    param_dict[f"clf__{k}"] = None
                else:
                    param_dict[f"clf__{k}"] = v

            model = pipe.set_params(**param_dict)
            model.fit(X_train, y_train)

            val_auc = validation_auc(model, X, y, cohort)
            optimizer.tell(params, -val_auc)

            print(f"Iter {i+1}/{N_CALLS} | Validation AUC = {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model = model
                best_params = param_dict

        train_auc_val = training_auc(best_model, X, y, cohort)

        print("Best Training AUC   :", train_auc_val)
        print("Best Validation AUC :", best_val_auc)
        print("Best Params         :", best_params)

        for ts in external_test_sets:
            idx = cohort == ts
            if not np.any(idx):
                continue

            y_prob = best_model.predict_proba(X[idx])[:, 1]
            auc = roc_auc_score(y[idx], y_prob)

            results.append({
                "model": model_name,
                "dataset": ts,
                "training_auc": train_auc_val,
                "validation_auc": best_val_auc,
                "test_auc": auc
            })

    df = pd.DataFrame(results)
    df.to_csv("train_valid_test_auc_bayesian.csv", index=False)

    print("\nSaved to train_valid_test_auc_bayesian.csv")
    print(df)


if __name__ == "__main__":
    main()

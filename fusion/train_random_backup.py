# =========================
# SCIENTIFICALLY CORRECT SETUP (Random Search + SHAP)
# =========================
# Training cohort   : model fitting + training AUC
# Test cohort       : hyperparameter tuning (validation AUC)
# External cohorts  : final testing ONLY
#
# ✔ No data leakage
# ✔ Paper-safe design
# ✔ Outputs SHAP values

import numpy as np
import pandas as pd
import shap  # pip install shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import openpyxl

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

from dataset import load_pcr_dataset
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_ITER = 40  # Random search iterations
TOP_K = 10    # Top-K models for gap selection

# =========================
# 1. Build pipeline
# =========================
def build_pipeline(model_name, num_features, cat_features):

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
        raise ValueError(f"Unknown model_name: {model_name}")

    return Pipeline([
        ("preprocess", preprocess),
        ("clf", clf)
    ])


# =========================
# 2. Parameter distributions
# =========================
def get_param_distributions(model_name):

    if model_name == "logistic":
        return {
            "clf__C": np.logspace(-3, 1.5, 30),
            "clf__l1_ratio": np.linspace(0.1, 0.9, 9),
            "clf__class_weight": [None, "balanced"],
        }

    elif model_name == "rf":
        return {
            "clf__n_estimators": [200, 400, 600, 800, 1200],
            "clf__max_depth": [2, 3, 4, 5, 6, None],
            "clf__min_samples_leaf": [3, 5, 8, 12, 20],
            "clf__min_samples_split": [5, 10, 20, 40],
            "clf__max_features": ["sqrt", "log2"],
            "clf__class_weight": [None, "balanced"],
        }

    elif model_name == "lgbm":
        return {
            "clf__n_estimators": [300, 500, 800, 1200],
            "clf__learning_rate": [0.005, 0.01, 0.02, 0.05],
            "clf__max_depth": [2, 3, 4, 5, 6],
            "clf__num_leaves": [7, 15, 31, 63],
            "clf__min_child_samples": [5, 10, 20, 40],
            "clf__reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "clf__reg_lambda": [0.5, 1, 3, 5, 10],
            "clf__subsample": [0.6, 0.7, 0.8, 0.9],
            "clf__colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        }

    elif model_name == "xgb":
        return {
            "clf__n_estimators": [300, 500, 800],
            "clf__learning_rate": [0.01, 0.02, 0.05],
            "clf__max_depth": [3, 4, 5],
            "clf__min_child_weight": [5, 8, 12],
            "clf__gamma": [0, 0.3, 1.0],
            "clf__subsample": [0.7, 0.8, 0.9],
            "clf__colsample_bytree": [0.7, 0.8, 0.9],
            "clf__reg_lambda": [1, 3, 5],
        }

    else:
        raise ValueError(f"Unknown model_name: {model_name}")


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

    # Load data with patient IDs
    X, y, cohort, patient_ids = load_pcr_dataset(include_patient_id=True)

    num_features = ["Age", "Tumor_size", "Predicted_score"]
    cat_features = [
        "Gender", "Smoking", "Drug", "Pathology",
        "T_stage", "N_stage", "Stage"
    ]

    train_idx = cohort == "Training"
    X_train, y_train = X[train_idx], y[train_idx]
    train_patient_ids = patient_ids[train_idx]

    val_idx = cohort == "Test"
    X_val, y_val = X[val_idx], y[val_idx]
    val_patient_ids = patient_ids[val_idx]

    external_test_sets = ["Toulouse", "External", "SPH_perspective"]
    model_names = ["lgbm", "logistic", "rf", "xgb"]

    results = []
    all_predictions = {}

    for model_name in model_names:
        print(f"\n===== Random Search + SHAP (Validation AUC): {model_name} =====")

        pipe = build_pipeline(model_name, num_features, cat_features)
        param_dist = get_param_distributions(model_name)

        sampler = ParameterSampler(
            param_dist,
            n_iter=N_ITER,
            random_state=RANDOM_STATE
        )

        candidates = []

        for i, params in enumerate(sampler, 1):
            print(f"Trial {i}/{N_ITER}")

            model = pipe.set_params(**params)
            model.fit(X_train, y_train)

            val_auc = validation_auc(model, X, y, cohort)
            train_auc_tmp = training_auc(model, X, y, cohort)
            gap = train_auc_tmp - val_auc

            candidates.append({
                'model': model,
                'params': params,
                'val_auc': val_auc,
                'train_auc': train_auc_tmp,
                'gap': gap
            })

        # ===== Top-K + gap 最小选择 =====
        top_k_models = sorted(candidates, key=lambda x: -x['val_auc'])[:TOP_K]
        best_entry = min(top_k_models, key=lambda x: x['gap'])
        best_model = best_entry['model']
        best_params = best_entry['params']
        best_val_auc = best_entry['val_auc']
        train_auc_val = best_entry['train_auc']
        print(f"Selected model: Top-{TOP_K} with minimum gap ({best_entry['gap']:.4f})")
        print("Best Training AUC:", train_auc_val)
        print("Best Validation AUC:", best_val_auc)
        print("Best Params:", best_params)

        # =========================
        # Predictions
        # =========================
        model_predictions = {}

        # Training set
        train_probs = best_model.predict_proba(X_train)[:, 1]
        train_preds = (train_probs >= 0.5).astype(int)
        model_predictions['train'] = {
            'ids': train_patient_ids,
            'probs': train_probs,
            'preds': train_preds,
            'true': y_train.values if hasattr(y_train, 'values') else y_train
        }

        # Validation set
        val_probs = best_model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        model_predictions['val'] = {
            'ids': val_patient_ids,
            'probs': val_probs,
            'preds': val_preds,
            'true': y_val.values if hasattr(y_val, 'values') else y_val
        }

        # External test sets
        for ts in external_test_sets:
            ts_idx = cohort == ts
            if not np.any(ts_idx):
                continue
            X_ts, y_ts = X[ts_idx], y[ts_idx]
            ts_patient_ids = patient_ids[ts_idx]
            ts_probs = best_model.predict_proba(X_ts)[:, 1]
            ts_preds = (ts_probs >= 0.5).astype(int)
            model_predictions[ts.lower()] = {
                'ids': ts_patient_ids,
                'probs': ts_probs,
                'preds': ts_preds,
                'true': y_ts.values if hasattr(y_ts, 'values') else y_ts
            }

        all_predictions[model_name] = model_predictions

        # =========================
        # SHAP
        # =========================
        print("Calculating SHAP values...")
        X_preprocessed = best_model.named_steps["preprocess"].transform(X_train)

        preprocess = best_model.named_steps["preprocess"]
        feature_names = list(num_features)
        cat_trans = preprocess.named_transformers_["cat"]
        cat_names = cat_trans.named_steps["onehot"].get_feature_names_out(cat_features)
        feature_names.extend(cat_names)

        clf = best_model.named_steps["clf"]

        if model_name in ["rf", "lgbm", "xgb"]:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_preprocessed)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:  # logistic
            explainer = shap.LinearExplainer(clf, X_preprocessed, feature_dependence="independent")
            shap_values = explainer.shap_values(X_preprocessed)

        os.makedirs("shap_plots", exist_ok=True)
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df.to_csv(f"shap_plots/shap_values_{model_name}.csv", index=False)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=True)

        plt.figure(figsize=(10, max(4, 0.25*len(feature_names))))
        plt.barh(importance_df["feature"], importance_df["mean_abs_shap"], color="skyblue")
        plt.xlabel("Mean |SHAP value|")
        plt.title(f"Feature Importance (SHAP) - {model_name}")
        plt.tight_layout()
        plt.savefig(f"shap_plots/shap_feature_importance_{model_name}.pdf", dpi=300)
        plt.close()

        shap.summary_plot(shap_values, features=X_preprocessed, feature_names=feature_names, show=False)
        plt.savefig(f"shap_plots/shap_summary_{model_name}.pdf", dpi=300)
        plt.close()

        # =========================
        # External test AUC
        # =========================
        for ts in external_test_sets:
            ts_idx = cohort == ts
            if not np.any(ts_idx):
                continue
            y_prob = best_model.predict_proba(X[ts_idx])[:, 1]
            auc = roc_auc_score(y[ts_idx], y_prob)
            results.append({
                "model": model_name,
                "dataset": ts,
                "training_auc": train_auc_val,
                "validation_auc": best_val_auc,
                "test_auc": auc
            })

    # =========================
    # Save predictions
    # =========================
    for model_name, model_preds in all_predictions.items():
        for dataset_name, data in model_preds.items():
            df_pred = pd.DataFrame({
                "检查编号": data['ids'],
                "是否预测成功": [True if pred == true else False for pred, true in zip(data['preds'], data['true'])],
                "预测概率": data['probs'],
                "预测结果": data['preds'],
                "ground truth": data['true']
            })
            os.makedirs("prediction_results", exist_ok=True)
            filename = f"prediction_results/{model_name}_{dataset_name}_predictions.xlsx"
            df_pred.to_excel(filename, index=False)

    df = pd.DataFrame(results)
    df.to_csv("train_valid_test_auc_random_search_shap.csv", index=False)
    print("\nSaved to train_valid_test_auc_random_search_shap.csv")
    print(df)


if __name__ == "__main__":
    main()

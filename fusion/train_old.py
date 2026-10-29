import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

from dataset import load_pcr_dataset


# ===== 1. 构建通用 pipeline =====
def build_model(model_name, num_features, cat_features):
    numeric_transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocess = ColumnTransformer(
        [("num", numeric_transformer, num_features), ("cat", categorical_transformer, cat_features)]
    )

    # 模型选择
    if model_name == "logistic":
        clf = LogisticRegression(
            penalty="elasticnet",
            l1_ratio=0.6,  # ↓ 比 0.7 稍微松一点
            C=0.4,  # ↑ 比 0.3 放一点点容量
            max_iter=4000,
            solver="saga",
            class_weight=None,
            random_state=42,
        )

    elif model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=600,  # ↓ 少一点树，防止过平滑
            max_depth=5,  # ↑ 放回一层
            min_samples_leaf=8,  # ↓ 比 10 松一点
            min_samples_split=16,
            max_features=0.7,  # 介于 sqrt 和 0.5 之间
            bootstrap=True,
            class_weight=None,
            random_state=42,
            n_jobs=-1,
        )

    elif model_name == "lgbm":
        clf = lgb.LGBMClassifier(
            n_estimators=700,
            learning_rate=0.03,  # ↑ 比 0.02 稍快
            max_depth=4,  # ↑ 放一层
            num_leaves=12,  # < 2^4 - 1，仍然保守
            min_child_samples=15,  # ↓ 比 20 松一点
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,  # ↓ 稍弱 L1
            reg_lambda=3.0,  # ↓ 稍弱 L2
            class_weight=None,
            random_state=42,
        )

    elif model_name == "xgb":
        clf = xgb.XGBClassifier(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=4,  # ↑ 放一层
            min_child_weight=8,  # ↓ 松一点
            gamma=0.5,  # ↓ 不那么狠
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=3.0,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model = Pipeline([("preprocess", preprocess), ("clf", clf)])

    return model


# ===== 2. ROC & Calibration 可视化函数 =====
def eval_and_plot(model, X, y, name):
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    return auc, y_prob


# ===== 3. 主程序 =====
def main():
    # 3.1 读取数据
    X, y, cohort = load_pcr_dataset()

    num_features = ["Age", "Tumor_size", "Predicted_score"]
    cat_features = ["Gender", "Smoking", "Drug", "Pathology", "T_stage", "N_stage", "Stage"]

    # 3.2 定义所有可能的测试集名称
    test_setnames = ["Toulouse", "External", "SPH_perspective"]

    # 获取所有可用的测试集
    available_test_sets = []
    for test_setname in test_setnames:
        test_idx = cohort == test_setname
        if np.any(test_idx):
            available_test_sets.append(test_setname)
        else:
            print(f"Warning: No data found for test set '{test_setname}'")

    print(f"Found {len(available_test_sets)} available test sets: {available_test_sets}")

    # 3.3 按 cohort 划分
    train_idx = cohort == "Training"
    val_idx = cohort == "Test"

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"Training: {len(X_train)}, Test: {len(X_val)}")

    # 3.4 多模型训练与评估
    model_names = ["logistic", "rf", "lgbm", "xgb"]
    all_results = {}

    for name in model_names:
        print(f"\n=== Training {name} ===")
        model = build_model(name, num_features, cat_features)
        model.fit(X_train, y_train)

        # 为每个可用测试集计算性能
        for test_setname in available_test_sets:
            test_idx = cohort == test_setname
            X_test, y_test = X[test_idx], y[test_idx]

            print(f"Evaluating on {test_setname}: {len(X_test)} samples")

            plt.figure(figsize=(6, 6))
            auc_train, _ = eval_and_plot(model, X_train, y_train, "Training")
            auc_val, _ = eval_and_plot(model, X_val, y_val, "Test")
            auc_test, y_test_prob = eval_and_plot(model, X_test, y_test, test_setname)

            plt.plot([0, 1], [0, 1], "--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{name} ROC by cohort - {test_setname}")
            plt.legend()
            plt.tight_layout()
            os.makedirs(test_setname, exist_ok=True)
            plt.savefig(f"{test_setname}/roc_{name}.png", dpi=300)
            # plt.show()

            # 存储结果
            if name not in all_results:
                all_results[name] = {}
            all_results[name][test_setname] = {
                "train_auc": roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]),
                "val_auc": roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]),
                "test_auc": roc_auc_score(y_test, y_test_prob),
            }

    # 3.5 输出表格
    # 将结果组织成易于阅读的格式
    results_summary = []
    for model_name, model_results in all_results.items():
        for test_setname, set_results in model_results.items():
            row = {
                "model": model_name,
                "dataset": test_setname,
                "train_auc": set_results["train_auc"],
                "val_auc": set_results["val_auc"],
                "test_auc": set_results["test_auc"],
            }
            results_summary.append(row)

    df_results = pd.DataFrame(results_summary)
    df_results.to_csv("model_comparison_results.csv", index=False)
    print("\nModel comparison saved to model_comparison_results.csv")
    print(df_results)

    # 创建透视表便于查看结果
    pivot_results = df_results.pivot(index="model", columns="dataset", values="test_auc")
    print("\nTest AUC by Model and Dataset:")
    print(pivot_results)


if __name__ == "__main__":
    main()

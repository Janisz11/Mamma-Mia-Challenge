import sys
import argparse
import os
import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing_pipeline import get_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Hard-gating Mixture-of-Experts: XGBoost + RandomForest.")
    parser.add_argument("--clinical-xlsx", default="data/gtp_NCCN_based_filled_data_clinical_info.xlsx",
                        help="Path to the NCCN-based clinical info Excel file")
    parser.add_argument("--output-csv", default="hard_gating_results.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_excel(args.clinical_xlsx)
    df = df[df["pcr"].notna()]
    X = df.drop(columns=["pcr"])
    y = df["pcr"].astype(int)

    pipeline = get_pipeline()
    X_transformed = pipeline.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42
    )

    # Expert 1: XGBoost (hyperparameters tuned via Optuna)
    xgb = XGBClassifier(
        n_estimators=124,
        max_depth=3,
        learning_rate=0.0411,
        subsample=0.978,
        colsample_bytree=0.623,
        gamma=0.769,
        reg_alpha=0.749,
        reg_lambda=0.781,
        eval_metric="error",
        tree_method="hist",
        random_state=42,
    )
    xgb.fit(X_train, y_train)

    # Expert 2: RandomForest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Gating labels: 1 = RF was right and XGB was wrong, 0 = use XGB
    xgb_correct = xgb.predict(X_train) == y_train.values
    rf_correct = rf.predict(X_train) == y_train.values
    gating_labels = (rf_correct & ~xgb_correct).astype(int)

    # Gating model (best Optuna Trial 11)
    gating_model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_error",
        boosting_type="gbdt",
        verbosity=-1,
        n_estimators=200,
        learning_rate=0.08225826152779128,
        max_depth=2,
        num_leaves=62,
        subsample=0.8529811277498729,
        colsample_bytree=0.9221309236747597,
        reg_alpha=0.09452586401207042,
        reg_lambda=1.432580643025585,
    )
    gating_model.fit(X_train, gating_labels)

    gating_choices = gating_model.predict(X_test)
    final_preds = np.where(gating_choices == 0, xgb.predict(X_test), rf.predict(X_test))

    num_xgb = np.sum(gating_choices == 0)
    num_rf = np.sum(gating_choices == 1)
    print(f"Gating → XGBoost: {num_xgb}  RandomForest: {num_rf}")

    accuracy = accuracy_score(y_test, final_preds)
    report = classification_report(y_test, final_preds, output_dict=True)

    row = {"accuracy": accuracy}
    fieldnames = ["accuracy"]
    for label in report:
        if isinstance(report[label], dict):
            for metric in report[label]:
                key = f"{label}_{metric}"
                row[key] = report[label][metric]
                fieldnames.append(key)

    file_exists = os.path.isfile(args.output_csv)
    with open(args.output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Hard Gating MoE accuracy: {accuracy:.4f}")
    print(f"Results saved → {args.output_csv}")


if __name__ == "__main__":
    main()

import argparse
import time
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import (
    load_labels,
    mpnet_feature_names,
    save_classification_artifacts,
    save_json,
    save_predictions,
    save_top_feature_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an XGBoost classifier on MPNet embeddings.")
    parser.add_argument("--feature-dir", default="features")
    parser.add_argument("--output-dir", default="experiments/xgboost/outputs")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--top-k-features", type=int, default=25)
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Skip saving the trained model file.",
    )
    return parser.parse_args()


def format_xgboost_feature_name(raw_feature: str, feature_names: np.ndarray) -> str:
    if raw_feature.startswith("f") and raw_feature[1:].isdigit():
        feature_idx = int(raw_feature[1:])
        if 0 <= feature_idx < len(feature_names):
            return str(feature_names[feature_idx])
    return raw_feature


def save_top_features(
    model: XGBClassifier,
    labels: list[str],
    feature_names: np.ndarray,
    top_k: int,
    output_path: Path,
) -> None:
    booster = model.get_booster()
    tree_df = booster.trees_to_dataframe()
    split_rows = tree_df[tree_df["Feature"] != "Leaf"].copy()
    rows: list[dict] = []

    if not split_rows.empty:
        split_rows["class_idx"] = split_rows["Tree"] % len(labels)
        gains = (
            split_rows.groupby(["class_idx", "Feature"], as_index=False)["Gain"]
            .sum()
            .sort_values(["class_idx", "Gain"], ascending=[True, False])
        )
        for class_idx, label in enumerate(labels):
            class_rows = gains[gains["class_idx"] == class_idx].head(top_k)
            for rank, (_, row) in enumerate(class_rows.iterrows(), start=1):
                rows.append(
                    {
                        "class": label,
                        "direction": "positive",
                        "rank": rank,
                        "feature": format_xgboost_feature_name(str(row["Feature"]), feature_names),
                        "coefficient": float(row["Gain"]),
                    }
                )

    if not rows:
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-top_k:][::-1]
        for rank, feature_idx in enumerate(top_indices, start=1):
            rows.append(
                {
                    "class": "__global__",
                    "direction": "positive",
                    "rank": rank,
                    "feature": str(feature_names[feature_idx]),
                    "coefficient": float(importances[feature_idx]),
                }
            )

    save_top_feature_rows(rows, output_path)


def main():
    args = parse_args()
    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(feature_dir)
    x_train = np.load(feature_dir / "mpnet_train_embeddings.npy")
    x_val = np.load(feature_dir / "mpnet_val_embeddings.npy")
    y_train = np.load(feature_dir / "y_train.npy")
    y_val = np.load(feature_dir / "y_val.npy")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(labels),
        eval_metric="mlogloss",
        tree_method="hist",
        device=args.device,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    start_time = time.time()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    train_seconds = time.time() - start_time

    probabilities = model.predict_proba(x_val)
    y_pred = probabilities.argmax(axis=1)

    accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    weighted_f1 = f1_score(y_val, y_pred, average="weighted")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_val,
        y_pred,
        labels=np.arange(len(labels)),
        zero_division=0,
    )

    metrics = {
        "model": "XGBoost",
        "feature_input": "MPNet embedding",
        "feature_shape_train": list(x_train.shape),
        "feature_shape_val": list(x_val.shape),
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
        "early_stopping_rounds": args.early_stopping_rounds,
        "device": args.device,
        "train_seconds": train_seconds,
        "best_iteration": int(getattr(model, "best_iteration", model.n_estimators)),
        "best_score": float(getattr(model, "best_score", 0.0)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {
            label: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, label in enumerate(labels)
        },
    }
    save_json(metrics, output_dir / "metrics.json")
    save_classification_artifacts(output_dir, y_val, y_pred, labels)
    save_predictions(
        feature_dir,
        y_val,
        y_pred,
        labels,
        output_dir / "val_predictions.csv",
        probabilities=probabilities,
    )
    save_top_features(
        model,
        labels,
        mpnet_feature_names(x_train.shape[1]),
        args.top_k_features,
        output_dir / "top_features_by_class.csv",
    )

    if not args.no_save_model:
        joblib.dump(model, output_dir / "xgboost_model.joblib")

    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation macro_f1: {macro_f1:.4f}")
    print(f"train_seconds: {train_seconds:.2f}")


if __name__ == "__main__":
    main()

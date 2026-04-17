import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


DEFAULT_LABELS = [
    "True_Correct",
    "False_Misconception",
    "False_Neither",
    "True_Neither",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Logistic Regression baseline.")
    parser.add_argument("--feature-dir", default="features")
    parser.add_argument("--output-dir", default="experiments/logistic_regression/outputs")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--solver", default="saga", choices=["saga", "liblinear", "lbfgs"])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-k-features", type=int, default=25)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Skip saving the trained model file.",
    )
    return parser.parse_args()


def load_metadata(feature_dir):
    metadata_path = feature_dir / "feature_metadata.json"
    if not metadata_path.exists():
        return DEFAULT_LABELS

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    return metadata.get("labels", DEFAULT_LABELS)


def load_data(feature_dir):
    x_train = sparse.load_npz(feature_dir / "X_train.npz")
    x_val = sparse.load_npz(feature_dir / "X_val.npz")
    y_train = np.load(feature_dir / "y_train.npy")
    y_val = np.load(feature_dir / "y_val.npy")
    return x_train, x_val, y_train, y_val


def plot_confusion_matrix(y_true, y_pred, labels, output_path, normalize=None):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)), normalize=normalize)
    fig, ax = plt.subplots(figsize=(8, 7))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    value_format = ".2f" if normalize else "d"
    display.plot(ax=ax, cmap="Blues", values_format=value_format, colorbar=True)
    ax.set_title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_per_class_metrics(report_df, labels, output_path):
    class_rows = report_df.loc[labels, ["precision", "recall", "f1-score"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    class_rows.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1")
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_predictions(feature_dir, y_true, y_pred, proba, labels, output_path):
    val_split_path = feature_dir / "val_split.csv"
    if val_split_path.exists():
        predictions = pd.read_csv(val_split_path)
    else:
        predictions = pd.DataFrame({"sample_index": np.arange(len(y_true))})

    predictions["true_label_id"] = y_true
    predictions["pred_label_id"] = y_pred
    predictions["true_label"] = [labels[idx] for idx in y_true]
    predictions["pred_label"] = [labels[idx] for idx in y_pred]
    predictions["is_correct"] = y_true == y_pred

    if proba is not None:
        for idx, label in enumerate(labels):
            predictions[f"prob_{label}"] = proba[:, idx]

    predictions.to_csv(output_path, index=False)


def get_feature_names(feature_dir, n_features):
    word_path = feature_dir / "tfidf_word_vectorizer.joblib"
    char_path = feature_dir / "tfidf_char_vectorizer.joblib"
    if not word_path.exists() or not char_path.exists():
        return None

    word_vectorizer = joblib.load(word_path)
    char_vectorizer = joblib.load(char_path)
    word_features = [f"word::{name}" for name in word_vectorizer.get_feature_names_out()]
    char_features = [f"char::{name}" for name in char_vectorizer.get_feature_names_out()]
    tfidf_features = word_features + char_features
    embedding_dim = n_features - len(tfidf_features)
    embedding_features = [f"mpnet_dim_{idx}" for idx in range(embedding_dim)]
    return np.array(tfidf_features + embedding_features, dtype=object)


def save_top_features(model, feature_names, labels, top_k, output_path):
    if feature_names is None or not hasattr(model, "coef_"):
        return

    rows = []
    coef = model.coef_
    for class_idx, label in enumerate(labels):
        class_coef = coef[class_idx]
        top_positive = np.argsort(class_coef)[-top_k:][::-1]
        top_negative = np.argsort(class_coef)[:top_k]

        for rank, feature_idx in enumerate(top_positive, start=1):
            rows.append(
                {
                    "class": label,
                    "direction": "positive",
                    "rank": rank,
                    "feature": feature_names[feature_idx],
                    "coefficient": float(class_coef[feature_idx]),
                }
            )
        for rank, feature_idx in enumerate(top_negative, start=1):
            rows.append(
                {
                    "class": label,
                    "direction": "negative",
                    "rank": rank,
                    "feature": feature_names[feature_idx],
                    "coefficient": float(class_coef[feature_idx]),
                }
            )

    pd.DataFrame(rows).to_csv(output_path, index=False)


def main():
    args = parse_args()
    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_metadata(feature_dir)
    x_train, x_val, y_train, y_val = load_data(feature_dir)

    print(f"X_train shape: {x_train.shape}")
    print(f"X_val shape: {x_val.shape}")
    print(f"Classes: {labels}")

    model = LogisticRegression(
        C=args.C,
        solver=args.solver,
        class_weight="balanced",
        max_iter=args.max_iter,
        random_state=args.random_state,
        verbose=args.verbose,
    )

    start_time = time.time()
    model.fit(x_train, y_train)
    train_seconds = time.time() - start_time

    y_pred = model.predict(x_val)
    proba = model.predict_proba(x_val) if hasattr(model, "predict_proba") else None

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
        "model": "LogisticRegression",
        "feature_input": "TF-IDF + MPNet embedding",
        "feature_shape_train": list(x_train.shape),
        "feature_shape_val": list(x_val.shape),
        "C": args.C,
        "solver": args.solver,
        "max_iter": args.max_iter,
        "class_weight": "balanced",
        "train_seconds": train_seconds,
        "n_iter": model.n_iter_.tolist(),
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

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    report_text = classification_report(y_val, y_pred, target_names=labels, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    report_dict = classification_report(
        y_val,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(output_dir / "classification_report.csv")

    cm = confusion_matrix(y_val, y_pred, labels=np.arange(len(labels)))
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / "confusion_matrix.csv")
    plot_confusion_matrix(y_val, y_pred, labels, output_dir / "confusion_matrix.png")
    plot_confusion_matrix(
        y_val,
        y_pred,
        labels,
        output_dir / "confusion_matrix_normalized.png",
        normalize="true",
    )
    plot_per_class_metrics(report_df, labels, output_dir / "per_class_metrics.png")

    save_predictions(
        feature_dir,
        y_val,
        y_pred,
        proba,
        labels,
        output_dir / "val_predictions.csv",
    )

    feature_names = get_feature_names(feature_dir, x_train.shape[1])
    save_top_features(
        model,
        feature_names,
        labels,
        args.top_k_features,
        output_dir / "top_features_by_class.csv",
    )

    if not args.no_save_model:
        joblib.dump(model, output_dir / "logreg_model.joblib")

    print(report_text)
    print(f"accuracy: {accuracy:.4f}")
    print(f"macro_f1: {macro_f1:.4f}")
    print(f"weighted_f1: {weighted_f1:.4f}")
    print(f"train_seconds: {train_seconds:.2f}")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()

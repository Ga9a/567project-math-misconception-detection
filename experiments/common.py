import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix


DEFAULT_LABELS = [
    "True_Correct",
    "False_Misconception",
    "False_Neither",
    "True_Neither",
]

TOP_FEATURE_COLUMNS = ["class", "direction", "rank", "feature", "coefficient"]


def load_labels(feature_dir: Path) -> list[str]:
    metadata_path = feature_dir / "feature_metadata.json"
    if not metadata_path.exists():
        return DEFAULT_LABELS

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    return metadata.get("labels", DEFAULT_LABELS)


def save_json(data: dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def mpnet_feature_names(n_features: int) -> np.ndarray:
    return np.array([f"mpnet_dim_{idx}" for idx in range(n_features)], dtype=object)


def save_top_feature_rows(rows: list[dict], output_path: Path) -> None:
    pd.DataFrame(rows, columns=TOP_FEATURE_COLUMNS).to_csv(output_path, index=False)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    output_path: Path,
    normalize: str | None = None,
) -> None:
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


def plot_per_class_metrics(report_df: pd.DataFrame, labels: list[str], output_path: Path) -> None:
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


def save_predictions(
    feature_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    output_path: Path,
    probabilities: np.ndarray | None = None,
) -> None:
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

    if probabilities is not None:
        for idx, label in enumerate(labels):
            predictions[f"prob_{label}"] = probabilities[:, idx]

    predictions.to_csv(output_path, index=False)


def save_classification_artifacts(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    report_text = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(output_dir / "classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / "confusion_matrix.csv")
    plot_confusion_matrix(y_true, y_pred, labels, output_dir / "confusion_matrix.png")
    plot_confusion_matrix(
        y_true,
        y_pred,
        labels,
        output_dir / "confusion_matrix_normalized.png",
        normalize="true",
    )
    plot_per_class_metrics(report_df, labels, output_dir / "per_class_metrics.png")
    return report_df

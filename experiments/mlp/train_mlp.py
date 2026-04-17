import argparse
import time
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple MLP on MPNet embeddings.")
    parser.add_argument("--feature-dir", default="features")
    parser.add_argument("--output-dir", default="experiments/mlp/outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k-features", type=int, default=25)
    parser.add_argument("--top-feature-samples-per-class", type=int, default=128)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Skip saving the trained model file.",
    )
    return parser.parse_args()


def build_dataloader(features, labels, batch_size, shuffle, num_workers):
    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(labels).long(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(model, dataloader, device):
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    probabilities = torch.softmax(logits, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    avg_loss = total_loss / len(labels)
    macro_f1 = f1_score(labels, predictions, average="macro")
    return avg_loss, labels, predictions, probabilities, macro_f1


def save_top_features(
    model: nn.Module,
    x_reference: np.ndarray,
    y_reference: np.ndarray,
    labels: list[str],
    device: torch.device,
    top_k: int,
    max_samples_per_class: int,
    output_path: Path,
) -> None:
    feature_names = mpnet_feature_names(x_reference.shape[1])
    rows: list[dict] = []
    model.eval()

    for class_idx, label in enumerate(labels):
        sample_indices = np.flatnonzero(y_reference == class_idx)[:max_samples_per_class]
        if len(sample_indices) == 0:
            continue

        class_inputs = torch.from_numpy(x_reference[sample_indices]).float().to(device)
        class_inputs.requires_grad_(True)
        logits = model(class_inputs)
        target = logits[:, class_idx].sum()
        gradients = torch.autograd.grad(target, class_inputs)[0]
        attribution = (gradients * class_inputs).mean(dim=0).detach().cpu().numpy()

        top_positive = np.argsort(attribution)[-top_k:][::-1]
        top_negative = np.argsort(attribution)[:top_k]

        for rank, feature_idx in enumerate(top_positive, start=1):
            rows.append(
                {
                    "class": label,
                    "direction": "positive",
                    "rank": rank,
                    "feature": str(feature_names[feature_idx]),
                    "coefficient": float(attribution[feature_idx]),
                }
            )
        for rank, feature_idx in enumerate(top_negative, start=1):
            rows.append(
                {
                    "class": label,
                    "direction": "negative",
                    "rank": rank,
                    "feature": str(feature_names[feature_idx]),
                    "coefficient": float(attribution[feature_idx]),
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

    device = torch.device(args.device)
    train_loader = build_dataloader(x_train, y_train, args.batch_size, True, args.num_workers)
    val_loader = build_dataloader(x_val, y_val, args.batch_size, False, args.num_workers)

    model = SimpleMLP(
        input_dim=x_train.shape[1],
        num_classes=len(labels),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    class_counts = np.bincount(y_train, minlength=len(labels))
    class_weights = torch.tensor(
        len(y_train) / (len(labels) * np.maximum(class_counts, 1)),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size

        train_loss = running_loss / seen
        val_loss, _, _, _, val_macro_f1 = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    train_seconds = time.time() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)

    _, y_true, y_pred, probabilities, _ = evaluate(model, val_loader, device)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(labels)),
        zero_division=0,
    )

    metrics = {
        "model": "SimpleMLP",
        "feature_input": "MPNet embedding",
        "feature_shape_train": list(x_train.shape),
        "feature_shape_val": list(x_val.shape),
        "device": str(device),
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_seconds": train_seconds,
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
    save_classification_artifacts(output_dir, y_true, y_pred, labels)
    save_predictions(
        feature_dir,
        y_true,
        y_pred,
        labels,
        output_dir / "val_predictions.csv",
        probabilities=probabilities,
    )
    save_top_features(
        model,
        x_val,
        y_true,
        labels,
        device,
        args.top_k_features,
        args.top_feature_samples_per_class,
        output_dir / "top_features_by_class.csv",
    )

    if not args.no_save_model:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "labels": labels,
                "input_dim": int(x_train.shape[1]),
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
            },
            output_dir / "mlp_model.pt",
        )

    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation macro_f1: {macro_f1:.4f}")
    print(f"train_seconds: {train_seconds:.2f}")


if __name__ == "__main__":
    main()

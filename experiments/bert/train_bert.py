import argparse
from collections import defaultdict
import shutil
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import (
    load_labels,
    save_classification_artifacts,
    save_json,
    save_predictions,
    save_top_feature_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a BERT model on raw text.")
    parser.add_argument("--feature-dir", default="features")
    parser.add_argument("--output-dir", default="experiments/bert/outputs")
    parser.add_argument("--model-name", default="google-bert/bert-base-uncased")
    parser.add_argument("--hf-cache-dir", default="hf_cache")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--top-k-features", type=int, default=25)
    parser.add_argument("--top-feature-samples-per-class", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Skip saving the fine-tuned model directory.",
    )
    return parser.parse_args()


def tokenize_dataframe(tokenizer, texts, max_length):
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
    }


def trainer_artifact_dir(output_dir: Path) -> Path:
    return output_dir.parent / f"{output_dir.name}_hf_trainer"


def save_top_token_features(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    device: str,
    max_length: int,
    top_k: int,
    max_samples_per_class: int,
    output_path: Path,
) -> None:
    rows: list[dict] = []
    special_ids = set(tokenizer.all_special_ids)
    model.eval()

    for class_idx, label in enumerate(labels):
        sample_indices = np.flatnonzero((y_true == class_idx) & (y_pred == class_idx))
        if len(sample_indices) == 0:
            sample_indices = np.flatnonzero(y_true == class_idx)
        sample_indices = sample_indices[:max_samples_per_class]
        if len(sample_indices) == 0:
            continue

        encoded = tokenizer(
            texts.iloc[sample_indices].fillna("").astype(str).tolist(),
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        inputs_embeds = model.get_input_embeddings()(input_ids).detach()
        inputs_embeds.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        target = outputs.logits[:, class_idx].sum()
        gradients = torch.autograd.grad(target, inputs_embeds)[0]
        token_scores = (gradients * inputs_embeds).sum(dim=-1).detach().cpu().numpy()
        input_ids_np = input_ids.detach().cpu().numpy()
        attention_mask_np = attention_mask.detach().cpu().numpy()

        token_sums: dict[str, float] = defaultdict(float)
        token_counts: dict[str, int] = defaultdict(int)
        for row_idx in range(token_scores.shape[0]):
            for token_idx in range(token_scores.shape[1]):
                if attention_mask_np[row_idx, token_idx] == 0:
                    continue
                token_id = int(input_ids_np[row_idx, token_idx])
                if token_id in special_ids:
                    continue
                token = tokenizer.convert_ids_to_tokens(token_id)
                token_sums[token] += float(token_scores[row_idx, token_idx])
                token_counts[token] += 1

        if not token_sums:
            continue

        token_means = {
            token: token_sums[token] / token_counts[token]
            for token in token_sums
        }
        ranked = sorted(token_means.items(), key=lambda item: item[1])
        top_negative = ranked[:top_k]
        top_positive = ranked[-top_k:][::-1]

        for rank, (token, score) in enumerate(top_positive, start=1):
            rows.append(
                {
                    "class": label,
                    "direction": "positive",
                    "rank": rank,
                    "feature": token,
                    "coefficient": float(score),
                }
            )
        for rank, (token, score) in enumerate(top_negative, start=1):
            rows.append(
                {
                    "class": label,
                    "direction": "negative",
                    "rank": rank,
                    "feature": token,
                    "coefficient": float(score),
                }
            )

    save_top_feature_rows(rows, output_path)


def main():
    args = parse_args()
    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    trainer_output_dir = trainer_artifact_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if trainer_output_dir.exists():
        shutil.rmtree(trainer_output_dir)

    labels = load_labels(feature_dir)
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for idx, label in enumerate(labels)}

    train_df = pd.read_csv(feature_dir / "train_split.csv")
    val_df = pd.read_csv(feature_dir / "val_split.csv")
    train_df["label"] = train_df["Category_4"].map(label_to_id)
    val_df["label"] = val_df["Category_4"].map(label_to_id)

    model_path = Path(args.model_name)
    if model_path.exists():
        snapshot_path = str(model_path)
        print(f"Using local model snapshot: {snapshot_path}")
    else:
        snapshot_path = snapshot_download(
            repo_id=args.model_name,
            cache_dir=args.hf_cache_dir,
            resume_download=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        snapshot_path,
        local_files_only=True,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    model.to(args.device)
    use_bf16 = args.device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = args.device == "cuda" and not use_bf16

    train_tokens = tokenize_dataframe(
        tokenizer,
        train_df["text"].fillna("").astype(str).tolist(),
        args.max_length,
    )
    val_tokens = tokenize_dataframe(
        tokenizer,
        val_df["text"].fillna("").astype(str).tolist(),
        args.max_length,
    )

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, tokenized, labels_array):
            self.tokenized = tokenized
            self.labels_array = labels_array

        def __len__(self):
            return len(self.labels_array)

        def __getitem__(self, idx):
            item = {key: torch.tensor(value[idx]) for key, value in self.tokenized.items()}
            item["labels"] = torch.tensor(self.labels_array[idx], dtype=torch.long)
            return item

    train_dataset = TextDataset(train_tokens, train_df["label"].to_numpy(dtype=np.int64))
    val_dataset = TextDataset(val_tokens, val_df["label"].to_numpy(dtype=np.int64))

    training_args = TrainingArguments(
        output_dir=str(trainer_output_dir),
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="no" if args.no_save_model else "epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        report_to=[],
        dataloader_num_workers=0,
        seed=args.seed,
        bf16=use_bf16,
        fp16=use_fp16,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        load_best_model_at_end=not args.no_save_model,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    trainer.train()
    train_seconds = time.time() - start_time

    prediction_output = trainer.predict(val_dataset)
    logits = prediction_output.predictions
    y_true = prediction_output.label_ids
    y_pred = logits.argmax(axis=1)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

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
        "model": "BERT",
        "model_name": args.model_name,
        "model_snapshot": snapshot_path,
        "feature_input": "raw text",
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "device": args.device,
        "bf16": use_bf16,
        "fp16": use_fp16,
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
    save_top_token_features(
        model,
        tokenizer,
        val_df["text"],
        y_true,
        y_pred,
        labels,
        args.device,
        args.max_length,
        args.top_k_features,
        args.top_feature_samples_per_class,
        output_dir / "top_features_by_class.csv",
    )

    if not args.no_save_model:
        trainer.save_model(str(output_dir / "model"))
        tokenizer.save_pretrained(str(output_dir / "model"))
    if trainer_output_dir.exists():
        shutil.rmtree(trainer_output_dir)

    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation macro_f1: {macro_f1:.4f}")
    print(f"train_seconds: {train_seconds:.2f}")


if __name__ == "__main__":
    main()

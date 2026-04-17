import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


LABELS = [
    "True_Correct",
    "False_Misconception",
    "False_Neither",
    "True_Neither",
]


def encode_labels(labels):
    label_to_id = {label: idx for idx, label in enumerate(LABELS)}
    unknown = sorted(set(labels) - set(label_to_id))
    if unknown:
        raise ValueError(f"Unknown labels found: {unknown}")
    return np.array([label_to_id[label] for label in labels], dtype=np.int64), label_to_id


def encode_texts(model, texts, batch_size):
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def download_model_snapshot(model_name, cache_dir):
    model_path = Path(model_name)
    if model_path.exists():
        print(f"Using local model snapshot: {model_path}")
        return str(model_path)

    print(f"Downloading {model_name} to cache: {cache_dir}")
    return snapshot_download(
        repo_id=model_name,
        cache_dir=str(cache_dir),
        resume_download=True,
    )


def save_split_csv(df, output_path):
    columns = ["row_id", "QuestionId", "text"]
    for optional_column in ("Category", "Category_4"):
        if optional_column in df.columns:
            columns.append(optional_column)
    df.loc[:, columns].to_csv(output_path, index=False)


def build_tfidf_features(train_texts, val_texts, test_texts, args):
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        min_df=args.word_min_df,
        max_features=args.word_max_features,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=args.char_min_df,
        max_features=args.char_max_features,
        sublinear_tf=True,
    )

    train_word = word_vectorizer.fit_transform(train_texts)
    val_word = word_vectorizer.transform(val_texts)
    test_word = word_vectorizer.transform(test_texts)

    train_char = char_vectorizer.fit_transform(train_texts)
    val_char = char_vectorizer.transform(val_texts)
    test_char = char_vectorizer.transform(test_texts)

    train_tfidf = sparse.hstack([train_word, train_char], format="csr")
    val_tfidf = sparse.hstack([val_word, val_char], format="csr")
    test_tfidf = sparse.hstack([test_word, test_char], format="csr")

    return train_tfidf, val_tfidf, test_tfidf, word_vectorizer, char_vectorizer


def combine_features(tfidf_matrix, embeddings):
    return sparse.hstack(
        [tfidf_matrix, sparse.csr_matrix(embeddings)],
        format="csr",
        dtype=np.float32,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build train/validation/test feature vectors for traditional ML."
    )
    parser.add_argument("--processed-train", default="data/processed_train.csv")
    parser.add_argument("--processed-test", default="data/processed_test.csv")
    parser.add_argument("--output-dir", default="features")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model used for dense embeddings.",
    )
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--word-max-features", type=int, default=50000)
    parser.add_argument("--char-max-features", type=int, default=50000)
    parser.add_argument("--word-min-df", type=int, default=2)
    parser.add_argument("--char-min-df", type=int, default=2)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
    )
    parser.add_argument(
        "--hf-cache-dir",
        default="hf_cache",
        help="Directory used to cache Hugging Face models before loading them locally.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.processed_train)
    test_df = pd.read_csv(args.processed_test)

    required_train_columns = {"row_id", "QuestionId", "text", "Category_4"}
    required_test_columns = {"row_id", "QuestionId", "text"}
    missing_train = required_train_columns - set(train_df.columns)
    missing_test = required_test_columns - set(test_df.columns)
    if missing_train:
        raise ValueError(f"Missing train columns: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"Missing test columns: {sorted(missing_test)}")

    train_split, val_split = train_test_split(
        train_df,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=train_df["Category_4"],
    )
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    save_split_csv(train_split, output_dir / "train_split.csv")
    save_split_csv(val_split, output_dir / "val_split.csv")
    save_split_csv(test_df, output_dir / "test_processed.csv")

    y_train, label_to_id = encode_labels(train_split["Category_4"].tolist())
    y_val, _ = encode_labels(val_split["Category_4"].tolist())
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_val.npy", y_val)

    train_texts = train_split["text"].fillna("").astype(str).tolist()
    val_texts = val_split["text"].fillna("").astype(str).tolist()
    test_texts = test_df["text"].fillna("").astype(str).tolist()

    print("Building TF-IDF features...")
    train_tfidf, val_tfidf, test_tfidf, word_vectorizer, char_vectorizer = build_tfidf_features(
        train_texts,
        val_texts,
        test_texts,
        args,
    )
    sparse.save_npz(output_dir / "tfidf_train.npz", train_tfidf)
    sparse.save_npz(output_dir / "tfidf_val.npz", val_tfidf)
    sparse.save_npz(output_dir / "tfidf_test.npz", test_tfidf)
    joblib.dump(word_vectorizer, output_dir / "tfidf_word_vectorizer.joblib")
    joblib.dump(char_vectorizer, output_dir / "tfidf_char_vectorizer.joblib")

    print(f"Encoding dense embeddings on {args.device} with {args.model_name}...")
    model_snapshot = download_model_snapshot(args.model_name, Path(args.hf_cache_dir))
    model = SentenceTransformer(model_snapshot, device=args.device, local_files_only=True)
    train_embeddings = encode_texts(model, train_texts, args.batch_size)
    val_embeddings = encode_texts(model, val_texts, args.batch_size)
    test_embeddings = encode_texts(model, test_texts, args.batch_size)
    np.save(output_dir / "mpnet_train_embeddings.npy", train_embeddings)
    np.save(output_dir / "mpnet_val_embeddings.npy", val_embeddings)
    np.save(output_dir / "mpnet_test_embeddings.npy", test_embeddings)

    print("Combining TF-IDF and dense embeddings...")
    x_train = combine_features(train_tfidf, train_embeddings)
    x_val = combine_features(val_tfidf, val_embeddings)
    x_test = combine_features(test_tfidf, test_embeddings)
    sparse.save_npz(output_dir / "X_train.npz", x_train)
    sparse.save_npz(output_dir / "X_val.npz", x_val)
    sparse.save_npz(output_dir / "X_test.npz", x_test)

    metadata = {
        "model_name": args.model_name,
        "model_snapshot": model_snapshot,
        "embedding_dim": int(train_embeddings.shape[1]),
        "labels": LABELS,
        "label_to_id": label_to_id,
        "id_to_label": {idx: label for label, idx in label_to_id.items()},
        "val_size": args.val_size,
        "random_state": args.random_state,
        "word_tfidf_shape": list(train_tfidf.shape), # type: ignore
        "combined_feature_shape": {
            "train": list(x_train.shape),# type: ignore
            "val": list(x_val.shape),# type: ignore
            "test": list(x_test.shape),# type: ignore
        },
        "train_label_counts": train_split["Category_4"].value_counts().to_dict(),
        "val_label_counts": val_split["Category_4"].value_counts().to_dict(),
    }
    with (output_dir / "feature_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Wrote feature files to {output_dir}")
    print(f"X_train shape: {x_train.shape}")
    print(f"X_val shape: {x_val.shape}")
    print(f"X_test shape: {x_test.shape}")


if __name__ == "__main__":
    main()

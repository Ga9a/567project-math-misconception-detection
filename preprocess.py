import argparse
import csv
from pathlib import Path


CATEGORY_MAP = {
    "True_Correct": "True_Correct",
    "False_Correct": "True_Correct",
    "False_Misconception": "False_Misconception",
    "True_Misconception": "False_Misconception",
    "False_Neither": "False_Neither",
    "True_Neither": "True_Neither",
}


TEXT_COLUMNS = ("QuestionText", "MC_Answer", "StudentExplanation")


def build_combined_text(row):
    return (
        f"Question: {row.get('QuestionText', '').strip()}\n"
        f"Answer: {row.get('MC_Answer', '').strip()}\n"
        f"Student explanation: {row.get('StudentExplanation', '').strip()}"
    )


def process_file(input_path, output_path, has_labels):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        fieldnames = reader.fieldnames or []
        missing_columns = [col for col in TEXT_COLUMNS if col not in fieldnames]
        if missing_columns:
            raise ValueError(f"{input_path} is missing required columns: {missing_columns}")
        if has_labels and "Category" not in fieldnames:
            raise ValueError(f"{input_path} is missing required column: Category")

        output_fields = ["row_id", "QuestionId", "text"]
        if has_labels:
            output_fields.extend(["Category", "Category_4"])

        with output_path.open("w", encoding="utf-8", newline="") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=output_fields)
            writer.writeheader()

            for row in reader:
                processed = {
                    "row_id": row.get("row_id", ""),
                    "QuestionId": row.get("QuestionId", ""),
                    "text": build_combined_text(row),
                }

                if has_labels:
                    original_category = row["Category"].strip()
                    if original_category not in CATEGORY_MAP:
                        raise ValueError(f"Unexpected category {original_category!r} in {input_path}")
                    processed["Category"] = original_category
                    processed["Category_4"] = CATEGORY_MAP[original_category]

                writer.writerow(processed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create processed train/test CSV files for 4-class classification."
    )
    parser.add_argument("--train", default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--test", default="data/test.csv", help="Path to test.csv")
    parser.add_argument(
        "--processed-train",
        default="data/processed_train.csv",
        help="Output path for processed training data",
    )
    parser.add_argument(
        "--processed-test",
        default="data/processed_test.csv",
        help="Output path for processed test data",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    process_file(args.train, args.processed_train, has_labels=True)
    process_file(args.test, args.processed_test, has_labels=False)
    print(f"Wrote {args.processed_train}")
    print(f"Wrote {args.processed_test}")


if __name__ == "__main__":
    main()

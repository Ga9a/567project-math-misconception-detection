import csv
from pathlib import Path
import tempfile
import unittest
from preprocess import CATEGORY_MAP, process_file

class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "input.csv"
        self.output = Path(self.directory.name) / "output.csv"
    def write(self, categories):
        with self.source.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=["row_id", "QuestionId", "QuestionText", "MC_Answer", "StudentExplanation", "Category"])
            writer.writeheader()
            for i, category in enumerate(categories):
                writer.writerow({"row_id": i, "QuestionId": 1, "QuestionText": "  Question  ", "MC_Answer": "A", "StudentExplanation": "Because", "Category": category})
    def test_all_labels_and_text_are_preserved(self):
        self.write(CATEGORY_MAP)
        process_file(self.source, self.output, True)
        with self.output.open(newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual([row["Category_4"] for row in rows], list(CATEGORY_MAP.values()))
        self.assertEqual(rows[0]["text"], "Question: Question\nAnswer: A\nStudent explanation: Because")
    def test_invalid_label_does_not_replace_existing_output(self):
        self.write(["True_Correct", "invalid"])
        self.output.write_text("previous output")
        with self.assertRaises(ValueError):
            process_file(self.source, self.output, True)
        self.assertEqual(self.output.read_text(), "previous output")
        self.assertFalse(list(self.output.parent.glob("*.tmp")))
    def test_cannot_overwrite_input(self):
        self.write(["True_Correct"])
        original = self.source.read_bytes()
        with self.assertRaises(ValueError):
            process_file(self.source, self.source, True)
        self.assertEqual(self.source.read_bytes(), original)
    def test_required_columns(self):
        self.source.write_text("Category\nTrue_Correct\n")
        with self.assertRaisesRegex(ValueError, "required columns"):
            process_file(self.source, self.output, True)
    def test_unlabelled_test_data(self):
        self.write([""])
        process_file(self.source, self.output, False)
        with self.output.open(newline="") as file:
            reader = csv.DictReader(file)
            self.assertNotIn("Category_4", reader.fieldnames)
            self.assertEqual(next(reader)["row_id"], "0")

if __name__ == "__main__":
    unittest.main()

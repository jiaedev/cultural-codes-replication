"""Explicit CSV labels and split boundaries; immutable source data."""
import csv
from collections import Counter
from pathlib import Path

LABEL2ID = {"False": 0, "True": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
CONDITIONS = ("base_text", "personality_only")

def convert_label(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str) and value in LABEL2ID:
        return LABEL2ID[value]
    raise ValueError(f"Unexpected Boolean label: {value!r}")

def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k] for k in (*CONDITIONS, "output")):
            raise ValueError("Missing input or label field")
        convert_label(row["output"])
    return rows

def development_rows(root):
    # No Test argument or Test loading on the development path.
    folder = Path(root) / "sources/upstream/data/correct_guess_task"
    return {split: read_rows(folder / filename) for split, filename in
            (("train", "train.csv"), ("validation", "val.csv"))}

class TextDataset:
    def __init__(self, texts, labels, tokenizer, max_length):
        if len(texts) != len(labels):
            raise ValueError("Inputs and labels are not aligned")
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
        self.labels = [convert_label(x) for x in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {**{key: values[index] for key, values in self.encodings.items()},
                "labels": self.labels[index]}

def label_summary(rows):
    converted = [convert_label(row["output"]) for row in rows]
    a = [convert_label(row["output"]) for row in rows if row["base_text"]]
    b = [convert_label(row["output"]) for row in rows if row["personality_only"]]
    if a != b or a != converted:
        raise ValueError("Paired condition labels differ")
    return dict(raw=dict(Counter(row["output"] for row in rows)),
                converted=dict(Counter(converted)), paired_labels_identical=True)

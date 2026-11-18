"""Export minimal Test decisions and verify public aggregates with the standard library."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean, stdev

from audit_stage5_results import BASE, PLAN, ROOT, close, read_json, read_jsonl, require, scores

REPORTS = ROOT / "reports"
EXPORT = REPORTS / "test_predictions.csv"
FIELDS = ("condition", "seed", "data_row_1based", "label_id", "prediction_id")


def local_rows():
    for run in PLAN["runs"]:
        records = read_jsonl(BASE / "runs" / run["run_id"] / "final_test_predictions.jsonl")
        for record in records:
            require(record["run_id"] == run["run_id"], "Unexpected prediction run")
            yield {field: record[field] for field in FIELDS}


def verify(rows):
    """Recalculate from released labels and class decisions in original row order."""
    runs = defaultdict(list)
    for row in rows:
        runs[row["condition"], int(row["seed"])].append(
            tuple(int(row[field]) for field in FIELDS[2:])
        )
    summary = read_json(REPORTS / "stage5_results.json")
    analysis = read_json(REPORTS / "results_analysis.json")
    seeds = analysis["seeds"]
    require(set(runs) == {(condition, seed) for condition in ("A", "B") for seed in seeds},
            "Unexpected condition or seed coverage")
    reference = None
    f1 = {}
    accuracy = {}
    for record in summary["per_run"]:
        key = record["condition"], record["seed"]
        indices, labels, predictions = zip(*runs[key])
        require(indices == tuple(range(1, 479)), f"Incomplete or misordered Test rows: {key}")
        if reference is None:
            reference = labels
        require(labels == reference and labels.count(0) == 109 and labels.count(1) == 369,
                f"Test labels differ: {key}")
        f1[key], accuracy[key], matrix = scores(labels, predictions)
        require(close(f1[key], record["test_macro_f1"]) and
                close(accuracy[key], record["test_accuracy"]) and
                matrix == analysis["confusion"][record["run_id"]], f"Run metrics differ: {key}")

    for condition in ("A", "B"):
        values = [f1[condition, seed] for seed in seeds]
        expected = summary["conditions"][condition]
        require(close(mean(values), expected["mean_macro_f1"]) and
                close(stdev(values), expected["sd_macro_f1_ddof1"]),
                f"Condition summary differs: {condition}")
    paired = [f1["B", seed] - f1["A", seed] for seed in seeds]
    expected = analysis["paired_difference"]
    require(all(close(actual, saved) for actual, saved in zip(paired, expected["values"], strict=True)) and
            close(mean(paired), expected["mean"]) and close(stdev(paired), expected["sd"]),
            "Paired summary differs")
    print(f"Verified {len(runs)} runs and {sum(map(len, runs.values()))} predictions. "
          f"Mean paired Macro-F1 difference: {mean(paired):.10f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true", help="Build the CSV from the original local run artifacts")
    args = parser.parse_args()
    if args.export:
        rows = list(local_rows())
        verify(rows)
        with EXPORT.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    else:
        with EXPORT.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            require(tuple(reader.fieldnames or ()) == FIELDS, "Unexpected prediction schema")
            verify(reader)


if __name__ == "__main__":
    main()

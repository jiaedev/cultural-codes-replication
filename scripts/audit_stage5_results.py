"""Independently audit completed fixed runs and publish concise results."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "configs/experimental_protocol.v1.json").read_text(encoding="utf-8"))
PLAN = json.loads((ROOT / "reports/experimental_run_plan.json").read_text(encoding="utf-8"))
BASE = ROOT / SPEC["outputs"]["root"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def require(test, message):
    if not test:
        raise RuntimeError(message)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12)


def scores(labels, predictions):
    matrix = [[0, 0], [0, 0]]
    for actual, predicted in zip(labels, predictions):
        require(actual in (0, 1) and predicted in (0, 1), "Nonbinary label")
        matrix[actual][predicted] += 1
    f1 = []
    for cls in (0, 1):
        tp = matrix[cls][cls]
        fp = matrix[1 - cls][cls]
        fn = matrix[cls][1 - cls]
        f1.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return statistics.mean(f1), (matrix[0][0] + matrix[1][1]) / len(labels), matrix


def write_once(path, content):
    encoded = content.encode("utf-8")
    if path.exists():
        require(path.read_bytes() == encoded, "Existing report differs: " + str(path))
    else:
        path.write_bytes(encoded)


def main():
    require(len(PLAN["runs"]) == 10, "Plan is not ten runs")
    activation = read_json(BASE / "activation.json")
    require(digest(ROOT / "configs/experimental_protocol.v1.json") == activation["protocol_sha256"],
            "Protocol changed")
    for path, expected in activation["code_sha256"].items():
        require(digest(ROOT / path) == expected, "Frozen execution source changed: " + path)
    require(digest(ROOT / "environment/requirements-win-gpu.lock.txt") ==
            activation["gpu_lock_sha256"], "GPU lock changed")
    require((BASE / "selection_lock.json").is_file(), "Ten selections are not locked; Test audit remains unavailable")
    selection_lock = read_json(BASE / "selection_lock.json")
    require(selection_lock["protocol_sha256"] == activation["protocol_sha256"],
            "Selection lock protocol changed")
    require(selection_lock["activation_sha256"] == digest(BASE / "activation.json"),
            "Selection lock activation changed")
    require(len(selection_lock["selected"]) == 10, "Ten selections not locked")
    summary = read_json(BASE / "summary.json")
    require(summary["runs_complete"] == summary["test_evaluations_complete"] == 10,
            "Final summary is incomplete")

    test_file = ROOT / SPEC["data"]["directory"] / SPEC["data"]["splits"]["test"]["file"]
    require(digest(test_file) == SPEC["data"]["splits"]["test"]["sha256"],
            "Test file bytes changed")
    with test_file.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    require(len(source_rows) == 478, "Test row count changed")
    labels = [SPEC["data"]["label2id"][row["output"]] for row in source_rows]
    row_ids = [
        hashlib.sha256(("test:" + SPEC["data"]["splits"]["test"]["sha256"] +
                        ":" + str(i)).encode()).hexdigest()
        for i in range(1, 479)
    ]
    require(len(set(row_ids)) == 478, "Case IDs collide")
    records = []
    orders = {}
    attempts_all = []
    for index, item in enumerate(PLAN["runs"]):
        run_id = item["run_id"]
        out = BASE / "runs" / run_id
        selection = read_json(out / "selected_checkpoint.json")
        require((selection["run_id"], selection["condition"], selection["seed"]) ==
                (run_id, item["condition"], item["seed"]), "Selection identity changed")
        require(selection["protocol_sha256"] == activation["protocol_sha256"],
                "Selection protocol changed")
        require(selection["optimizer_steps"] == 3990 and
                selection["train_examples_per_epoch"] == 8497 and
                selection["validation_examples_per_epoch"] == 544,
                "Research coverage count changed")
        require(digest(out / selection["checkpoint"] / "model.safetensors") ==
                selection["checkpoint_weight_sha256"], "Selected weights changed")
        require(selection_lock["selected"][index]["run_id"] == run_id and
                selection_lock["selected"][index]["selection_manifest_sha256"] ==
                digest(out / "selected_checkpoint.json"), "Selection lock mismatch")
        attempts = read_jsonl(out / "attempts.jsonl")
        require(attempts[-1]["status"] == "complete" and len(
            [a for a in attempts if a["status"] == "running"]) <= 2,
            "Attempt history incomplete")
        attempts_all.extend({"run_id": run_id, **a} for a in attempts)
        epoch_order = []
        for attempt in [a for a in attempts if a["status"] == "running"]:
            epoch_order.extend(read_jsonl(out / (
                "attempt-" + str(attempt["attempt"]) + "-epoch_case_order_hashes.jsonl")))
        require([x["epoch_1based"] for x in epoch_order] == list(range(1, 16)) and
                all(x["count"] == 8497 for x in epoch_order),
                "Train epoch rows or sequence incomplete")
        orders[item["seed"], item["condition"]] = [
            x["ordered_source_row_sha256"] for x in epoch_order]
        log = read_json(out / "train_log.json")
        evaluations = [entry for entry in log if "eval_validation_macro_f1" in entry]
        require(len(evaluations) == 15, "Expected fifteen Validation evaluations")
        best = max(e["eval_validation_macro_f1"] for e in evaluations)
        first_best = next(i for i, e in enumerate(evaluations, 1)
                          if e["eval_validation_macro_f1"] == best)
        require(selection["selected_epoch"] == first_best and
                close(selection["validation_macro_f1"], best) and
                selection["selected_step"] == first_best * 266,
                "Checkpoint selection differs from earliest Validation maximum")
        metrics = read_json(out / "final_test_metrics.json")
        predictions = read_jsonl(out / "final_test_predictions.jsonl")
        require(len(predictions) == 478 and metrics["evaluated_rows"] == 478,
                "Test prediction count incomplete")
        predicted = []
        column = item["input_column"]
        for i, (entry, row) in enumerate(zip(predictions, source_rows), 1):
            require(entry["run_id"] == run_id and entry["data_row_1based"] == i and
                    entry["case_id"] == row_ids[i - 1] and
                    entry["source_export_index"] == row[""] and
                    entry["input_sha256"] ==
                    hashlib.sha256(row[column].encode()).hexdigest() and
                    entry["label_id"] == labels[i - 1],
                    "Test prediction row, input or label mismatch")
            a, b = entry["logit_0"], entry["logit_1"]
            require(math.isfinite(a) and math.isfinite(b), "Nonfinite Test logit")
            pred = int(b > a)
            require(entry["prediction_id"] == pred, "Argmax prediction mismatch")
            predicted.append(pred)
        f1, accuracy, matrix = scores(labels, predicted)
        require(close(metrics["test_macro_f1"], f1) and
                close(metrics["test_accuracy"], accuracy) and
                metrics["confusion_matrix_labels_0_1"] == matrix and
                metrics["selection_lock_sha256"] == digest(BASE / "selection_lock.json"),
                "Independently calculated Test metrics differ")
        require(close(metrics["validation_macro_f1"], best) and
                metrics["selected_checkpoint_weight_sha256"] ==
                selection["checkpoint_weight_sha256"], "Metric provenance differs")
        records.append({key: metrics[key] for key in (
            "run_id", "condition", "seed", "status", "selected_epoch",
            "selected_checkpoint", "validation_macro_f1", "test_macro_f1",
            "test_accuracy", "evaluated_rows", "runtime_seconds",
            "protocol_sha256", "environment_lock_sha256")})
        records[-1]["train_runtime_seconds"] = selection["elapsed_seconds"]
        records[-1]["peak_reserved_bytes"] = selection["peak_reserved_bytes"]
    for seed in SPEC["scope"]["seeds"]:
        require(orders[seed, "A"] == orders[seed, "B"],
                "Paired case orders differ at seed " + str(seed))
    by = {(r["seed"], r["condition"]): r for r in records}
    pairs = [{"seed": seed,
              "A_macro_f1": by[seed, "A"]["test_macro_f1"],
              "B_macro_f1": by[seed, "B"]["test_macro_f1"],
              "B_minus_A": by[seed, "B"]["test_macro_f1"] -
                           by[seed, "A"]["test_macro_f1"]}
             for seed in SPEC["scope"]["seeds"]]
    for condition in ("A", "B"):
        values = [by[seed, condition]["test_macro_f1"]
                  for seed in SPEC["scope"]["seeds"]]
        reported = summary["conditions"][condition]
        require(close(reported["mean_macro_f1"], statistics.mean(values)) and
                close(reported["sd_macro_f1_ddof1"], statistics.stdev(values)),
                "Condition summary mismatch")
    differences = [p["B_minus_A"] for p in pairs]
    require(close(summary["paired_B_minus_A"]["mean"],
                  statistics.mean(differences)) and
            close(summary["paired_B_minus_A"]["sd_ddof1"],
                  statistics.stdev(differences)), "Paired summary mismatch")
    with (BASE / "per_run.csv").open(newline="", encoding="utf-8") as stream:
        primary_csv = list(csv.DictReader(stream))
    require(len(primary_csv) == 10 and
            [x["run_id"] for x in primary_csv] ==
            [x["run_id"] for x in records], "Per-run CSV order changed")
    for row, record in zip(primary_csv, records):
        for field in ("condition", "status", "selected_checkpoint"):
            require(row[field] == str(record[field]), "Per-run CSV field changed: " + field)
        for field in ("seed", "selected_epoch", "evaluated_rows"):
            require(int(row[field]) == record[field], "Per-run CSV count changed: " + field)
        for field in ("validation_macro_f1", "test_macro_f1", "test_accuracy"):
            require(close(row[field], record[field]), "Per-run CSV score changed: " + field)
    with (BASE / "paired_differences.csv").open(newline="", encoding="utf-8") as stream:
        paired_csv = list(csv.DictReader(stream))
    require(len(paired_csv) == 5, "Paired CSV row count changed")
    for row, pair in zip(paired_csv, pairs):
        require(int(row["seed"]) == pair["seed"] and
                all(close(row[field], pair[field])
                    for field in ("A_macro_f1", "B_macro_f1", "B_minus_A")),
                "Paired CSV differs from independently calculated scores")
    training_seconds = sum(float(a.get("elapsed_seconds", 0)) for a in attempts_all
                           if a["status"] in ("complete", "failed"))
    budget = read_json(ROOT / "configs/execution_budget.json")
    require(training_seconds <= budget["total_gpu_wall_time_limit_seconds"],
            "Declared training budget exceeded")
    report = {
        "status": "passed",
        "scope": "independent post-selection result and coverage audit",
        "protocol_id": SPEC["protocol_id"],
        "protocol_sha256": activation["protocol_sha256"],
        "activation_sha256": digest(BASE / "activation.json"),
        "selection_lock_sha256": digest(BASE / "selection_lock.json"),
        "audit_script_sha256": digest(Path(__file__)),
        "test_source_sha256": digest(test_file),
        "runs_complete": 10,
        "test_evaluations_complete": 10,
        "test_rows_per_run": 478,
        "total_test_prediction_rows": 4780,
        "train_epochs_per_run": 15,
        "validation_evaluations_per_run": 15,
        "paired_case_orders_equal": True,
        "training_wall_seconds_including_failed_attempts": training_seconds,
        "training_budget_seconds": budget["total_gpu_wall_time_limit_seconds"],
        "attempts": attempts_all,
        "per_run": records,
        "conditions": summary["conditions"],
        "paired_B_minus_A": summary["paired_B_minus_A"],
        "limitations": [
            "No participant/game-level leakage audit is possible from released IDs.",
            "Exact identity of the paper's original pretrained checkpoint is unknown.",
            "Descriptive paired seed variation is not uncertainty over Test cases."
        ]
    }
    write_once(ROOT / "reports/stage5_results.json",
               json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for name in ("per_run.csv", "paired_differences.csv"):
        write_once(ROOT / "reports" / ("stage5_" + name),
                   (BASE / name).read_text(encoding="utf-8"))
    print(json.dumps({"status": "passed", "runs": 10, "test_rows": 4780,
                      "training_wall_seconds": training_seconds,
                      "condition_means": summary["conditions"],
                      "paired_mean": summary["paired_B_minus_A"]["mean"]}, indent=2))


if __name__ == "__main__":
    main()



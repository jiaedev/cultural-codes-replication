"""Export public, path-free provenance for the completed fixed experiment."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    spec = read(ROOT / "configs/experimental_protocol.v1.json")
    plan = read(ROOT / "reports/experimental_run_plan.json")
    base = ROOT / spec["outputs"]["root"]
    activation = read(base / "activation.json")
    locked = read(base / "selection_lock.json")
    correction = read(ROOT / "reports/stage5_evaluation_correction.json")
    results = read(ROOT / "reports/stage5_results.json")
    require(activation["protocol_sha256"] ==
            sha(ROOT / "configs/experimental_protocol.v1.json"), "Protocol changed")
    require(locked["activation_sha256"] == sha(base / "activation.json"),
            "Selection lock changed")
    require(len(plan["runs"]) == len(locked["selected"]) == 10, "Runs missing")
    require(results["status"] == "passed" and results["runs_complete"] == 10,
            "Independent result audit missing")
    require(correction["corrected_evaluator_sha256"] ==
            sha(ROOT / "scripts/evaluate_final_corrected.py"),
            "Corrected evaluator changed")
    for name, digest in activation["code_sha256"].items():
        require(sha(ROOT / name) == digest, "Frozen source changed: " + name)
    selected = []
    for item, entry in zip(plan["runs"], locked["selected"]):
        require(item["run_id"] == entry["run_id"], "Run order differs")
        out = base / "runs" / item["run_id"]
        manifest = out / "selected_checkpoint.json"
        record = read(manifest)
        require(sha(manifest) == entry["selection_manifest_sha256"],
                "Selection manifest changed")
        require(sha(out / record["checkpoint"] / "model.safetensors") ==
                entry["checkpoint_weight_sha256"], "Selected weights changed")
        selected.append({
            "run_id": item["run_id"],
            "condition": item["condition"],
            "seed": item["seed"],
            "selected_epoch": record["selected_epoch"],
            "checkpoint": record["checkpoint"],
            "selection_manifest_sha256": entry["selection_manifest_sha256"],
            "checkpoint_weight_sha256": entry["checkpoint_weight_sha256"],
            "initial_snapshot_sha256": record["initial_snapshot_sha256"]
        })
    public = {
        "status": "completed_and_audited",
        "protocol_id": spec["protocol_id"],
        "protocol_sha256": activation["protocol_sha256"],
        "source_commit_at_activation": activation["source_commit"],
        "source_tree_dirty_at_activation": activation["source_tree_dirty"],
        "frozen_execution_code_sha256": activation["code_sha256"],
        "evaluation_correction_code_sha256":
            correction["corrected_evaluator_sha256"],
        "gpu_environment_lock_sha256": activation["gpu_lock_sha256"],
        "environment": activation["environment"],
        "model_repository": spec["model"]["repository"],
        "model_revision": spec["model"]["revision"],
        "pretrained_weights_sha256": activation["source_weights_sha256"],
        "tokenizer_manifest_sha256": activation["tokenizer_manifest_sha256"],
        "data_sha256": activation["data_sha256"],
        "budget_sha256": activation["budget_sha256"],
        "activation_manifest_sha256": sha(base / "activation.json"),
        "selection_lock_sha256": sha(base / "selection_lock.json"),
        "result_audit_sha256": sha(ROOT / "reports/stage5_results.json"),
        "evaluation_correction_report_sha256":
            sha(ROOT / "reports/stage5_evaluation_correction.json"),
        "selected": selected,
        "exporter_sha256": sha(Path(__file__))
    }
    path = ROOT / "reports/stage5_provenance.json"
    data = (json.dumps(public, indent=2, ensure_ascii=False) + "\n").encode()
    if path.exists():
        require(path.read_bytes() == data, "Existing public provenance differs")
    else:
        path.write_bytes(data)
    print(json.dumps({"status": "passed", "selected": len(selected),
                      "dirty_activation": public["source_tree_dirty_at_activation"],
                      "public_manifest": str(path.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
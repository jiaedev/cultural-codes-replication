"""Record evidence for the protocol's execution gates without opening Test cases."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

def main():
    spec_path = ROOT / "configs/experimental_protocol.v1.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec["protocol_id"] != "cc-fixed-v1.0.1":
        raise ValueError("Unexpected protocol; review the readiness checker")
    checks = {}
    for report_name, script_name in (
        ("protocol_validation.json", "validate_protocol.py"),
        ("accumulation_validation.json", "validate_accumulation.py"),
    ):
        report = json.loads((ROOT / "reports" / report_name).read_text(encoding="utf-8"))
        checks[report_name] = (
            report["protocol_sha256"] == sha256(spec_path)
            and report["bridge_sha256"] == sha256(ROOT / "src/cultural_codes/protocol.py")
            and report["script_sha256"] == sha256(ROOT / "scripts" / script_name)
        )
    plan = json.loads((ROOT / "reports/experimental_run_plan.json").read_text(encoding="utf-8"))
    resolved = json.loads((ROOT / "reports/resolved_protocol_arguments.json").read_text(encoding="utf-8"))
    checks["ten_planned_runs_consistent"] = (
        plan["protocol_id"] == spec["protocol_id"]
        and len(plan["runs"]) == len(resolved["runs"]) == 10
        and all(run["run_id"] in resolved["runs"] and
                resolved["runs"][run["run_id"]]["seed"] == run["seed"] for run in plan["runs"])
    )
    for split, entry in spec["data"]["splits"].items():
        path = ROOT / spec["data"]["directory"] / entry["file"]
        checks[f"{split}_source_bytes_match"] = path.is_file() and sha256(path) == entry["sha256"]
    model_file = ROOT / spec["model"]["local_config"] / spec["model"]["pretrained_file"]
    checks["pretrained_weights_present_and_verified"] = (
        model_file.is_file() and model_file.stat().st_size == spec["model"]["pretrained_bytes"]
        and sha256(model_file) == spec["model"]["pretrained_sha256"]
    )
    checks["gpu_lock_recorded"] = any((ROOT / "environment").glob("requirements-*-gpu.lock.txt"))
    checks["full_model_gpu_preflight_recorded"] = (ROOT / "reports/full_model_gpu_preflight.json").is_file()
    checks["research_training_runner_present"] = (ROOT / "scripts/train_research.py").is_file()
    checks["final_test_runner_present"] = (ROOT / "scripts/evaluate_final.py").is_file()
    checks["resource_ceiling_recorded"] = (ROOT / "configs/execution_budget.json").is_file()
    checks["activation_manifest_recorded"] = (ROOT / "artifacts/experiments" / spec["protocol_id"] / "activation.json").is_file()
    checks["research_activation_enabled"] = spec["execution_ready"] is True
    import torch
    environment = {"operating_system": platform.system(), "architecture": platform.machine(),
                   "python": platform.python_version(), "torch": torch.__version__,
                   "torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    checks["verified_target_gpu_available_here"] = bool(torch.cuda.is_available() and torch.version.cuda)
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_id": spec["protocol_id"], "protocol_sha256": sha256(spec_path),
              "source_commit": git("rev-parse", "HEAD"), "working_tree_dirty": bool(git("status", "--porcelain")),
              "environment": environment, "checks": checks,
              "ready_for_research_training": all(checks.values()),
              "completed_training_runs": 0, "completed_final_test_evaluations": 0,
              "note": "Readiness evidence only. Source splits are hashed without parsing or tokenizing Test content. No training or predictions."}
    dest = ROOT / "reports/execution_readiness.json"
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

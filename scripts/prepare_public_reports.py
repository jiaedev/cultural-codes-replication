"""Export publication summaries without modifying original validation records."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def main():
    environment = read("reports/environment.json")
    aggregate = read("reports/stage3_validation.json")
    # Refuse to publish summaries against altered original evidence.
    for name, expected in aggregate["report_hashes"].items():
        if sha(name) != expected:
            raise ValueError(f"Historical evidence hash mismatch: {name}")
    for name, expected in aggregate["implementation_hashes"].items():
        if sha(name) != expected:
            raise ValueError(f"Historical implementation hash mismatch: {name}")

    fields = ("captured_at_utc", "python", "is_virtual_environment", "platform", "machine",
              "cpu", "logical_cpu_count", "physical_cpu_count", "ram_bytes",
              "torch_version", "torch_cuda_runtime", "torch_cuda_available",
              "nvidia_smi", "nvidia_smi_returncode", "validated_device",
              "validated_world_size", "installed_packages",
              "gpu_training_validated", "full_model_training_validated")
    summary = {name: environment[name] for name in fields}
    summary["publication_note"] = "Derived from an unchanged local environment record; interpreter paths and free-disk state omitted."
    summary["source_record_sha256"] = sha("reports/environment.json")
    write("reports/environment_summary.json", summary)

    installed = {normalized(name): version for name, version in environment["installed_packages"].items()}
    artifacts = {}
    install_reports = sorted((ROOT / "reports").glob("install_*.json"))
    for report_path in install_reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for item in report["install"]:
            metadata = item["metadata"]
            name, version = metadata["name"], metadata["version"]
            if installed.get(normalized(name)) != version:
                continue
            download = item["download_info"]
            artifacts[normalized(name)] = dict(name=name, version=version, url=download["url"],
                archive_hashes=download.get("archive_info", {}).get("hashes", {}))
    artifact_report = dict(installed_artifacts=[artifacts[name] for name in sorted(artifacts)],
        bootstrap_or_unrecorded_distributions=sorted(set(installed)-set(artifacts)),
        note="Observed download records matching final installed versions; bootstrap distributions may lack pip-install download records.",
        source_install_report_sha256={p.name:sha(p.relative_to(ROOT)) for p in install_reports})
    write("reports/dependency_artifacts.json", artifact_report)

    checks = {name:value for name,value in aggregate["checks"].items()
              if name not in ("relocation_inventory_and_git_preserved", "no_remote")}
    hashes = {name:value for name,value in aggregate["report_hashes"].items()
              if name not in ("reports/relocation.json", "reports/environment.json")}
    hashes.update({name:sha(name) for name in
                   ("reports/environment_summary.json", "reports/dependency_artifacts.json")})
    implementations = {name:value for name,value in aggregate["implementation_hashes"].items()
                       if name != "scripts/relocation_audit.py"}
    write("reports/research_validation_summary.json", dict(
        validated_at_utc=aggregate["generated_at_utc"], status=aggregate["status"], checks=checks,
        report_hashes=hashes, implementation_hashes=implementations,
        research_training_performed=aggregate["research_training_performed"],
        research_performance_comparison_performed=aggregate["research_performance_comparison_performed"],
        research_test_predictions_generated=aggregate["research_test_predictions_generated"],
        stage4_protocol_frozen=aggregate["stage4_protocol_frozen"],
        publication_note="Derived from unchanged Stage 3 records. Local path/inventory checks are omitted; no experiment was rerun.",
        source_aggregate_sha256=sha("reports/stage3_validation.json")))
    print("Wrote environment_summary.json, dependency_artifacts.json, research_validation_summary.json")


if __name__ == "__main__":
    main()

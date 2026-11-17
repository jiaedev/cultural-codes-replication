"""Capture the actual final environment and an exact Windows CPU package lock."""
import importlib.metadata as meta
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
import psutil
import torch
ROOT = Path(__file__).resolve().parents[1]

def main():
    packages = dict(sorted((d.metadata["Name"], d.version) for d in meta.distributions()))
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True, encoding="utf-8")
    environment = ROOT / "environment"
    environment.mkdir(exist_ok=True)
    (environment / "requirements-win-cpu.lock.txt").write_text(
        "# Exact installed snapshot: Windows x64, CPython 3.11.17, CPU execution.\n"
        "--extra-index-url https://download.pytorch.org/whl/cpu\n"+frozen, encoding="utf-8")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                         capture_output=True, text=True, encoding="utf-8")
    disk = shutil.disk_usage(ROOT)
    report = dict(captured_at_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
        executable=sys.executable, base_executable=getattr(sys, "_base_executable", None),
        prefix=sys.prefix, base_prefix=sys.base_prefix, is_virtual_environment=sys.prefix != sys.base_prefix,
        platform=platform.platform(), machine=platform.machine(), cpu=platform.processor(),
        logical_cpu_count=psutil.cpu_count(), physical_cpu_count=psutil.cpu_count(logical=False),
        ram_bytes=psutil.virtual_memory().total, disk_free_bytes=disk.free,
        torch_version=torch.__version__, torch_cuda_runtime=torch.version.cuda,
        torch_cuda_available=torch.cuda.is_available(),
        nvidia_smi=smi.stdout.strip(), nvidia_smi_returncode=smi.returncode,
        validated_device="cpu", validated_world_size=1, installed_packages=packages,
        gpu_training_validated=False, full_model_training_validated=False)
    (ROOT / "reports/environment.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "installed_packages"}, indent=2))

if __name__ == "__main__":
    main()

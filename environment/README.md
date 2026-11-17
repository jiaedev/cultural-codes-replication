# Environment and reproduction

## Verified environment

The CPU execution path was verified on Windows x64 (OS build 10.0.26300), PowerShell 7.6.9 and CPython 3.11.17 in a dedicated virtual environment.

| Package | Version |
| --- | --- |
| PyTorch | 2.9.1+cpu |
| Transformers | 4.57.1 |
| Accelerate | 1.12.0 |
| scikit-learn | 1.7.2 |
| tokenizers | 0.22.2 |
| regex distribution | 2025.9.18 |

The [exact installed lock](requirements-win-cpu.lock.txt), [environment summary](../reports/environment_summary.json) and [dependency artifact records](../reports/dependency_artifacts.json) retain versions, host capabilities and observed wheel URLs/hashes. This is a pinned Windows CPU snapshot, not the original 2023 environment or a cross-platform lock.

Compatibility references consulted on 2026-11-17: [PyTorch versioned installation](https://pytorch.org/get-started/previous-versions/), [Transformers 4.57.1 installation](https://huggingface.co/docs/transformers/v4.57.1/en/installation), [Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer) and [scikit-learn F1](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.f1_score.html). Installed imports, dependency checks and CPU execution were also tested.

## 1. Create the environment

Run from the repository root in PowerShell. Select an existing Python 3.11 executable; 3.11.17 was tested. If necessary, replace `python` with the selected interpreter in the first two commands.

```powershell
python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"
python -m venv .venv
$env:PIP_CACHE_DIR = Join-Path (Get-Location) '.cache\pip'
.\.venv\Scripts\python.exe -m pip install -r environment/requirements-win-cpu.lock.txt
.\.venv\Scripts\python.exe -m pip check
```

[setup_environment.ps1](../scripts/setup_environment.ps1) provides equivalent setup with a `-PythonExecutable` argument. Direct invocation of the virtual-environment interpreter avoids shell activation. Recreate environments after changing their location rather than copying them. The helper was successfully replayed against the installed lock; this was not a second clean-machine installation.

## 2. Restore recorded inputs

The repository retains manifests and small source files; CSVs and tokenizer snapshots are omitted from Git. Restore missing files from the exact recorded URLs:

```powershell
.\.venv\Scripts\python.exe scripts/restore_recorded_files.py
```

This checks both [source](../sources/manifest.json) and [tokenizer](../reports/tokenizer_manifest.json) manifests. Existing matching files are left unchanged; different existing bytes cause an error. Downloads are verified against recorded SHA-256 before writing. Only the pinned tokenizer/config files are retrieved, not pretrained weights.

Use `--check-only` for offline verification when the files are already present. On a fresh checkout, `acquire_sources.py` and `prepare_tokenizer.py` are not substitutes for restoration: their existing-manifest guards assume the original snapshot is already present. They remain the original acquisition utilities.

## 3. Run the checks

```powershell
.\.venv\Scripts\python.exe scripts/validate_dataset.py
.\.venv\Scripts\python.exe scripts/audit_labels.py
.\.venv\Scripts\python.exe scripts/audit_tokenization.py
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HOME = Join-Path (Get-Location) '.cache\huggingface'
.\.venv\Scripts\python.exe scripts/smoke_test.py --device cpu
```

The standard-library dataset audit verifies source integrity, schema, labels, paired inputs, missingness, duplicates and overlap. The label audit checks all splits without Test predictions. The tokenizer audit reads Train/Validation only and uses local files; it disables tensor backends because tokenizer/config checks do not require PyTorch. A message that models are unavailable in that tokenizer-only process is expected.

The smoke test uses invented strings and randomly initialized tiny-model weights. Its two optimizer steps, short sequence length, seed, rate and per-step evaluation/saving are technical fixtures, not a research protocol. It writes [structured results](../reports/smoke_cpu.json) and local synthetic checkpoints. The adapter is verified for one process and explicitly rejects multi-process evaluation.

Outputs are documented in the combined [validation report](../docs/validation.md). Check exit status and output timestamps; an earlier report may remain after a failed command.

## Windows execution constraints

The official torch 2.9.1 CUDA 12.8 wheel installed but failed to import under Windows application control (WinError 4551; Code Integrity events 3033/3077). The import named `shm.dll`; relevant events named `torch_cpu.dll`. The official CPU wheel imported and passed execution checks. The initially resolved regex 2026.11.15 extension was similarly blocked; the pinned 2025.9.18 wheel imported successfully. The policy's underlying rationale was not established, so this is not evidence of a CUDA driver defect.

The host has an RTX 5080 Laptop GPU (16,303 MiB reported VRAM, driver 592.27), but the current lock is CPU only. CUDA execution, full-model memory use, mixed precision and throughput require separate validation. Retain this CPU lock if a GPU environment specification is later added.

## Public evidence and local diagnostics

Public reports retain scientific checks, package versions and artifact provenance. Raw interpreter paths, installation transcripts, machine-specific operational history and the original aggregate record remain local and ignored. No raw result, source manifest or data file is edited for publication.

`scripts/capture_environment.py` captures local diagnostics and the installed lock. When publishing an existing verification snapshot, `python scripts/prepare_public_reports.py` derives the path-free environment/dependency/validation summaries from the local records. It does not rerun experiments. A fresh checkout already contains these public summaries; recapturing and replacing an evidence snapshot is unnecessary for running the checks.

## Protocol target environment

The recommended main-experiment target is a separate Linux x86_64, single NVIDIA GPU environment with CPython 3.11.17, torch 2.9.1+cu128 and the same core Transformers/Accelerate/metric versions. This is not an installed or validated GPU lock. Preserve the verified Windows CPU specification. Before research runs, record the target's complete installed lock and run the full-size synthetic preflight specified in the [protocol](../docs/experimental_protocol.md), including FP32/eager micro-batch 8, length 512 and four-way accumulation. No Windows CUDA reinstall or security-policy change was made during protocol design.

`python scripts/validate_protocol.py` uses the existing CPU environment to resolve/test settings with tiny random models and invented inputs only. It does not provide a research training CLI or establish GPU readiness.

The v1.0.1 loss correction is checked with `python scripts/validate_accumulation.py`. It uses invented inputs and tiny random models on CPU, compares accumulated gradients against a whole-group mean loss, exercises clipping and resumes a complete epoch checkpoint in a separate process. Results are in [accumulation validation](../reports/accumulation_validation.json); these checks do not validate GPU or full-model execution.

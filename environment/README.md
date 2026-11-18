# Environment and reproduction

## Historical CPU verification

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

## Historical Windows CUDA import check

The official torch 2.9.1 CUDA 12.8 wheel installed but failed to import under Windows application control (WinError 4551; Code Integrity events 3033/3077). The import named `shm.dll`; relevant events named `torch_cpu.dll`. The official CPU wheel imported and passed execution checks. The initially resolved regex 2026.11.15 extension was similarly blocked; the pinned 2025.9.18 wheel imported successfully. The policy's underlying rationale was not established, so this is not evidence of a CUDA driver defect.

This observation belongs to the CPU verification period. The later CUDA environment and executed training are documented below.

## Public evidence and local diagnostics

Public reports retain scientific checks, package versions and artifact provenance. Raw interpreter paths, installation transcripts, machine-specific operational history and the original aggregate record remain local and ignored. No raw result, source manifest or data file is edited for publication.

`scripts/capture_environment.py` captures local diagnostics and the installed lock. When publishing an existing verification snapshot, `python scripts/prepare_public_reports.py` derives the path-free environment/dependency/validation summaries from the local records. It does not rerun experiments. A fresh checkout already contains these public summaries; recapturing and replacing an evidence snapshot is unnecessary for running the checks.

## Protocol target environment

At the earlier protocol-design point, a separate Linux x86_64 single-GPU environment was recommended because the first Windows CUDA import had failed. That was a dated contingency. The later verified local Windows GPU snapshot below is the executed environment; the Windows CPU specification remains preserved. No security policy was changed.

`python scripts/validate_protocol.py` uses the existing CPU environment to resolve/test settings with tiny random models and invented inputs only. It does not provide a research training CLI or establish GPU readiness.

The v1.0.1 loss correction is checked with `python scripts/validate_accumulation.py`. It uses invented inputs and tiny random models on CPU, compares accumulated gradients against a whole-group mean loss, exercises clipping and resumes a complete epoch checkpoint in a separate process. Results are in [accumulation validation](../reports/accumulation_validation.json); these checks do not validate GPU or full-model execution.

## Local GPU execution snapshot

The user selected this PC's RTX 5080 Laptop GPU with no external service cost and a total GPU training ceiling of 24 hours. An isolated `.venv-gpu` uses CPython 3.11.17, torch 2.9.1+cu128, Transformers 4.57.1 and Accelerate 1.12.0; [the exact installed lock](requirements-win-gpu.lock.txt) preserves the full package set. CUDA import, a CUDA tensor operation, the 512-token full-size pretrained [synthetic preflight](../reports/full_model_gpu_preflight.json) and 100/170-token [short timing blocks](../reports/gpu_synthetic_benchmark.json) passed locally. The earlier Windows application-control failure was not reproduced with this new isolated wheel. No security policy was changed.

The preflight's peak reserved memory was about 7.7 GB at length 512. The five-update invented blocks took 3.96 seconds at 100 tokens and 4.65 seconds at 170 tokens. Multiplying those short blocks by the planned updates gives roughly 9.5 hours of training-loop work across ten runs; real data lengths, Validation, checkpoint I/O, laptop thermal behavior and interruptions make that an uncertain estimate. The declared ceiling was 24 hours. The measured ten-fit total was 18,811.539 seconds (5 h 13 min 31.5 s), distinct from the short-block estimate; see the [execution report](../docs/stage5_training_evaluation.md).

The completed local result can be checked without retraining by running the post-run audit with the recorded GPU interpreter: .\.venv-gpu\Scripts\python.exe -B scripts/audit_stage5_results.py. The corrected final evaluator and its one-line change are documented in [evaluation correction provenance](../reports/stage5_evaluation_correction.json); completed prediction files are protected against overwrite.

## Full training and evaluation replay

Use a separate checkout with a fresh artifacts directory. The replay writes new validation reports, resolved arguments and run artifacts there. The recorded execution used PowerShell 7 on Windows x64, CPython 3.11.17, PyTorch 2.9.1+cu128, Transformers 4.57.1, Accelerate 1.12.0 and one RTX 5080 Laptop GPU with driver 592.27. The checked-in evidence documents that environment. A separate local checkout passed hash verification for all 16 source files and five tokenizer files using locally retained inputs. Fresh package installation, downloading the inputs again and replaying all ten fits remain unverified. [GPU lock](requirements-win-gpu.lock.txt) · [Execution evidence](../docs/stage5_training_evaluation.md)

The ten fit durations on that GPU summed to 18,811.539 seconds (5 h 13 min 31.5 s). The retained experiment and model-candidate directories together occupy about 30.7 GiB, measured as the sum of file sizes after the run. Reserve at least 50 GiB to allow for the environment, caches and temporary checkpoints. Setup, downloads, evaluation and time on another GPU can change the total.

### 1. Install and restore pinned inputs

From the separate checkout root in PowerShell 7.4 or newer, select a CPython 3.11.17 executable as `python`.

```powershell
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
python -c "import platform; assert platform.python_version() == '3.11.17'"
python -m venv .venv-gpu
$gpuPython = '.\.venv-gpu\Scripts\python.exe'
$env:PIP_CACHE_DIR = Join-Path (Get-Location) '.cache\pip'
& $gpuPython -m pip install -r environment/requirements-win-gpu.lock.txt
& $gpuPython -m pip check
& $gpuPython -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print(torch.__version__, torch.cuda.get_device_name(0))"
& $gpuPython -B scripts/restore_recorded_files.py
& $gpuPython -B scripts/restore_pretrained.py
& $gpuPython -B scripts/restore_recorded_files.py --check-only
```

The restoration commands retrieve missing files from pinned revisions and verify SHA-256. The data and tokenizer files come from the recorded manifests. The pretrained safetensors file comes from the pinned Hugging Face revision and must match the size and hash in the protocol configuration. [Source manifest](../sources/manifest.json) · [Tokenizer manifest](../reports/tokenizer_manifest.json) · [Model and training configuration](../configs/experimental_protocol.v1.json)

### 2. Prepare pairs and activate

Run this block only in the separate checkout, using the same shell. Snapshot creation records the configuration hash and the activation gate expects the archived preactivation hash. This block supplies that configuration during creation and restores the active bytes before running the checks. Both configurations use the same initialization and scientific settings.

```powershell
$configPath = (Resolve-Path 'configs/experimental_protocol.v1.json').Path
$archivedConfigPath = (Resolve-Path 'reports/history/cc-fixed-v1.0.1-preactivation/configs/experimental_protocol.v1.json').Path
$activeConfig = [IO.File]::ReadAllBytes($configPath)
try {
    [IO.File]::WriteAllBytes($configPath, [IO.File]::ReadAllBytes($archivedConfigPath))
    foreach ($seed in 13, 37, 73, 109, 149) {
        & $gpuPython -B scripts/train_research.py prepare --seed $seed
    }
} finally {
    [IO.File]::WriteAllBytes($configPath, $activeConfig)
}
& $gpuPython -B scripts/validate_protocol.py
& $gpuPython -B scripts/validate_accumulation.py
& $gpuPython -B scripts/preflight_full_model_gpu.py
& $gpuPython -B scripts/activate_experiment.py
```

The first two checks use the CPU even when invoked from the CUDA environment. The synthetic GPU preflight verifies the full candidate path and final partial batch before activation. Activation checks the package lock, run order, paired snapshots and local evidence. Treat a failed gate as a stop and inspect its reported mismatch. [Run plan](../reports/experimental_run_plan.json) · [Fixed protocol](../docs/experimental_protocol.md)

### 3. Run training and final evaluation

```powershell
& $gpuPython -B scripts/execute_plan.py
& $gpuPython -B scripts/evaluate_final_corrected.py
```

The first command runs the ten scheduled fits. Both runners check the Test file hash when loading the protocol. The separate evaluator locks all ten Validation selections before parsing Test rows. It writes per-row predictions, per-run metrics and aggregate results under `artifacts/experiments/cc-fixed-v1.0.1/`. Its label-conversion correction is part of the documented original evaluation. Use a fresh result directory for this replay. After an interrupted fit the runner requires a complete checkpoint to resume. If the first attempt stops before producing one, its CLI cannot perform the protocol’s fallback restart from initialization. Both conditions use the same model reload path, which selected SDPA in a check of the recorded environment. [Attention on model reload](../docs/stage5_training_evaluation.md#attention-on-model-reload). [Evaluation correction](../reports/stage5_evaluation_correction.json)

### 4. Audit and analyze the new run

The audit writes three reports under fixed names and refuses to overwrite different bytes. In the separate checkout, move those reference copies into an ignored folder first. Keep ReportLab in a separate environment so the CUDA package lock stays exact.

```powershell
$referenceReports = 'artifacts/reference-results'
New-Item -ItemType Directory -Path $referenceReports | Out-Null
foreach ($file in 'stage5_results.json', 'stage5_per_run.csv', 'stage5_paired_differences.csv') {
    Move-Item -LiteralPath (Join-Path 'reports' $file) -Destination $referenceReports
}
& $gpuPython -B scripts/audit_stage5_results.py
python -m venv .venv-report
$reportPython = '.\.venv-report\Scripts\python.exe'
& $reportPython -m pip install 'reportlab==4.4.9'
& $reportPython -B scripts/analyze_results.py --replay --output-dir artifacts/replay-analysis
```

The replay flag omits checks specific to the original evaluator failure and uses the same metric and analysis rules. The class CSV, analysis JSON and seed SVG go into `artifacts/replay-analysis/`. Compare new and reference outputs with the same metric definitions. Exact numerical or checkpoint-byte equality across different systems is unverified.

To re-evaluate the original selected models, use a separate copy of the original activation and selection records, checkpoint weights, pinned data and tokenizer with the matching environment. The committed prediction CSV reproduces scores from saved decisions. It cannot reconstruct model weights. In the separate copy, omit the ten per-run prediction and metric outputs and the top-level `per_run.csv`, `paired_differences.csv` and `summary.json`. The evaluator creates these files exclusively and stops if they already exist.

# Fixed training and final evaluation

## Scope and provenance

The project executed protocol cc-fixed-v1.0.1 on the released correct_guess_task Train/Validation/Test splits. Condition A used base_text; condition B used personality_only. Five paired seeds (13, 37, 73, 109, 149) produced ten independent pretrained RoBERTa-base starts. No hyperparameter search, performance-based setting change, model reuse between runs, data cleaning or Test-based checkpoint selection was performed.

The active configuration SHA-256 is dea267bebe4c276547d07ce986a8969d47ac67eef06601356e57a3c5e8c05498. The ignored activation manifest at artifacts/experiments/cc-fixed-v1.0.1/activation.json records the dirty Git tree's execution-source hashes, released-data hashes, tokenizer/model revision, exact pretrained weight bytes, paired initial snapshots, resolved arguments and run order. The candidate is FacebookAI/roberta-base at revision e2da8e2f811d1448a5b465c236feacd80ffbac7b; identity with the paper's original checkpoint remains unverified. The local environment used Python 3.11.17, torch 2.9.1+cu128, Transformers 4.57.1 and one RTX 5080 Laptop GPU (driver 592.27). See the [GPU environment lock](../environment/requirements-win-gpu.lock.txt), [protocol](experimental_protocol.md) and [source manifest](../sources/manifest.json).

The [public provenance manifest](../reports/stage5_provenance.json) exports file-level hashes from the activation and locked selections without personal absolute paths. It records that the activation working tree was dirty; the recorded commit alone does not identify the executed source.

Each run used 15 epochs and 3,990 optimizer updates, with 8,497 Train examples and 544 Validation examples per epoch. The runner logged each full Train case order and Validation result. Validation Macro-F1 selected the highest scoring epoch, retaining the earliest epoch for an exact tie. All ten selection manifests were locked before the separate final Test evaluator opened the 478-row Test split. The complete per-run and paired values are in [machine-readable results](../reports/stage5_results.json), [per-run CSV](../reports/stage5_per_run.csv) and [paired CSV](../reports/stage5_paired_differences.csv).

## Per-run results

Scores are reported for every planned run, including low scores. Test rows are 478 in every row below. Checkpoint numbers are optimizer steps.

| Condition | Seed | Selected epoch / checkpoint | Validation Macro-F1 | Test Macro-F1 | Test accuracy | Training min |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| A | 13 | 12 / 3192 | 0.5917 | 0.5665 | 0.7657 | 26.42 |
| B | 13 | 4 / 1064 | 0.6774 | 0.6743 | 0.7782 | 36.28 |
| B | 37 | 8 / 2128 | 0.6969 | 0.6859 | 0.7929 | 36.01 |
| A | 37 | 6 / 1596 | 0.5530 | 0.5792 | 0.7762 | 26.67 |
| A | 73 | 10 / 2660 | 0.5847 | 0.5397 | 0.7699 | 26.98 |
| B | 73 | 14 / 3724 | 0.6975 | 0.6578 | 0.7908 | 36.02 |
| B | 109 | 6 / 1596 | 0.6288 | 0.5556 | 0.7636 | 36.08 |
| A | 109 | 13 / 3458 | 0.5706 | 0.5473 | 0.7657 | 26.37 |
| A | 149 | 7 / 1862 | 0.5733 | 0.5603 | 0.7699 | 26.67 |
| B | 149 | 6 / 1596 | 0.6578 | 0.5995 | 0.7490 | 36.02 |

Condition A mean Test Macro-F1 was **0.5586** (sample SD 0.0156 across five seeds). Condition B was **0.6346** (sample SD 0.0553). The five paired B−A differences were 0.1078, 0.1066, 0.1180, 0.0083 and 0.0392 in seed order; their mean was **0.0760** and sample SD 0.0491. These are descriptive seed summaries, not Test-sample uncertainty or a significance test. The published 0.57/0.67 values are reference points only; the original checkpoint, search and environment are not established as identical.

## Integrity, runtime and failure history

The [independent post-run audit](../reports/stage5_results.json) passed: ten complete attempt records, 150 Validation evaluations, 15 complete Train case-order records per run, equal A/B case orders within each paired seed, verified earliest-best Validation selection, unchanged selected checkpoint weight hashes, and 4,780 Test predictions with unique stable row IDs, correct original-row/label alignment, finite logits and independently recomputed metrics. The frozen Test rule accepted all inputs without overflow; no Test cases were omitted by an incomplete batch. Original texts and labels were not changed.

Total measured training time was **18,811.539 seconds (5 h 13 min 31.5 s)**, below the user-selected 24-hour local GPU training ceiling. This is the sum of ten fit durations, distinct from short synthetic timing extrapolations and final inference time. The largest recorded PyTorch peak reserved GPU memory during the ten fits was approximately 3.36 GiB; this is not a continuous whole-system VRAM maximum. All ten fits completed on their first attempts with no training failure, resume, exclusion or scientific protocol change.

The original final-evaluation entry point failed before producing any Test predictions: it passed already converted integer labels to a strict dataset label converter. All ten Validation selections had already been locked. A separate [one-line corrected evaluator](../scripts/evaluate_final_corrected.py) passed the original CSV Boolean strings to that converter while retaining integer labels for reference checks. A synthetic label check passed before the corrected final run. Training, selected checkpoints and protocol settings were unchanged; all ten final Test evaluations then completed and passed the independent audit. [Correction provenance](../reports/stage5_evaluation_correction.json) records both evaluator hashes, the first failure and the effect boundary. The original failure log and full run artifacts remain under ignored artifacts/experiments/cc-fixed-v1.0.1/.

## Recheck and interpretation boundary

From the project root, with the recorded GPU environment and local artifacts present, rerun the read-only consistency audit:

    .\.venv-gpu\Scripts\python.exe -B scripts/audit_stage5_results.py

This audit will fail if existing public results differ; the final evaluator intentionally refuses to overwrite existing predictions. Run directories retain configurations, case-order hashes, Trainer histories, optimizer/scheduler checkpoints, selected weights, raw per-row prediction IDs and logits, metrics and attempt records. Large artifacts and raw Test input text are excluded from Git.

The next stage can compare the run variation and published reference more carefully, inspect errors without tuning this frozen experiment, and state the limits of the released dataset. A positive paired predictive difference does not establish a causal personality effect, participant-level split independence or a model of human inference.
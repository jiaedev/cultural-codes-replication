# Experimental protocol

**Fixed-settings design v1.0.1: executed on the verified local GPU.** The scientific choices below define one limited replication. Ten planned runs and final Test evaluations are complete; [the execution report](stage5_training_evaluation.md) records results, integrity checks, the attention reload discrepancy and an evaluation-only correction.

The executable source of truth is [experimental_protocol.v1.json](../configs/experimental_protocol.v1.json). [Resolved arguments](../reports/resolved_protocol_arguments.json) expose installed Trainer defaults for all ten run identities on CPU; they are not proof of a CUDA environment. The [execution plan](../reports/experimental_run_plan.json) fixes IDs, paired order and artifact locations.

## 1. Question, scope and reproduction claim

Does adding giver and guesser personality information improve prediction of intentional guessing success on the released Cultural Codes examples?

Use the official `correct_guess_task` split/rows/labels unchanged: A = `base_text`, B = `personality_only`. Preserve missing profiles, duplicates and conflicting labels. Primary metric is two-class Macro-F1; five paired seeds produce ten final training runs. There are no cleaning conditions or new analysis arms.

This is **fixed-settings limited replication**, not reproduction of the paper's full hyperparameter search or exact original checkpoints. Appendix E searches rates {1e-4, 5e-4, 1e-5, 5e-5, 1e-6, 5e-6} and epochs 1-15. Public code fixes 5e-6, 25 epochs, batch size 32. The recommendation uses the code's rate, which is in the published set, and the paper's maximum of 15 epochs. Every epoch is a Validation checkpoint candidate. This limits the main plan to ten runs rather than adding a separate rate search and avoids selecting a favorable configuration after results appear.

## 2. Data, model and initialization

Split sizes and SHA-256 are in the configuration and [source manifest](../sources/manifest.json): Train 8,497, Validation 544, Test 478. Labels map exactly `False -> 0`, `True -> 1`. No weighting, resampling, relabeling, imputation or row removal is used.

The executed model/tokenizer is `FacebookAI/roberta-base` at revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`. The paper names base RoBERTa; public code names `roberta-base-cased`, whose identity remains unresolved. Executability is not evidence of exact original-model identity. [Methods mapping](methods_mapping.md) records that boundary.

Use `RobertaForSequenceClassification`, a two-output head and single-label cross-entropy, with all parameters trainable. Set hidden, attention and classifier dropout to 0.1. Retain source architecture settings, GELU and LayerNorm epsilon 1e-5. New linear head weights use the source initializer's normal distribution (mean 0, SD 0.02), with zero biases. The specified dtype is float32 and the specified attention implementation is eager. The [execution record](stage5_training_evaluation.md#attention-on-model-reload) documents the backend selected by the actual reload path.

Before any run, verify the fixed source safetensors file against the [official revision metadata](../reports/protocol_model_metadata.json): 498,818,054 bytes, SHA-256 `5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81`. The download was deferred until local GPU preparation. The executed runs used the verified weights.

For each seed, create one CPU initial classification snapshot from those pretrained bytes plus the seeded random head. Save it once; A/B load that same snapshot in fresh processes. Verify full parameter/head hashes before optimizer creation. The ten research fits used the full-model procedure. The matched initial snapshot hashes are in the execution provenance.

## 3. Fixed training settings and rationale

| Setting | Executed value | Reason / departure |
| --- | --- | --- |
| Learning rate / duration | 5e-6; 15 full epochs; no hyperparameter search | Public-code rate within Appendix E; maximum follows paper range rather than code's 25. |
| Optimizer | torch AdamW; betas (0.9,0.999), epsilon 1e-8 | Explicit common optimizer settings; not inferred as the authors' exact environment. |
| Weight decay | 0 | Matches the inspected script's effective unconfigured decay; no new regularization search. |
| Optimizer execution flags | foreach=false, fused=false, amsgrad=false, maximize=false, capturable=false, differentiable=false | Explicit implementation rather than hardware-dependent foreach/fused selection. |
| Scheduler / warmup | Linear to zero over 3,990 updates; warmup_steps=0, warmup_ratio=0 | Pins the compatible Trainer behavior rather than leaving it implicit. |
| Gradient clipping | Global norm 1.0 | Explicit Trainer-compatible setting; nonfinite loss/gradients stop execution. |
| Micro-batch / accumulation | 8 / 4, one process | Nominal effective batch 32 with lower activation-memory demand; not numerically identical to physical batch 32. |
| Final epoch update | 17 examples | All 8,497 rows retained: 265 nominal batches of 32 plus the last 17. |
| Precision | FP32; fp16=false, bf16=false, tf32=false; highest float32 matmul precision | Avoids an unvalidated mixed-precision choice; may cost throughput/memory. |
| Train/eval dropping | False / False | Keeps every case; differs from the public training drop flag. |
| Loss | Unweighted single-label cross-entropy; label smoothing 0 | Released binary target; no class balancing or resampling. |
| Length grouping / workers | group_by_length=false; workers=0; pin_memory=false | Input length must not change A/B example order; minimizes worker-state uncertainty. |
| Compilation / gradient checkpointing | Both off | Excludes additional unverified execution paths. |

[PyTorch AdamW](https://docs.pytorch.org/docs/2.9/generated/torch.optim.AdamW.html) and [Trainer 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer) document the relevant interfaces. The separate [protocol bridge](../src/cultural_codes/protocol.py) uses explicit optimizer kwargs while preserving the earlier adapter. The resolved JSON records other library defaults, including checkpoint/logging/reporting options.

### Accumulation normalization amendment (v1.0.1)

The independent review found that v1.0.0 inherited a mean of micro-batch means: in the final [8,8,1] group, the single last case received eight times the coefficient of each other case in that update. Version 1.0.1 uses each micro-batch's **summed cross-entropy divided by the total number of examples in its accumulation group**. The last update's denominator is 17, not three micro-batches or the nominal 32. Evaluation uses the current evaluation batch's example count. Missing counts fail explicitly.

The bridge supplies `compute_loss_func` to Trainer 4.57.1, which counts group labels and avoids a second division by the accumulation-step count. Gradient clipping remains once per completed group, before the optimizer step; the scheduler advances once per optimizer update. This does not claim numerical equivalence to a physical batch with dropout enabled.

No research runs existed under v1.0.0. Its configuration, implementation and generated evidence are preserved byte for byte in the [historical manifest](../reports/history/cc-fixed-v1.0.0/manifest.json). New run IDs and paths use v1.0.1. Seeds, rate, epoch budget, rows, selection rules and planned update counts are unchanged. [Regression evidence](../reports/accumulation_validation.json) tests equal and unequal groups, evaluation averaging, clipping, and recovery in separate CPU processes. GPU validation and the separately recorded local activation have since passed.

## 4. Paired seeds and randomness

Seeds are **13, 37, 73, 109, 149**. They are distinct predeclared identifiers, not values believed to perform well. Pair A/B at each seed; use the same pretrained bytes, initial snapshot, epoch budget, optimizer/scheduler, case order, selection and evaluation rules.

Trainer seed/data_seed are both the run seed. The installed deterministic helper seeds Python, NumPy, PyTorch and CUDA generators; deterministic algorithms are required, cuDNN benchmarking is disabled, and TF32 is disabled. The inspected helper sets CUDA_LAUNCH_BLOCKING=1 and CUBLAS_WORKSPACE_CONFIG=:16:8. Unsupported deterministic operators are technical stop conditions, not reasons for silent fallback.

Accelerate's seedable sampler is enabled explicitly. With data_seed and epoch, its generator uses seed+epoch independently of model RNG consumption. Two synthetic epochs were verified for every pair even after deliberately differing global RNG consumption. Preserve epoch-order hashes in main runs.

Same seeds do not guarantee identical dropout paths, arithmetic, trajectories across A/B input shapes, or reproducibility across hardware/library versions. Dynamic padding and longer B inputs change tensor shapes and computation; all uncertainty is retained in interpretation.

## 5. Inputs, Validation and checkpoint selection

Keep source strings unchanged. Pin the same fast tokenizer/revision for A/B; maximum length is **512 including special tokens**, right truncation and right padding. Pad dynamically to the longest example in each batch, with no multiple-of padding. Maximum length is a ceiling, not a requirement to pad every batch to 512.

Existing Train/Validation audits show a maximum of 197 and no information loss at 512. A protocol guard checks untruncated lengths first and rejects overflow rather than silently truncating. Apply it to all intended inputs under the frozen rule; Test lengths must not choose the rule.

Evaluate and save at every epoch, using Validation only. Maximize exact key **`eval_validation_macro_f1`**. Use strict greater comparison; an exactly equal floating score retains the earliest epoch. No early stopping: run all 15 epochs unless a technical failure occurs. Keep at most two complete epoch checkpoints (best and latest), with optimizer/scheduler/RNG state for recovery; export the selected model and a checkpoint-hash/selection manifest. Missing/nonfinite selection metrics fail clearly.

The development builder has no Test dataset argument. The [research runner](../scripts/train_research.py) uses Train/Validation only, for model inputs and selection. Protocol loading hashes the bytes of every split, including Test, for integrity. The [corrected final evaluator](../scripts/evaluate_final_corrected.py) locks all ten selections before parsing or tokenizing Test rows.

## 6. Final Test evaluation and reporting

First complete all ten fits and lock their Validation-selected checkpoint manifests. Freeze the protocol/config/environment activation hashes. Then evaluate **each of the ten selected models** on the same 478 Test rows through a separate final-evaluation path. This means ten model evaluations, not one total Test call. Do not reselect epochs, seeds or settings from their Test scores.

Audit lengths with the frozen rule before scoring. If any Test input overflows, stop all Test scoring and record the integrity issue. Do not shorten/selectively drop cases or silently change token rules. Any amendment requires a new version, explicit disclosure of Test-length exposure and treatment of previous runs; it cannot be concealed as the original primary comparison.

Use sequential, complete batches and retain source row order. Stable case ID is SHA-256 of `split:source_file_sha256:one_based_data_row`, so duplicate-content rows remain distinct and A/B IDs align. Assert 478 unique IDs, prediction/reference count 478, exact ordered IDs/labels and checkpoint/config hashes. Save row ordinal, export index, input hash, labels, predictions and both logits; raw Test text need not be included.

Primary score: `f1_score(labels=[0,1], average="macro", zero_division=0)` after two-logit argmax; exact logit ties choose class 0. Accuracy, per-class precision/recall/F1/support and a 2x2 confusion matrix are auxiliary. No Test predictions were generated in this stage.

Report every seed/condition score, condition arithmetic means and sample SD (ddof=1), all five paired differences B−A, and their mean/sample SD. Compare descriptively with paper Table 5 (0.57/0.67), acknowledging fixed settings, candidate identity and implementation departures. Seed variation is not Test-sampling uncertainty. No significance test or bootstrap interval is added under unsupported participant/game independence.

## 7. Run plan, failures and computational scope

The [machine-readable plan](../reports/experimental_run_plan.json) specifies ten IDs and paths. Alternate A/B order within successive seed pairs to reduce systematic scheduling order; all scientific settings remain common. Each run is a fresh process and records config/environment/source/initialization hashes, case-order hashes, logs, selected checkpoint, predictions, metrics and attempt history.

| Planned quantity | Count |
| --- | ---: |
| Hyperparameter-search training runs | 0 |
| Paired initialization snapshots | 5 |
| Final training runs | 10 |
| Epochs / optimizer updates per run | 15 / 3,990 |
| Optimizer updates across ten runs | 39,900 |
| Training example presentations | 1,274,550 |
| Validation calls / example evaluations | 150 / 81,600 |
| Final Test calls / example evaluations | 10 / 4,780 |

These are arithmetic planning quantities, confirmed against installed Trainer step calculation, not measured wall time. The structural model has 124,647,170 parameters (counted on a meta device without full-model execution). FP32 parameters plus two Adam moment arrays imply roughly 12 bytes/parameter per complete checkpoint before overhead. Retaining two checkpoints per run, ten selected exports, five initialization snapshots and the source weights suggests roughly 38 GB of model/state files before filesystem/log overhead; reserve at least 50 GiB. This is a storage estimate, not GPU-memory measurement.

No credible end-to-end duration or monetary cost is available before GPU validation. A short synthetic block can characterize execution/memory but is not a measured full training time; any later extrapolation must state its sequence-length/padding/IO assumptions and uncertainty. No GPU service or paid budget is committed here.

Normal finite low performance is valid and retained. For a transient interruption, allow one recovery attempt at the same seed/configuration: resume the latest complete epoch checkpoint, including optimizer/scheduler/RNG/data-order state; otherwise restart from the original initial snapshot within that limit. Retain all attempt records. Ignore incomplete checkpoint writes and never replace a failed seed with a favorable one. The current runner requires a complete checkpoint once a start has been recorded. The fallback restart from initialization is specified but unavailable through that CLI. All ten reported fits completed on their first attempts.

OOM, nonfinite loss/gradients, hash mismatch or unsupported deterministic operators stop the plan. Do not silently change batch, precision, seeds or inputs. A scientific-setting change creates a new protocol version; retain old attempts as superseded and do not pool them into the new primary ten-run comparison. Report missing runs; the five-pair primary summary is incomplete until the prescribed runs are valid.

## 8. Execution readiness and activation gates

The active target is this PC’s single RTX 5080 Laptop GPU on Windows x64, CPython 3.11.17 and the [isolated GPU lock](../environment/requirements-win-gpu.lock.txt). The prior Linux recommendation and Windows DLL block are retained as historical context. A new CUDA wheel imported and passed a tensor operation without changing the security policy.

Before activation, require:

1. A selected GPU location and explicit time/cost ceiling.
2. A recorded GPU lock and full-size pretrained synthetic-only preflight: FP32, eager attention, micro-batch 8, length 512, accumulation 4, two optimizer steps and evaluation batch 8. Record device/driver/runtime, finite loss/gradients, partial-batch coverage, save/reload, peak allocated/reserved VRAM and timing conditions. If the specified configuration fails, amend the design before research runs.
3. An implemented/audited research training and final-evaluation runner with source/row/initialization/checkpoint checks and no Test development path.
4. Verified initial snapshots, Train/Validation input integrity/no-overflow checks and an immutable activation manifest tying protocol/config/environment hashes.

Test length checks remain a separate pre-scoring gate after all ten selections (§6).

The [CPU protocol check](../reports/protocol_validation.json), [loss regression](../reports/accumulation_validation.json), [full-size GPU preflight](../reports/full_model_gpu_preflight.json) and [short synthetic timing blocks](../reports/gpu_synthetic_benchmark.json) match the active protocol hash. The user selected a 24-hour total GPU ceiling and no external service cost in [execution_budget.json](../configs/execution_budget.json). The ignored activation manifest freezes the actual code hashes, GPU environment, pinned bytes, initial snapshots, Train/Validation coverage and run order. It records the dirty Git tree with per-file hashes. Test length inspection took place after all ten Validation selections were locked.

## 9. Reproduce the configuration check

```powershell
.\.venv\Scripts\python.exe scripts/validate_protocol.py
.\.venv\Scripts\python.exe scripts/validate_accumulation.py
```

The first command resolves all ten run arguments, hashes recorded research files without training/tokenizing their contents, checks five paired tiny-model initializations and two epoch orders, executes two optimizer updates on 35 invented examples, verifies exact-score ties and overflow rejection, and writes the plan/resolved/validation JSON. Technical overrides are explicit in the report. It neither downloads pretrained weights nor provides a main-training CLI.

The second command compares accumulated gradients with a single-batch mean CE reference for [8,8,8,8], [8,8,1] and [3], with dropout disabled only for that comparison. It also runs a 49-case, three-epoch tiny-model fixture with dropout enabled and a deliberately wrong initial bias to exercise norm-1 clipping. A separate process resumes the first epoch checkpoint and checks final parameters, optimizer/scheduler state, case order, complete evaluation and earliest-tie reload. Constant selection scores are a technical fixture, not model performance. Outputs go to a unique ignored artifact directory and `reports/accumulation_validation.json`.

Underlying data and interpretation limits remain in [methods mapping](methods_mapping.md) and [validation](validation.md).

## 10. Executed protocol record

All ten planned local GPU fits and final Test evaluations completed. The configuration file stayed fixed after activation. Its eager attention setting was not forwarded on model reload in either condition. The first final-evaluation entry point failed before prediction because it passed preconverted integer labels to the strict CSV label converter; a separately hashed one-line evaluation-only correction was used after all ten Validation selections were locked. The [execution report](stage5_training_evaluation.md), [independent result audit](../reports/stage5_results.json) and [correction provenance](../reports/stage5_evaluation_correction.json) contain actual outputs and the failure boundary. This record does not alter the frozen protocol.

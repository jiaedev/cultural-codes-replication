# Validation

## 1. Scope and summary

The recorded checks establish two things: the released A/B fields describe the same cases with the same base content and labels, and a separate adapter can execute the intended data/metric/selection path on a small synthetic CPU model. They do not establish a personality-related performance improvement.

**Evidence date: 2026-11-17.** The retained dataset audit is timestamped 05:35:38 UTC, tokenizer audit 05:27:43 UTC and CPU smoke 05:36:55 UTC. Exact timestamps, versions, hashes and settings remain in the linked JSON. This document consolidates those existing findings; consolidation did not rerun the analyses or change their dates/results.

Dataset validation is complete; implementation compatibility is verified for one CPU process. The [recommended experimental protocol](experimental_protocol.md) now specifies the design; execution freeze remains pending.

## 2. Data validation

### Sources, schema and labels

Official artifacts use commit `9bf4550e681f7a42ac406439b00b0c717f59f13c`; the paper source is the ACL publication. All 16 recorded source files passed SHA-256 verification, and repository file bytes matched pinned Git blob IDs during acquisition. [Source provenance](../sources/README.md) and the [unchanged manifest](../sources/manifest.json) record URLs, acquisition timestamps and hashes.

| Split | Rows | True | False |
| --- | ---: | ---: | ---: |
| Train | 8,497 | 6,672 | 1,825 |
| Validation | 544 | 414 | 130 |
| Test | 478 | 369 | 109 |
| Total | 9,519 | 7,455 | 2,064 |

All splits have eight columns: blank-header export index, `base_text`, `leaning_only`, `event_only`, `demo_only`, `personality_only`, `all_text`, and `output`. Raw CSV values are strings; the export index restarts at zero and is not an event/participant identifier. The observed schema corrects the upstream README's inconsistent column count.

Only exact `True`/`False` labels occur. No structurally absent fields, empty cells, whitespace-only cells or whole-cell sentinel values were found. Embedded `None` is a separate form of unavailable information, described below. The standard-library audit first ran with CPython 3.13.16; the retained later audit uses CPython 3.11.17.

### Paired inputs and profile availability

The observed serialization is:

```text
A: remaining: <list>, rationale: <giver rationale>, target: <one target>, hint: <clue>
B: GIVER: [<profile>], GUESSER: [<profile>], <exact A string>
```

All 9,519 rows contain both input fields and one shared output. Full grammar matching, unambiguous separators, safe list parsing, role-profile names/order/numeric checks, independent parsing of both base sections, ordered remaining-list equality, rationale/target/hint equality and exact suffix comparison all passed. There were zero non-personality content changes or parsing exceptions. Targets occur in their remaining lists, with no empty rationale/target/hint or multi-whitespace-word hints. These are release-serialization checks, not reconstruction of original game states or proof of label correctness.

[Train-only examples](../reports/train_examples.md) show full paired strings for data rows 1 and 35. A single output column ensures label alignment; separate datasets were not joined.

| Row-level personality availability | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Both profiles available | 4,493 | 391 | 378 |
| Giver missing only | 2,132 | 0 | 0 |
| Guesser missing only | 1,872 | 153 | 100 |
| Both missing | 0 | 0 | 0 |
| At least one missing | 4,004 | 153 | 100 |

Counts refer to **rows, not unique people**. Each missing profile is `[None]`, not individually missing trait fields; no partial five-trait profiles were observed. The same embedded `None` row counts occur in `leaning_only`, `demo_only` and `all_text`, while none occur in base/event/index/output fields. No other tested sentinel was found. No missingness mechanism or participant identity was inferred; zero-valued traits were not treated as missing. Score-construction uncertainty is detailed in [methods mapping](methods_mapping.md).

### Duplicates, conflicts and split overlap

Duplicate groups are keys occurring more than once; excess rows count each group's size minus one. All observed duplicate groups have two rows.

| Within-split key | Train groups / excess | Validation groups / excess | Test groups / excess |
| --- | ---: | ---: | ---: |
| All columns including export index | 0 / 0 | 0 / 0 | 0 / 0 |
| Semantic columns excluding index | 2 / 2 | 2 / 2 | 0 / 0 |
| Exact `base_text` | 17 / 17 | 4 / 4 | 0 / 0 |
| Exact `personality_only` | 4 / 4 | 4 / 4 | 0 / 0 |

Each condition also has two conflicting-label input groups in Train and two in Validation, covering four rows per split; Test has none. The same row pairs conflict under A and B, so these are four combined groups covering eight rows, not eight separate groups. [Dataset JSON](../reports/dataset_validation.json) records row/label locations and input hashes.

These are confirmed conflicts conditional on the released inputs, **not confirmed annotation errors**. Different outcomes or omitted context may account for identical inputs. The origin of duplicates/conflicts cannot be established without original logs and IDs. No rows were deleted, imputed or relabeled.

All three split pairs have zero shared exact full rows (with or without index), `base_text` or `personality_only`. Accordingly there are no cross-split shared-input label conflicts under those keys. No fuzzy/semantic overlap search was performed. Exact text non-overlap does not establish participant/game independence, and trait profiles are not substitute identifiers.

`all.csv` contains 9,519 rows and matches the three splits' semantic-column multiset exactly, with zero extra rows in either direction. It is a consistency reference, not another experimental split.

## 3. Implementation and evaluation validation

Pinned upstream code is unchanged; the separate adapter is mapped in [methods](methods_mapping.md).

| Check | Recorded result |
| --- | --- |
| Labels | Exact strings/native Booleans map `False -> 0`, `True -> 1`; unexpected values are rejected. [Label audit](../reports/label_conversion.json) preserves every split distribution and paired alignment. Test access is label integrity only. |
| Metrics | Finite two-logit argmax, explicit classes [0,1], unweighted Macro-F1 and zero_division=0; accuracy is separately named. Five known-answer fixtures passed. For example, positive-only predictions on balanced labels give F1 1/3 and accuracy 1/2. Full arrays/expectations are in [smoke JSON](../reports/smoke_cpu.json). |
| Validation selection | Builder takes Train/Validation only. Actual `eval_validation_macro_f1` matches `metric_for_best_model` with `greater_is_better=True`; wrong/missing keys fail explicitly. Callback selection/reload and a controlled non-monotonic best-metric fixture passed. |
| Final evaluation | Separate `final_evaluate` clones arguments with in-training evaluation/save/selection disabled and checks full prediction/reference counts. Tested on synthetic held-out cases only. |

Evaluation coverage was reproduced specifically with **Transformers 4.57.1, Accelerate 1.12.0, one CPU process, five synthetic cases, batch size 2 and `dataloader_drop_last=True`**:

| Path | Batch sizes | Cases |
| --- | --- | ---: |
| Stock Trainer evaluation / prediction | [2,2] | 4/5 |
| Adapter evaluation / final prediction | [2,2,1] | 5/5 |
| Adapter training with dropping enabled | [2,2] | 4/5 |

The adapter sets evaluation/prediction dropping to false independently of training and observed indices are exactly [0,1,2,3,4]. This is not a universal claim about Trainer or other versions/configurations. The adapter explicitly rejects multiple processes.

## 4. Tokenization and input preservation

The candidate's [pinned tokenizer/config files](../reports/tokenizer_manifest.json) provide a fast RoBERTa tokenizer with two special tokens per single input, right padding/truncation and a 512-token total capacity. Config has 514 position embeddings and pad ID 1, consistent with that bound.

| Split / condition | Median tokens | P95 | Maximum |
| --- | ---: | ---: | ---: |
| Train A | 98 | 108 | 149 |
| Train B | 162 | 186 | 197 |
| Validation A | 98 | 108 | 118 |
| Validation B | 173 | 186 | 197 |

Lengths include special tokens. B adds 46 or 80 tokens in observed profiles. At the 512 bound, zero token sequences changed, every full input string round-tripped exactly, and offset checks found zero cuts to remaining/rationale/target/hint content in either condition.

This verifies content preservation, not equality of every base token ID: byte-level tokenization can differ at the prefix/space boundary. A shorter bound could cut B's late base fields earlier; left truncation could instead remove profiles/remaining words. The bound is diagnostic, not a frozen research setting. **Test was not tokenized.**

[Tokenization JSON](../reports/tokenization_audit.json) retains full distributions. Character and whitespace-word statistics remain separately in the dataset JSON and are not described as model tokens.

## 5. Synthetic execution

The CPU smoke uses invented strings and a randomly initialized tiny RoBERTa: one layer, hidden size 32, four heads, intermediate size 64 and 1,634,722 parameters. Both input forms passed tokenization/forward/loss checks; two optimizer steps, Validation metric/selection, checkpoint saving, reload and complete synthetic final prediction passed.

Maximum absolute logits difference after reload was **0.0**, and five synthetic final predictions were returned. A filesystem timestamp warning triggered numerical checkpoint ordering; selection/reload assertions still passed. All fixture settings and code hashes are retained in [smoke JSON](../reports/smoke_cpu.json) and the [smoke script](../scripts/smoke_test.py). They are technical settings, not adopted main-experiment defaults.

No pretrained weights, research-data training, A/B performance comparison or research Test predictions were used. Tiny-model loss, metrics and runtime do not validate full-model performance, GPU memory or training time.

## 6. Unverified questions

- Original guesses, participant/game/turn IDs and the release-construction pipeline are unavailable. Labels cannot be regenerated and split independence cannot be checked beyond exact text.
- Shared contexts support target-level expansion but do not recover unique turns. For example, Train rows 17/18 share clue `baseball` and context but differ in target/output (`pitch`/True, `field`/False). Candidate context-group counts remain in the dataset JSON and are not actual turn counts.
- Raw survey responses/scoring and raw-to-normalized rationale pairs are absent. Personality scoring and normalization remain unverified.
- An executable official model/tokenizer candidate is pinned, but exact original-checkpoint identity and main-experiment adoption remain unresolved; see [methods](methods_mapping.md).
- CPU single-process execution is verified. CUDA import was blocked on the tested host; full-size GPU execution, resource use, precision and distributed behavior remain unverified. Necessary setup details are in [environment](../environment/README.md).
- Technical checks do not establish predictive improvement, causal personality effects or equivalence to human inference.

## 7. Reproduction and evidence

Follow [environment setup](../environment/README.md) to install the pinned stack and restore omitted files from manifests. From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/validate_dataset.py
.\.venv\Scripts\python.exe scripts/audit_labels.py
.\.venv\Scripts\python.exe scripts/audit_tokenization.py
.\.venv\Scripts\python.exe scripts/smoke_test.py --device cpu
```

The [source manifest](../sources/manifest.json), [environment lock](../environment/requirements-win-cpu.lock.txt), [environment summary](../reports/environment_summary.json), [dependency artifacts](../reports/dependency_artifacts.json) and [public validation summary](../reports/research_validation_summary.json) preserve provenance and recorded evidence. The publication summaries derive from unchanged original records; local operational diagnostics are excluded from the public artifact set. The [recommended protocol](experimental_protocol.md) separates specified scientific choices and pending activation gates from executed checks.

## 8. Protocol configuration checks

A separate CPU check on 2026-11-17 resolved the recommended configuration through the installed Trainer. It was rerun for the v1.0.1 correction; [protocol_validation.json](../reports/protocol_validation.json) records the exact timestamp/configuration/code hashes. This is additional synthetic evidence and does not replace the original reports above.

For seeds 13, 37, 73, 109 and 149, A/B loaded the same tiny random initialization snapshot. Entire parameter/head hashes matched within each pair, while different seeds produced different heads. Actual seedable DataLoaders preserved all 35 invented examples and matched two epoch orders across conditions despite deliberately differing global RNG consumption. The original data were only hashed, not tokenized or trained in this check.

One tiny synthetic model completed two optimizer updates with the specified AdamW/accumulation settings (technical override: one epoch on 35 invented cases, CPU, tiny dimensions and ignored output paths). Exact-score ties retained the earliest best, a partial evaluation batch of five was included, and an invented overlength input was rejected before silent truncation. Stable-ID generation produced 478 unique IDs without reading or predicting Test cases.

The installed Trainer's step calculation agrees with 266 updates/epoch and 3,990 per planned run. A full-size configuration was instantiated on a meta device only to count 124,647,170 parameters; no full-model forward/backward, pretrained download, GPU memory test or wall-time estimate was validated. Research activation remains explicitly blocked. The earlier adapter and CPU lock are unchanged.

[Resolved arguments](../reports/resolved_protocol_arguments.json) include library defaults for the ten planned run identities on CPU; [execution plan](../reports/experimental_run_plan.json) records the arithmetic budget. The [model metadata](../reports/protocol_model_metadata.json) verifies the pinned source weight file's advertised SHA-256/size without downloading it. GPU environment and research-runner checks remain required before Stage 5.

## 9. Accumulation normalization and recovery regression

Independent review reproduced a v1.0.0 error in the final accumulation group [8,8,1]: averaging three micro-batch means assigned the single last case eight times the coefficient of another case in the same update. The earlier 35-case fixture ended in a single [3] micro-batch and could not expose unequal sizes within a group. No research data had been trained. The [pre-fix manifest](../reports/history/cc-fixed-v1.0.0/manifest.json) preserves the old configuration, code and reports unchanged.

Version 1.0.1 supplies summed CE divided by the actual accumulated example count through Trainer's custom loss interface. A missing count raises an error. The [regression script](../scripts/validate_accumulation.py) and [machine-readable results](../reports/accumulation_validation.json) record the following executed CPU checks:

| Check | Result |
| --- | --- |
| Groups [8,8,8,8], [8,8,1], [3] | Counts 32, 17, 3; all 25 parameter-gradient tensors match a single-batch mean CE reference within rtol=2e-5, atol=2e-7. Maximum absolute difference across fixtures is 2.24e-8. Dropout is disabled only for this analytic comparison. |
| Negative control | Restoring the pre-fix mean-of-means path produces a maximum gradient discrepancy of about 0.27451 for [8,8,1]; the regression detects the original error. |
| Complete 49-case epochs | Each case appears once per epoch; three epochs produce six optimizer updates. Groups carry denominator 32 or 17, with no epoch-boundary carry-over. |
| Clipping and schedule | Norm-1 clipping actually activates, once after each full accumulated group and before optimization. The linear schedule advances once per update and reaches zero at update six. |
| Evaluation | Nine invented cases yield batches [8,1]; evaluation loss matches the full-case mean CE within 2e-6. |
| Selection and reload | Constant 0.8 selection scores retain and reload the first epoch's checkpoint. These scores are injected technical fixtures, not model results. |
| Resume in a fresh process | Starting from the first complete epoch checkpoint preserves remaining case orders and steps; final parameter maximum difference is 0.0 and optimizer/scheduler states match continuous execution exactly. |

Integration fixtures use random tiny RoBERTa (vocabulary 64), invented seven-token inputs, seed 13, dropout 0.1 and deliberately wrong initial classification biases [-4,4] on class-0 labels to exercise clipping. The three-epoch duration, reduced architecture, biases, constant metric and CPU execution are explicit test overrides, not amendments to the research settings. No research text is tokenized or trained; split files are only hashed. No Test predictions, pretrained download, full-size model execution or GPU validation occurs.

Run `.\.venv\Scripts\python.exe scripts/validate_accumulation.py` from the project root. Each invocation uses a new ignored artifact directory; results include the protocol/code hashes and exact versions. The planned research counts remain 266 updates per epoch, 3,990 per run and 39,900 across ten runs. This regression establishes CPU behavior for the correction; execution activation still requires the protocol's GPU and runner gates.

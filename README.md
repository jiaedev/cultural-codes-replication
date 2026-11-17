# Cultural Codes replication

Replication of the target-level success-prediction task in Shaikh et al., [*Modeling Cross-Cultural Pragmatic Inference with Codenames Duet*](https://aclanthology.org/2023.findings-acl.410/) (Findings of ACL, 2023).

**Research question:** Does adding giver and guesser personality information improve prediction of intentional guessing success on the released Cultural Codes examples?

**Status:** Dataset validation and CPU, single-process implementation checks are complete (Stages 2-3). The fixed-settings protocol design is specified; execution freeze and GPU/runner gates remain pending. No research-data training, A/B performance comparison, Test predictions, or pretrained-weight download has been performed. GPU execution and full-size model resource requirements remain unverified.

## Comparison design

The project uses the official preprocessed `correct_guess_task` splits without cleaning or repartitioning. Each row concerns one intended target in a clue context, with a binary success label.

| Component | Scope |
| --- | --- |
| Condition A | `base_text`: remaining words, giver rationale, target, and clue |
| Condition B | `personality_only`: giver/guesser personality profiles, followed by the same base input |
| Model candidate | `FacebookAI/roberta-base`, [pinned revision and configuration](configs/model_candidate.json) |
| Primary metric | Macro-F1 across failure and success |
| Planned repetition | Five paired seeds; fixed settings specified in the [protocol](docs/experimental_protocol.md) |

The classifier receives the intended target and giver rationale, which are hidden from the human guesser. This is success prediction with privileged information, rather than a model of the guesser's information state.

## Verified so far

- **Paired data:** All 9,519 examples share the same label and exact base content across A/B. Train/Validation/Test contain 8,497/544/478 rows. Missing profiles, duplicates and conflicting labels were recorded and preserved; no exact input overlap was found across splits.
- **Labels and metrics:** Explicit `False -> 0` / `True -> 1` conversion preserves split distributions. Five known-answer synthetic fixtures verify two-class Macro-F1 and distinguish it from accuracy.
- **Accumulation:** Protocol v1.0.1 normalizes loss by the actual group example count, including [8,8,1]. Synthetic regression checks cover gradients, clipping and checkpoint recovery in separate CPU processes.
- **Selection and coverage:** A separate adapter connects `eval_validation_macro_f1` to checkpoint selection, excludes Test from the development builder, and preserves final incomplete evaluation batches under the tested configuration.
- **Tokenizer and execution:** The pinned tokenizer preserves all Train/Validation inputs at its 512-token capacity (maximum observed: 197, including special tokens). A small randomly initialized RoBERTa completes synthetic forward/loss, two optimizer steps, evaluation, checkpoint selection, save and reload on CPU.

These are data and implementation checks, not evidence of a personality-related performance improvement. Detailed evidence is in [validation](docs/validation.md) and the [public validation summary](reports/research_validation_summary.json).

## Reproduce the checks

Use PowerShell from the repository root with an existing CPython 3.11 interpreter; 3.11.17 was tested.

```powershell
python -m venv .venv
$env:PIP_CACHE_DIR = Join-Path (Get-Location) '.cache\pip'
.\.venv\Scripts\python.exe -m pip install -r environment/requirements-win-cpu.lock.txt
.\.venv\Scripts\python.exe scripts/restore_recorded_files.py
.\.venv\Scripts\python.exe scripts/validate_dataset.py
.\.venv\Scripts\python.exe scripts/audit_labels.py
.\.venv\Scripts\python.exe scripts/audit_tokenization.py
.\.venv\Scripts\python.exe scripts/smoke_test.py --device cpu
.\.venv\Scripts\python.exe scripts/validate_protocol.py
.\.venv\Scripts\python.exe scripts/validate_accumulation.py
```

Check that `python --version` reports Python 3.11 before creating the environment. Restoration retrieves omitted data/tokenizer files from recorded URLs and verifies their hashes without replacing different existing bytes. Later checks use local inputs. The smoke test uses invented examples and random tiny-model weights.

See [environment setup and troubleshooting](environment/README.md) for interpreter selection, offline execution and output details, and [source provenance](sources/README.md) for acquisition and hashes.

## Limits and next steps

Raw guesses, participant/game/turn IDs, and the original data-construction pipeline are unavailable in the inspected release. Labels cannot be regenerated, participant-level split separation cannot be independently checked, and personality score construction remains unverified. The executable model candidate has not been established as the exact original checkpoint.

Any future predictive gain would not establish a causal personality effect or equivalence to human cognitive processes. [Methods mapping](docs/methods_mapping.md) distinguishes the paper, public artifacts and this implementation. The [roadmap](ROADMAP.md) and [recommended experimental protocol](docs/experimental_protocol.md) describe the remaining work.

## Repository guide

- `src/`, `scripts/`: compatibility adapter and reproducible checks.
- `docs/`: methods, validation findings and protocol decisions.
- `environment/`, `configs/`: environment lock and model candidate.
- `sources/`, `reports/`: source provenance and recorded validation evidence.

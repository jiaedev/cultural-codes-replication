# Experimental protocol

**Draft — experimental decisions pending; not yet frozen.**

## Question and comparison

Does adding giver and guesser personality information improve prediction of intentional guessing success on the released Cultural Codes examples?

The working scope is target-level binary classification on the official `correct_guess_task` splits: A = `base_text`, B = `personality_only`, primary metric = Macro-F1, with five paired runs per condition planned. These describe the existing project scope; this draft does not freeze the experiment.

## Decisions to record before training

| Item | Current evidence / options | Decision | Rationale / difference from original |
| --- | --- | --- | --- |
| Data and labels | Preserve released splits, rows and missing-profile strings; adapter maps `False -> 0`, `True -> 1`. | Existing working scope; confirm at protocol freeze. | [Methods](methods_mapping.md) and [validation](validation.md) document source limits and pairing checks. |
| Model and provenance | Candidate `FacebookAI/roberta-base`, revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`; exact original `roberta-base-cased` identity is unverified. | Main-experiment adoption pending. | Record acceptance of the candidate/uncertainty or stronger artifact evidence; tokenizer/config executability is not checkpoint identity. |
| Final execution environment | Validated CPU environment is available; GPU execution and full-size resource use require separate checks. | Pending. | Record the final environment/lock, device and capacity evidence before setting a practical run budget. |
| Learning rate, epochs and search | Paper classification search includes 5e-6 and epochs 1-15; public script fixes 5e-6 and 25 epochs. Fixed settings or a predefined Validation-only search are possible. | Pending. | Explain the chosen reproduction target, departures and search/compute budget. |
| Five paired seeds and randomness | Match A/B initialization and data-order seeds across five runs; original seeds are unavailable. | Seed list and controls pending. | Record Python/NumPy/tensor/data-loader controls and the limits of determinism. |
| Batch, optimizer and scheduler | Batch size, accumulation, optimizer, scheduler and precision affect exposure, numerical behavior and memory. | Pending. | Record effective batch size and implementation differences, rather than inheriting smoke settings. |
| Training `drop_last` | Keeping or dropping an incomplete training batch changes case exposure. Evaluation preservation is independently implemented. | Pending. | Document cases/updates affected and any departure from public code. |
| Maximum length, padding and truncation | All Train/Validation strings fit the candidate's 512-token capacity. Shorter bounds can remove more base content from B. | Final bound, padding policy and truncation direction pending. | Require a common, justified A/B rule and record information-preservation checks. |
| Validation and checkpoint selection | Named Validation Macro-F1 wiring is verified; keep Test outside training/selection. | Research evaluation/save cadence pending. | Specify exact metric name/direction, checkpoint retention and row-coverage checks. |
| Ties and early stopping | Tie-breaking, patience and stopping criteria can change selected checkpoints. | Pending. | State rules before observing run results. |
| Final Test evaluation | Apply the frozen tokenizer/rule, audit coverage without retuning, then use the separate final-evaluation path. | Timing and final-evaluation procedure pending. | Specify full-row prediction/reference checks and retain per-example outputs. |
| Failures and reruns | Crashes, nonfinite results or interrupted runs need a consistent treatment. | Failure definitions, retry limits and reporting pending. | Predefine treatment of incomplete/failed runs and preserve attempt histories to avoid selective reporting. |
| Results and paired A/B differences | Macro-F1 is primary; report each run and paired differences, means/variation and any uncertainty interval. | Aggregation and uncertainty method pending. | Fix summaries and handling of failures before viewing outcomes; distinguish paper reference values from project results. |

## Freeze record

Before training, replace pending entries with explicit choices, rationale and departures from the paper/public code; record the protocol version/date and final configuration references. Synthetic fixture settings are not experimental defaults.

Data limitations and interpretation boundaries are described once in [methods mapping](methods_mapping.md); executed technical checks and their limits are in [validation](validation.md). Stage 4 remains incomplete until the experimental decisions are fixed.

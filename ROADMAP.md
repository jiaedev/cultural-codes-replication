# Roadmap

| Stage | Goal and completion criteria | Status |
| --- | --- | --- |
| 0 - Feasibility | Identify the task, accessible artifacts, reproduction scope and major evidence/compute limits. | Conditional Go. Current basis: [methods and source boundaries](docs/methods_mapping.md). |
| 1 - Research Background | Establish pragmatic inference, common ground, individual differences, task operationalization and prediction/causation distinctions. | Background preparation complete; task interpretation is recorded in [methods mapping](docs/methods_mapping.md). |
| 2 - Dataset Validation | Pin sources; check schema, labels, paired inputs, missingness, duplicates, text overlap and verification limits with executable evidence. | Complete. [Data validation](docs/validation.md#2-data-validation). |
| 3 - Environment & Implementation Compatibility | Pin an executable model/tokenizer candidate and environment; verify labels, metrics, selection, evaluation coverage, tokenization and a synthetic execution path. | Complete for CPU, one process. Original checkpoint identity and GPU/full-model feasibility remain unverified. [Implementation validation](docs/validation.md#3-implementation-and-evaluation-validation). |
| 4 - Experimental Protocol | Freeze paired conditions, five-run seeds, model/settings, Validation-only selection, Test handling and reporting rules; document deviations. | Draft; experimental decisions pending, not frozen. [Protocol](docs/experimental_protocol.md). |
| 5 - Training & Evaluation | Execute the frozen protocol; retain configurations, logs, checkpoints, predictions, coverage checks and failures for all runs. | Not started. |
| 6 - Comparison & Interpretation | Compare A/B and published results with run variation, implementation differences and explicit interpretation limits. | Not started. |
| 7 - Reporting | Present reproducible methods, results, limitations, deviations and artifact instructions. | Not started. |

Stages 2-3 establish data comparability and a technical execution path. They do not provide research performance results. Remaining decisions are in the draft protocol; synthetic smoke settings are not experimental defaults.

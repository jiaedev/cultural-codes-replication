# Roadmap

| Stage | Goal and completion criteria | Status |
| --- | --- | --- |
| 0 - Feasibility | Identify the task, accessible artifacts, reproduction scope and major evidence/compute limits. | Conditional Go. Current basis: [methods and source boundaries](docs/methods_mapping.md). |
| 1 - Research Background | Establish pragmatic inference, common ground, individual differences, task operationalization and prediction/causation distinctions. | Background preparation complete; task interpretation is recorded in [methods mapping](docs/methods_mapping.md). |
| 2 - Dataset Validation | Pin sources; check schema, labels, paired inputs, missingness, duplicates, text overlap and verification limits with executable evidence. | Complete. [Data validation](docs/validation.md#2-data-validation). |
| 3 - Environment & Implementation Compatibility | Pin an executable model/tokenizer candidate and environment; verify labels, metrics, selection, evaluation coverage, tokenization and a synthetic execution path. | Complete for one process: CPU compatibility and local RTX 5080 full-size synthetic CUDA preflight passed. Original checkpoint identity remains unverified. [Implementation validation](docs/validation.md#3-implementation-and-evaluation-validation). |
| 4 - Experimental Protocol | Freeze paired conditions, five-run seeds, model/settings, Validation-only selection, Test handling and reporting rules; document deviations. | Frozen v1.0.1 for the verified local Windows GPU with recorded 24-hour budget, runner code hashes and activation manifest. [Protocol](docs/experimental_protocol.md). |
| 5 - Training & Evaluation | Execute the frozen protocol; retain configurations, logs, checkpoints, predictions, coverage checks and failures for all runs. | Complete: ten fits and ten full-row Test evaluations passed independent audit; the recorded evaluation-only label fix is documented in [results](docs/stage5_training_evaluation.md). |
| 6 - Comparison & Interpretation | Compare A/B and published results with run variation, implementation differences and explicit interpretation limits. | Complete. [Prediction audit, error analysis and interpretation](docs/results_analysis.md). |
| 7 - Reporting | Present reproducible methods, results, limitations, deviations and artifact instructions. | Not started. |

Stages 2-3 establish data comparability and a technical execution path. The fixed ten-run predictive result is recorded separately in the execution report; the completed comparison informs the final reporting stage. Scientific settings are in the protocol, and synthetic smoke settings are not experimental defaults.

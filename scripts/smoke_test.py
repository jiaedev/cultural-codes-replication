"""Synthetic-only compatibility checks. No research examples or pretrained weights."""
import argparse
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from transformers import (AutoTokenizer, DataCollatorWithPadding, RobertaConfig,
                          RobertaForSequenceClassification, TrainingArguments, Trainer, set_seed)
from cultural_codes.data import TextDataset, convert_label, LABEL2ID, ID2LABEL
from cultural_codes.metrics import from_predictions, compute_metrics
from cultural_codes.trainer import build_development_trainer, final_evaluate, SELECTION_METRIC

class ObservedDataset(TextDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = []
    def __getitem__(self, index):
        self.seen.append(index)
        return super().__getitem__(index)

def check_metrics():
    fixtures = [
        ("perfect", [0,1,0,1], [0,1,0,1], 1., 1.),
        ("all_wrong", [0,1], [1,0], 0., 0.),
        ("only_positive_balanced", [0,0,1,1], [1,1,1,1], 1/3, .5),
        ("only_negative_imbalanced", [0,0,0,1], [0,0,0,0], 3/7, .75),
        ("absent_class_still_in_macro", [0,0], [0,0], .5, 1.),
    ]
    result = {}
    for name, y, p, f1, accuracy in fixtures:
        got = from_predictions(y, p)
        assert math.isclose(got["macro_f1"], f1, abs_tol=1e-12), (name, got)
        assert got["accuracy"] == accuracy
        logits = np.array([[2.,-2.] if x == 0 else [-2.,2.] for x in p])
        assert compute_metrics((logits, np.array(y))) == got
        result[name] = dict(expected_macro_f1=f1, expected_accuracy=accuracy, observed=got)
    assert convert_label("False") == 0 and convert_label("True") == 1
    assert convert_label(False) == 0 and convert_label(True) == 1
    for invalid in ("false", "TRUE", "1", "", " False", None, 1, 0):
        try:
            convert_label(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unexpected label accepted: {invalid!r}")
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    torch.set_num_threads(4)
    seed = 314159
    set_seed(seed)
    spec = json.loads((ROOT / "configs/model_candidate.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(ROOT/spec["local_path"], local_files_only=True)
    collator = DataCollatorWithPadding(tokenizer)
    # These invented strings are not copied from any research CSV.
    texts = [f"remaining: ['apple', 'stone'], rationale: synthetic example {i}, target: apple, hint: fruit"
             for i in range(5)]
    labels = ["False", "True", "False", "True", "False"]
    personality_texts = ["GIVER: [None], GUESSER: [None], " + text for text in texts]
    personality = TextDataset(personality_texts, labels, tokenizer, max_length=64)
    train = ObservedDataset(texts, labels, tokenizer, max_length=64)
    assert personality.labels == train.labels
    validation = ObservedDataset(texts, labels, tokenizer, max_length=64)
    holdout = ObservedDataset(texts, labels, tokenizer, max_length=64)
    config = RobertaConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1,
                           num_attention_heads=4, intermediate_size=64, max_position_embeddings=514,
                           num_labels=2, pad_token_id=tokenizer.pad_token_id,
                           bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                           label2id=LABEL2ID, id2label=ID2LABEL)
    model = RobertaForSequenceClassification(config)
    output = ROOT / "artifacts" / f"synthetic_smoke_{args.device}"
    training_args = TrainingArguments(output_dir=str(output), overwrite_output_dir=True,
        max_steps=2, learning_rate=1e-3, seed=seed, data_seed=seed,
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        eval_strategy="steps", eval_steps=1, save_strategy="steps", save_steps=1,
        load_best_model_at_end=True, metric_for_best_model=SELECTION_METRIC,
        greater_is_better=True, dataloader_drop_last=True, dataloader_pin_memory=False,
        dataloader_num_workers=0, save_total_limit=2, logging_strategy="no",
        report_to=[], disable_tqdm=True, use_cpu=args.device == "cpu")
    trainer = build_development_trainer(model, training_args, train, validation, collator)
    assert set(trainer.eval_dataset) == {"validation"}
    # Observe the installed upstream implementation, not merely arithmetic expectations.
    stock = Trainer(model=RobertaForSequenceClassification(config), args=training_args,
                    eval_dataset=validation, data_collator=collator)
    stock_batches = [len(batch["labels"]) for batch in stock.get_eval_dataloader()]
    stock_test_batches = [len(batch["labels"]) for batch in stock.get_test_dataloader(holdout)]
    train_batches = [len(batch["labels"]) for batch in trainer.get_train_dataloader()]
    eval_batches = [len(batch["labels"]) for batch in trainer.get_eval_dataloader(validation)]
    assert train_batches == [2, 2], train_batches
    assert eval_batches == [2, 2, 1], eval_batches
    assert len(trainer.get_test_dataloader(holdout)) == 3
    batch = collator([train[0], train[1]])
    model = trainer.model
    model.eval()
    with torch.no_grad():
        forward = model(**{k:v.to(trainer.args.device) for k,v in batch.items()})
    assert torch.isfinite(forward.loss)
    initial_forward_loss = forward.loss.item()
    personality_batch = collator([personality[0], personality[1]])
    with torch.no_grad():
        personality_forward = model(**{k:v.to(trainer.args.device) for k,v in personality_batch.items()})
    assert torch.isfinite(personality_forward.loss)
    result = trainer.train()
    validation.seen.clear()
    evaluated = trainer.evaluate(metric_key_prefix="eval")
    assert validation.seen == list(range(5)), validation.seen
    assert SELECTION_METRIC in evaluated
    assert not any("test" in key for log in trainer.state.log_history for key in log)
    assert trainer.state.global_step == 2
    assert trainer.state.best_model_checkpoint is not None
    checkpoint_states = list(output.glob("checkpoint-*/trainer_state.json"))
    checkpoint_metrics = [json.loads(p.read_text())["best_metric"] for p in checkpoint_states]
    assert trainer.state.best_metric == max(checkpoint_metrics)
    with open(Path(trainer.state.best_model_checkpoint)/"trainer_state.json", encoding="utf-8") as f:
        best_state = json.load(f)
    assert best_state["best_metric"] == trainer.state.best_metric
    # Check missing actual returned key and incorrect configured selection key fail clearly.
    try:
        trainer._determine_best_metric({"eval_validation_accuracy": .5}, trial=None)
    except ValueError as exc:
        missing_metric_error = str(exc)
    else:
        raise AssertionError("Missing metric did not fail")
    original_key = training_args.metric_for_best_model
    training_args.metric_for_best_model = "f1"
    try:
        build_development_trainer(model, training_args, train, validation, collator)
    except ValueError as exc:
        invalid_key_error = str(exc)
    else:
        raise AssertionError("Wrong metric key did not fail")
    finally:
        training_args.metric_for_best_model = original_key
    preserved_state = copy.deepcopy(trainer.state)
    try:
        trainer.state.best_metric = None
        selection_flags = []
        for step, value in enumerate([0.2, 0.8, 0.1], 101):
            trainer.state.global_step = step
            selection_flags.append(trainer._determine_best_metric({SELECTION_METRIC: value}, trial=None))
        assert selection_flags == [True, True, False]
        assert trainer.state.best_metric == 0.8 and trainer.state.best_global_step == 102
    finally:
        trainer.state = preserved_state
    saved = output / "reloaded_model"
    trainer.save_model(saved)
    reloaded = RobertaForSequenceClassification.from_pretrained(saved, local_files_only=True)
    reloaded.to(trainer.args.device).eval()
    trainer.model.eval()
    device_batch = {k:v.to(trainer.args.device) for k,v in batch.items()}
    with torch.no_grad():
        original_logits = trainer.model(**device_batch).logits
        reloaded_logits = reloaded(**device_batch).logits
    difference = (original_logits-reloaded_logits).abs().max().item()
    assert difference == 0.0, difference
    assert reloaded.config.label2id == LABEL2ID
    holdout.seen.clear()
    predicted = final_evaluate(reloaded, training_args, holdout, collator)
    assert holdout.seen == list(range(5)), holdout.seen
    assert len(predicted.predictions) == 5
    report = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
        scope="Invented synthetic examples, random tiny RoBERTa, no research-data training or Test predictions",
        device=str(training_args.device), world_size=training_args.world_size,
        pretrained_weights_used=False, full_size_model_capacity_validated=False,
        technical_only_settings=dict(seed=seed, learning_rate=1e-3, max_steps=2,
                                     batch_size=2, max_length=64, train_drop_last=True),
        tiny_model_config=config.to_dict(), model_parameter_count=sum(p.numel() for p in model.parameters()),
        metric_fixtures=check_metrics(), invalid_label_cases_rejected=True,
        stock_drop_last_true_eval_batch_sizes=stock_batches,
        stock_drop_last_true_test_batch_sizes=stock_test_batches,
        adapted_train_batch_sizes=train_batches, adapted_eval_batch_sizes=eval_batches,
        validation_indices_seen=validation.seen, synthetic_final_indices_seen=holdout.seen,
        synthetic_input_conditions_forward_passed=["base_text", "personality_only"],
        synthetic_condition_labels_identical=True,
        controlled_selector_fixture=dict(values=[0.2,0.8,0.1], new_best_flags=selection_flags, selected_step=102),
        finite_forward_loss=True, initial_forward_loss=initial_forward_loss,
        completed_optimizer_steps=trainer.state.global_step,
        validation_metric_key=SELECTION_METRIC, validation_only_selection=True,
        missing_metric_error=missing_metric_error, wrong_metric_configuration_error=invalid_key_error,
        best_checkpoint=str(Path(trainer.state.best_model_checkpoint).relative_to(ROOT)),
        selected_best_metric_matches_checkpoint_states=True,
        reload_logits_max_absolute_difference=difference,
        final_prediction_count=len(predicted.predictions),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        adapter_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "src/cultural_codes").glob("*.py")},
        note="Synthetic metric values are plumbing checks, not research performance.")
    (ROOT / "reports" / f"smoke_{args.device}.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("tiny_model_config","metric_fixtures")}, indent=2))

if __name__ == "__main__":
    main()

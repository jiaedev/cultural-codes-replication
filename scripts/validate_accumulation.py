"""Regression checks for example-weighted accumulation; invented inputs, CPU only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from safetensors.torch import load_file
from transformers import RobertaForSequenceClassification, TrainerCallback, default_data_collator
from transformers.trainer_utils import enable_full_determinism
from cultural_codes.protocol import (load_protocol, make_arguments, model_config,
    build_protocol_trainer, example_mean_cross_entropy, state_hash, sha256)


class Invented:
    def __init__(self, size):
        self.size, self.seen = size, []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.seen.append(index)
        return {"input_ids": [0, 5 + index % 30, 7, 8, 9, 10, 2],
                "attention_mask": [1] * 7, "labels": 0}


def arguments(spec, output):
    args = make_arguments(spec, "A", 13, cpu_check=True)
    args.output_dir = str(output)
    args.logging_dir = str(output / "logs")
    args.disable_tqdm = True
    args.num_train_epochs = 3
    return args


def tiny_model(spec, dropout=True):
    enable_full_determinism(13)
    config = model_config(ROOT, spec, tiny=True)
    config.vocab_size = 64
    if not dropout:
        config.hidden_dropout_prob = config.attention_probs_dropout_prob = config.classifier_dropout = 0.0
    return RobertaForSequenceClassification(config)


def gradients(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def gradient_checks(spec, output):
    results = []
    for sizes in ([8, 8, 8, 8], [8, 8, 1], [3]):
        model = tiny_model(spec, dropout=False)
        args = arguments(spec, output / ("gradient-" + "-".join(map(str, sizes))))
        trainer = build_protocol_trainer(spec, model, args, Invented(sum(sizes)), Invented(9), default_data_collator)
        labels = [0] * (sum(sizes) - 1) + [1]
        joined = {"input_ids": torch.tensor([[0, 5 + i % 30, 7, 8, 9, 10, 2] for i in range(sum(sizes))]),
                  "attention_mask": torch.ones(sum(sizes), 7, dtype=torch.long),
                  "labels": torch.tensor(labels)}
        batches, start = [], 0
        for size in sizes:
            batches.append({k: v[start:start + size] for k, v in joined.items()})
            start += size
        sampled, count = trainer.get_batch_samples(iter(batches), len(sizes), args.device)
        assert int(count) == sum(sizes)
        assert trainer.accelerator.gradient_accumulation_steps == 1
        trainer.current_gradient_accumulation_steps = len(sampled)
        accumulated_loss = 0.0
        for batch in sampled:
            accumulated_loss += trainer.training_step(model, dict(batch), count).item()
        actual = gradients(model)
        model.zero_grad()
        reference_loss = model(**joined).loss
        reference_loss.backward()
        reference = gradients(model)
        assert actual.keys() == reference.keys()
        for name in reference:
            torch.testing.assert_close(actual[name], reference[name], rtol=2e-5, atol=2e-7)
        assert abs(accumulated_loss - reference_loss.item()) < 2e-7
        result = {"microbatch_sizes": sizes, "denominator": int(count),
                  "parameter_gradient_tensors_checked": len(reference),
                  "max_gradient_absolute_difference": max((actual[n] - reference[n]).abs().max().item() for n in reference),
                  "loss_absolute_difference": abs(accumulated_loss - reference_loss.item()),
                  "rtol": 2e-5, "atol": 2e-7}
        if sizes == [8, 8, 1]:
            # Negative control: reproduce the pre-fix mean-of-microbatch-means.
            model.zero_grad()
            trainer.compute_loss_func = None
            for batch in batches:
                trainer.training_step(model, dict(batch), None)
            legacy = gradients(model)
            result["legacy_max_gradient_absolute_difference"] = max((legacy[n] - reference[n]).abs().max().item() for n in reference)
            assert result["legacy_max_gradient_absolute_difference"] > 0.01
        results.append(result)
    dummy = type("Output", (), {"logits": torch.zeros(1, 2)})()
    try:
        example_mean_cross_entropy(dummy, torch.tensor([0]))
    except ValueError:
        missing_count_rejected = True
    else:
        raise AssertionError("Missing accumulation count was accepted")
    return {"groups": results, "missing_count_rejected": missing_count_rejected,
            "dropout_disabled": True, "reference": "one physical batch with built-in mean CE, same parameters and cases"}


class Observe(TrainerCallback):
    def __init__(self):
        self.steps, self.orders, self.microbatches, self.clipping = [], [], [], []

    def on_epoch_begin(self, args, state, control, train_dataloader=None, **kwargs):
        train_dataloader.dataset.seen.clear()

    def on_epoch_end(self, args, state, control, train_dataloader=None, **kwargs):
        order = list(train_dataloader.dataset.seen)
        assert sorted(order) == list(range(49))
        self.orders.append(hashlib.sha256(json.dumps(order).encode()).hexdigest())

    def on_step_end(self, args, state, control, lr_scheduler=None, **kwargs):
        self.steps.append({"step": state.global_step, "epoch": state.epoch,
                           "learning_rate": lr_scheduler.get_last_lr()[0]})


def run_worker(spec, output, resume=None):
    args = arguments(spec, output)
    model = tiny_model(spec)
    # Deliberately wrong initial bias on invented class-0 inputs ensures clipping
    # is actually exercised at the protocol's norm=1, without changing its loss.
    with torch.no_grad():
        model.classifier.out_proj.bias.copy_(torch.tensor([-4.0, 4.0]))
    train, validation = Invented(49), Invented(9)
    trainer = build_protocol_trainer(spec, model, args, train, validation, default_data_collator)
    observed = Observe()
    trainer.add_callback(observed)
    trainer.compute_metrics = lambda prediction: {"macro_f1": 0.8, "accuracy": 0.5}
    original_step = trainer.training_step
    original_clip = trainer.accelerator.clip_grad_norm_

    def step(model, inputs, num_items_in_batch=None):
        observed.microbatches.append({"size": len(inputs["labels"]), "denominator": int(num_items_in_batch)})
        return original_step(model, inputs, num_items_in_batch)

    def clip(parameters, max_norm, norm_type=2):
        parameters = list(parameters)
        before = original_clip(parameters, max_norm, norm_type)
        after = torch.sqrt(sum(p.grad.detach().square().sum() for p in parameters if p.grad is not None))
        observed.clipping.append({"global_step_before_update": trainer.state.global_step,
                                  "microbatches_seen": len(observed.microbatches),
                                  "max_norm": float(max_norm), "before": float(before), "after": float(after)})
        return before

    trainer.training_step = step
    trainer.accelerator.clip_grad_norm_ = clip
    trainer.train(resume_from_checkpoint=resume)
    expected_epochs = 2 if resume else 3
    assert trainer.state.global_step == 6 and trainer.state.best_global_step == 2
    assert len(observed.steps) == len(observed.clipping) == expected_epochs * 2
    assert len(observed.orders) == expected_epochs
    assert [x["size"] for x in observed.microbatches] == [8, 8, 8, 8, 8, 8, 1] * expected_epochs
    assert [x["denominator"] for x in observed.microbatches] == ([32] * 4 + [17] * 3) * expected_epochs
    assert [x["microbatches_seen"] for x in observed.clipping] == [7 * e + b for e in range(expected_epochs) for b in (4, 7)]
    assert any(x["before"] > 1.0 for x in observed.clipping)
    assert all(x["after"] <= 1.00001 and x["max_norm"] == 1 for x in observed.clipping)
    selected = RobertaForSequenceClassification.from_pretrained(trainer.state.best_model_checkpoint, local_files_only=True)
    assert state_hash(selected) == state_hash(trainer.model)
    batches = [len(b["labels"]) for b in trainer.get_eval_dataloader(validation)]
    assert batches == [8, 1]
    evaluation = trainer.evaluate()
    model.eval()
    whole = default_data_collator([validation[i] for i in range(9)])
    with torch.no_grad():
        expected_loss = model(**whole).loss.item()
    assert abs(evaluation["eval_validation_loss"] - expected_loss) < 2e-6
    result = {"steps": observed.steps, "epoch_order_hashes": observed.orders,
              "microbatches": observed.microbatches, "clipping": observed.clipping,
              "every_case_once_per_epoch": True, "selected_step": 2,
              "selected_weights_reloaded": True, "evaluation_batch_sizes": batches,
              "evaluation_loss_absolute_difference": abs(evaluation["eval_validation_loss"] - expected_loss)}
    (output / "worker_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--resume")
    cli = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    spec = load_protocol(ROOT)
    if cli.worker:
        run_worker(spec, cli.worker, cli.resume)
        return
    base = ROOT / "artifacts/accumulation_validation" / spec["protocol_id"]
    base.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="check-", dir=base))
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_id": spec["protocol_id"], "protocol_sha256": sha256(ROOT / "configs/experimental_protocol.v1.json"),
              "script_sha256": sha256(Path(__file__)), "bridge_sha256": sha256(ROOT / "src/cultural_codes/protocol.py"),
              "versions": {k: version(k) for k in ("torch", "transformers", "accelerate")},
              "scope": "CPU, tiny random RoBERTa and invented token IDs only. Research split files only hashed. No Test predictions or pretrained weights.",
              "technical_overrides": {"device": "cpu", "model": "one layer, hidden 32, vocab 64",
                  "gradient_probe_dropout": 0, "integration_dropout": 0.1, "integration_training_examples": 49,
                  "integration_validation_examples": 9, "integration_epochs": 3,
                  "initial_classifier_bias": [-4, 4], "integration_labels": "all zero",
                  "selection_metric": "constant 0.8 to test earliest exact ties", "progress_display": False},
              "gradient_checks": gradient_checks(spec, output)}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for name in ("continuous", "resumed"):
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(output / name)]
        if name == "resumed":
            command += ["--resume", str(output / "continuous/checkpoint-2")]
        with (output / (name + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    continuous = json.loads((output / "continuous/worker_result.json").read_text())
    resumed = json.loads((output / "resumed/worker_result.json").read_text())
    left, right = output / "continuous/checkpoint-6", output / "resumed/checkpoint-6"
    a, b = load_file(str(left / "model.safetensors")), load_file(str(right / "model.safetensors"))
    difference = max((a[k] - b[k]).abs().max().item() for k in a)
    assert difference == 0.0
    assert continuous["epoch_order_hashes"][1:] == resumed["epoch_order_hashes"]
    assert continuous["steps"][2:] == resumed["steps"]
    states = {}
    for filename in ("optimizer.pt", "scheduler.pt"):
        # These checkpoints were just produced by this local synthetic fixture.
        states[filename] = state_equal(torch.load(left / filename, weights_only=False), torch.load(right / filename, weights_only=False))
        assert states[filename]
    report.update(continuous=continuous, resumed=resumed,
                  resume_comparison={"separate_processes": True, "max_parameter_absolute_difference": difference,
                                     "epoch_orders_and_steps_equal": True, "states_equal": states},
                  research_training_performed=False, gpu_validated=False,
                  local_artifacts=str(output.relative_to(ROOT)).replace("\\", "/"))
    (ROOT / "reports/accumulation_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": spec["protocol_id"], "gradient_checks": report["gradient_checks"],
                      "resume_comparison": report["resume_comparison"], "local_artifacts": report["local_artifacts"]}, indent=2))


if __name__ == "__main__":
    main()

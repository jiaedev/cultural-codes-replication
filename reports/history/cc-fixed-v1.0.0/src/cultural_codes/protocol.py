"""Versioned configuration bridge; research execution stays gated, no training CLI."""
import copy
import hashlib
import json
import math
from pathlib import Path
import torch
from transformers import AutoConfig, TrainingArguments, TrainerCallback
from .trainer import CompleteEvalTrainer, SELECTION_METRIC
from .metrics import compute_metrics
from .data import TextDataset


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol(root, name="configs/experimental_protocol.v1.json"):
    root = Path(root)
    spec = json.loads((root / name).read_text(encoding="utf-8"))
    assert spec["scope"]["final_training_runs"] == 10
    assert len(set(spec["scope"]["seeds"])) == 5
    assert spec["scope"]["conditions"] == {"A": "base_text", "B": "personality_only"}
    assert spec["model"]["revision"] == spec["tokenization"]["revision"]
    assert spec["data"]["label2id"] == {"False": 0, "True": 1}
    assert spec["training_arguments"]["metric_for_best_model"] == SELECTION_METRIC
    assert spec["training_arguments"]["dataloader_drop_last"] is False
    assert spec["final_evaluation"]["expected_rows"] == 478
    for split in spec["data"]["splits"].values():
        path = root / spec["data"]["directory"] / split["file"]
        if sha256(path) != split["sha256"]:
            raise ValueError("Recorded split bytes changed")
    return spec


def make_arguments(spec, condition, seed, cpu_check=False):
    if condition not in spec["scope"]["conditions"] or seed not in spec["scope"]["seeds"]:
        raise ValueError("Run is outside the specified plan")
    if not cpu_check:
        if not spec["execution_ready"]:
            raise RuntimeError("Research activation is blocked: satisfy execution gates first")
        if not torch.cuda.is_available() or torch.version.cuda is None:
            raise RuntimeError("Research target requires a separately verified CUDA environment")
    kwargs = copy.deepcopy(spec["training_arguments"])
    run_id = spec["outputs"]["run_id"].format(condition=condition, seed=seed)
    kwargs.update(output_dir=spec["outputs"]["root"] + "/runs/" + run_id,
                  run_name=run_id, logging_dir=spec["outputs"]["root"] + "/runs/" + run_id + "/logs",
                  seed=seed, data_seed=seed, use_cpu=cpu_check)
    return TrainingArguments(**kwargs)


def model_config(root, spec, tiny=False):
    config = AutoConfig.from_pretrained(Path(root) / spec["model"]["local_config"], local_files_only=True)
    for key, value in spec["model"]["config_overrides"].items():
        if key == "id2label":
            value = {int(k): v for k, v in value.items()}
        setattr(config, key, value)
    config._attn_implementation = spec["model"]["attention_implementation"]
    if tiny:
        config.hidden_size = 32
        config.num_hidden_layers = 1
        config.num_attention_heads = 4
        config.intermediate_size = 64
    return config


def make_dataset(spec, texts, labels, tokenizer):
    sizes = [len(x) for x in tokenizer(texts, truncation=False, add_special_tokens=True)["input_ids"]]
    overflow = [i + 1 for i, n in enumerate(sizes) if n > spec["tokenization"]["max_length"]]
    if overflow:
        raise ValueError(f"Protocol token overflow at provided data rows {overflow}; stop without truncating")
    return TextDataset(texts, labels, tokenizer, spec["tokenization"]["max_length"])


class FiniteGradients(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite gradient; protocol requires a recorded stop")


class ProtocolTrainer(CompleteEvalTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss; protocol requires a recorded stop")
        return loss


def build_protocol_trainer(spec, model, args, train_dataset, validation_dataset, collator):
    if args.metric_for_best_model != SELECTION_METRIC or not args.greater_is_better:
        raise ValueError("Wrong Validation selection configuration")
    settings = spec["optimizer"]
    optimizer_kwargs = {k: v for k, v in settings.items() if k != "class"}
    optimizer_kwargs.update(lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                            eps=args.adam_epsilon, weight_decay=args.weight_decay)
    return ProtocolTrainer(model=model, args=args, train_dataset=train_dataset,
        eval_dataset={"validation": validation_dataset}, data_collator=collator,
        compute_metrics=compute_metrics, optimizer_cls_and_kwargs=(torch.optim.AdamW, optimizer_kwargs),
        callbacks=[FiniteGradients()])


def state_hash(model, prefix=None):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if prefix is not None and not name.startswith(prefix):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def case_id(spec, split, data_row_1based):
    if not 1 <= data_row_1based <= spec["data"]["splits"][split]["rows"]:
        raise ValueError("Row outside the recorded split")
    raw = f"{split}:{spec['data']['splits'][split]['sha256']}:{data_row_1based}"
    return hashlib.sha256(raw.encode()).hexdigest()


def execution_plan(spec):
    train = spec["data"]["splits"]["train"]["rows"]
    args = spec["training_arguments"]
    micro = math.ceil(train / args["per_device_train_batch_size"])
    updates = math.ceil(micro / args["gradient_accumulation_steps"])
    epochs = int(args["num_train_epochs"])
    runs = []
    for index, seed in enumerate(spec["scope"]["seeds"]):
        for condition in (("A", "B") if index % 2 == 0 else ("B", "A")):
            run_id = spec["outputs"]["run_id"].format(condition=condition, seed=seed)
            runs.append(dict(run_id=run_id, seed=seed, condition=condition,
                input_column=spec["scope"]["conditions"][condition],
                output_directory=spec["outputs"]["root"] + "/runs/" + run_id,
                initial_snapshot=spec["outputs"]["initializations"].format(seed=seed),
                final_test_status="not run; wait for locked selection of all 10 models"))
    return dict(protocol_id=spec["protocol_id"], hyperparameter_search_runs=0, runs=runs,
        initialization_snapshots=5, train_microbatches_per_epoch=micro,
        optimizer_updates_per_epoch=updates, optimizer_updates_per_run=updates*epochs,
        optimizer_updates_all_runs=updates*epochs*len(runs),
        training_example_presentations=train*epochs*len(runs),
        validation_calls=epochs*len(runs), final_test_calls=len(runs),
        validation_example_evaluations=spec["data"]["splits"]["validation"]["rows"]*epochs*len(runs),
        final_test_example_evaluations=478*len(runs),
        nominal_effective_batch=args["per_device_train_batch_size"]*args["gradient_accumulation_steps"],
        last_update_examples_per_epoch=train % (args["per_device_train_batch_size"]*args["gradient_accumulation_steps"]),
        wall_time_measured=False)

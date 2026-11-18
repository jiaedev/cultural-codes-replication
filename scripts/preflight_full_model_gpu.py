"""Two-step full-size pretrained CUDA diagnostic on invented inputs only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, RobertaForSequenceClassification, TrainingArguments
from transformers.trainer_utils import enable_full_determinism
from cultural_codes.protocol import load_protocol, model_config, build_protocol_trainer, state_hash

class Observed:
    def __init__(self, encoding, count):
        self.encoding, self.count, self.seen = encoding, count, []
    def __len__(self): return self.count
    def __getitem__(self, index):
        self.seen.append(index)
        return {"input_ids": list(self.encoding), "attention_mask": [1] * len(self.encoding),
                "labels": index % 2}

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    spec = load_protocol(ROOT)
    model_dir = ROOT / spec["model"]["local_config"]
    weight = model_dir / spec["model"]["pretrained_file"]
    if not weight.is_file() or weight.stat().st_size != spec["model"]["pretrained_bytes"] or file_hash(weight) != spec["model"]["pretrained_sha256"]:
        raise RuntimeError("Pinned pretrained weights are absent or hash-mismatched")
    if torch.__version__ != spec["execution"]["target_versions"]["torch"] or not torch.cuda.is_available():
        raise RuntimeError("Verified CUDA torch environment is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("The protocol preflight requires one visible GPU")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    enable_full_determinism(13)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    tokenizer.padding_side = tokenizer.truncation_side = "right"
    # This uses invented text solely to exercise the full 512-token ceiling.
    encoding = tokenizer(" invented" * 2000, truncation=True, max_length=512)["input_ids"]
    assert len(encoding) == 512
    train, validation = Observed(encoding, 49), Observed(encoding, 9)
    collator = DataCollatorWithPadding(tokenizer, padding="longest")
    args_values = copy.deepcopy(spec["training_arguments"])
    preflight_dir = ROOT / 'artifacts' / ('gpu_preflight_' + file_hash(ROOT / 'configs/experimental_protocol.v1.json')[:10])
    args_values.update(output_dir=str(preflight_dir),
                       logging_dir=str(preflight_dir / 'logs'),
                       run_name="synthetic-full-model-preflight", seed=13, data_seed=13,
                       max_steps=2, num_train_epochs=1, report_to=[])
    args = TrainingArguments(**args_values)
    if args.device.type != "cuda" or args.n_gpu != 1:
        raise RuntimeError("Unexpected training device")
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Invented text and labels only; full-size pretrained RoBERTa on one GPU. No research training or Test predictions.",
              "protocol_id": spec["protocol_id"], "protocol_sha256": file_hash(ROOT / "configs/experimental_protocol.v1.json"),
              "bridge_sha256": file_hash(ROOT / "src/cultural_codes/protocol.py"),
              "script_sha256": file_hash(Path(__file__)),
              "source_weight_sha256": file_hash(weight),
              "environment": {"os": platform.system(), "python": platform.python_version(),
                              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name(0),
                              "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory},
              "technical_overrides": {"train_examples": 49, "validation_examples": 9,
                                      "token_length_with_special_tokens": 512, "max_steps": 2,
                                      "num_train_epochs": 1, "seed": 13}, "status": "running"}
    result_path = ROOT / "reports/full_model_gpu_preflight.json"
    started = perf_counter()
    try:
        model = RobertaForSequenceClassification.from_pretrained(
            model_dir, config=model_config(ROOT, spec), local_files_only=True,
            torch_dtype=torch.float32, attn_implementation="eager")
        report["model_parameters"] = sum(p.numel() for p in model.parameters())
        report["all_parameters_trainable"] = all(p.requires_grad for p in model.parameters())
        torch.cuda.reset_peak_memory_stats()
        trainer = build_protocol_trainer(spec, model, args, train, validation, collator)
        train_result = trainer.train()
        torch.cuda.synchronize()
        report["optimizer_steps"] = trainer.state.global_step
        report["examples_seen"] = len(train.seen)
        report["each_example_seen_once"] = sorted(train.seen) == list(range(49))
        report["validation_example_reads"] = len(validation.seen)
        report["best_checkpoint_exists"] = Path(trainer.state.best_model_checkpoint).is_dir()
        report["selected_checkpoint_step"] = trainer.state.best_global_step
        report["selection_metric_key_present"] = any(
            "eval_validation_macro_f1" in item for item in trainer.state.log_history)
        report["source_saved_reload_equal"] = (
            state_hash(trainer.model) == state_hash(
                RobertaForSequenceClassification.from_pretrained(
                    trainer.state.best_model_checkpoint, local_files_only=True)))
        report["evaluation_batch_sizes"] = [len(batch["labels"]) for batch in trainer.get_eval_dataloader(validation)]
        report["scheduler_final_learning_rate"] = trainer.lr_scheduler.get_last_lr()[0]
        assert report["optimizer_steps"] == 2 and report["each_example_seen_once"]
        assert report["source_saved_reload_equal"] and report["evaluation_batch_sizes"] == [8, 1]
        assert report["selection_metric_key_present"] and report["best_checkpoint_exists"]
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error_message"] = str(error)
        raise
    finally:
        report["elapsed_seconds"] = round(perf_counter() - started, 3)
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["gpu_utilization_context"] = "Single local RTX 5080 Laptop; synthetic 49/9 inputs all length 512; includes loading, train, Validation, checkpoint save/reload in elapsed time."
        result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))

if __name__ == "__main__": main()

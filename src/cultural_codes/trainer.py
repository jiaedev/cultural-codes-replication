"""Validation-only selection and complete evaluation batches."""
import math
from dataclasses import replace
from torch.utils.data import DataLoader, SequentialSampler
from transformers import Trainer
from .metrics import compute_metrics

SELECTION_METRIC = "eval_validation_macro_f1"

class CompleteEvalTrainer(Trainer):
    """Train drop_last may vary later; evaluation never drops the final batch."""
    def get_eval_dataloader(self, eval_dataset=None):
        if self.args.world_size != 1:
            raise NotImplementedError("This adapter has only been validated for one process")
        dataset = self.eval_dataset[eval_dataset] if isinstance(eval_dataset, str) else eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(dataset, dict):
            raise ValueError("Pass one evaluation dataset, not a split mapping")
        loader = DataLoader(dataset, batch_size=self.args.per_device_eval_batch_size,
                            sampler=SequentialSampler(dataset), collate_fn=self.data_collator,
                            drop_last=False, num_workers=0, pin_memory=False)
        return self.accelerator.prepare(loader)

    def get_test_dataloader(self, test_dataset):
        return self.get_eval_dataloader(test_dataset)

    def _determine_best_metric(self, metrics, trial):
        key = self.args.metric_for_best_model
        key = key if key.startswith("eval_") else "eval_" + key
        if key not in metrics:
            raise ValueError(f"Checkpoint selection metric {key!r} absent; available: {sorted(metrics)}")
        if not math.isfinite(float(metrics[key])):
            raise ValueError("Checkpoint selection metric is nonfinite")
        return super()._determine_best_metric(metrics, trial)

def build_development_trainer(model, args, train_dataset, validation_dataset, collator):
    if args.metric_for_best_model != SELECTION_METRIC or args.greater_is_better is not True:
        raise ValueError(f"metric_for_best_model must be {SELECTION_METRIC!r}, greater_is_better=True")
    return CompleteEvalTrainer(model=model, args=args, train_dataset=train_dataset,
                               eval_dataset={"validation": validation_dataset},
                               data_collator=collator, compute_metrics=compute_metrics)

def final_evaluate(model, args, dataset, collator):
    """Separate final evaluation API. Not called on research Test during Stage 3."""
    final_args = replace(args, eval_strategy="no", save_strategy="no",
                         load_best_model_at_end=False, metric_for_best_model=None, greater_is_better=None)
    trainer = CompleteEvalTrainer(model=model, args=final_args, data_collator=collator,
                                 compute_metrics=compute_metrics)
    output = trainer.predict(dataset, metric_key_prefix="final")
    if len(output.predictions) != len(dataset) or len(output.label_ids) != len(dataset):
        raise RuntimeError("Final evaluation did not preserve every example")
    return output

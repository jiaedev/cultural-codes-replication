"""Resolve the specified plan and test it with invented data/tiny random models only."""
import copy
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import random
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from transformers import (AutoTokenizer, DataCollatorWithPadding,
                          RobertaForSequenceClassification)
from transformers.trainer_utils import enable_full_determinism
from cultural_codes.data import TextDataset
from cultural_codes.protocol import (load_protocol, make_arguments, model_config,
    build_protocol_trainer, state_hash, case_id, execution_plan, sha256, make_dataset)


class Observed(TextDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = []
    def __getitem__(self, index):
        self.seen.append(index)
        return super().__getitem__(index)


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = load_protocol(ROOT)
    plan = execution_plan(spec)
    output = ROOT / "artifacts/protocol_dry_run"
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(ROOT/spec["model"]["local_config"], local_files_only=True)
    tokenizer.padding_side = spec["tokenization"]["padding_side"]
    tokenizer.truncation_side = spec["tokenization"]["truncation_side"]
    collator = DataCollatorWithPadding(tokenizer, padding=True,
                                     pad_to_multiple_of=spec["tokenization"]["pad_to_multiple_of"])
    texts = [f"remaining: ['apple', 'stone'], rationale: invented technical case {i}, target: apple, hint: fruit" for i in range(35)]
    profiles = ["GIVER: [None], GUESSER: [None], " + text for text in texts]
    labels = ["True" if i % 2 else "False" for i in range(35)]
    make_dataset(spec, texts, labels, tokenizer)
    try:
        make_dataset(spec, ["invented " * 600], ["False"], tokenizer)
    except ValueError:
        overflow_rejected = True
    else:
        raise AssertionError("Silent token truncation accepted")
    datasets = {c:Observed(t,labels,tokenizer,spec["tokenization"]["max_length"])
                for c,t in (("A",texts),("B",profiles))}
    validation = Observed(texts[:5],labels[:5],tokenizer,spec["tokenization"]["max_length"])
    tiny = model_config(ROOT,spec,tiny=True)
    resolved = {}
    checks = []
    first_trainer = None
    for seed in spec["scope"]["seeds"]:
        enable_full_determinism(seed)
        common = RobertaForSequenceClassification(tiny)
        initial = output / f"seed-{seed}-initial"
        common.save_pretrained(initial)
        seed_trainer = {}
        for condition in ("A","B"):
            model = RobertaForSequenceClassification.from_pretrained(initial,local_files_only=True)
            if condition == "A":
                args = make_arguments(spec,condition,seed,cpu_check=True)
            else:
                args = copy.deepcopy(seed_trainer["A"].args)
                args.run_name = spec["outputs"]["run_id"].format(condition=condition, seed=seed)
            # Keep the scientific settings; change only fixture location/progress display.
            args.output_dir = str(output/f"seed-{seed}-{condition}")
            args.logging_dir = str(output/f"seed-{seed}-{condition}"/"logs")
            args.disable_tqdm = True
            trainer = build_protocol_trainer(spec,model,args,datasets[condition],validation,collator)
            resolved[args.run_name] = args.to_dict()
            resolved[args.run_name]["output_dir"] = spec["outputs"]["root"] + "/runs/" + args.run_name
            resolved[args.run_name]["logging_dir"] = resolved[args.run_name]["output_dir"] + "/logs"
            seed_trainer[condition] = trainer
        equal_all = state_hash(seed_trainer["A"].model)==state_hash(seed_trainer["B"].model)
        equal_head = state_hash(seed_trainer["A"].model,"classifier.")==state_hash(seed_trainer["B"].model,"classifier.")
        assert equal_all and equal_head
        orders = {}
        for condition,trainer in seed_trainer.items():
            loader=trainer.get_train_dataloader()
            assert "SeedableRandomSampler" in type(loader.sampler).__name__ or "SeedableRandomSampler" in type(loader.batch_sampler.sampler).__name__
            orders[condition]=[]
            for epoch in (0,1):
                datasets[condition].seen.clear()
                loader.set_epoch(epoch)
                # Deliberately consume differing global RNG amounts between conditions.
                torch.rand(7 if condition=="A" else 127)
                for batch in loader: pass
                sequence=list(datasets[condition].seen)
                assert sorted(sequence)==list(range(35))
                orders[condition].append(sequence)
        assert orders["A"]==orders["B"] and orders["A"][0]!=orders["A"][1]
        checks.append(dict(seed=seed,all_initial_parameters_equal=equal_all,head_parameters_equal=equal_head,initial_head_sha256=state_hash(seed_trainer["A"].model,"classifier."),
            epoch_orders_equal=True,every_case_seen_each_epoch=True,
            order_hashes=[hashlib.sha256(json.dumps(x).encode()).hexdigest() for x in orders["A"]]))
        if first_trainer is None:
            first_trainer=seed_trainer["A"]
    # One short synthetic train, not five pairs of performance experiments.
    args=make_arguments(spec,"A",spec["scope"]["seeds"][0],cpu_check=True)
    args.output_dir=str(output/"short-synthetic-train")
    args.logging_dir=str(output/"short-synthetic-train/logs")
    args.disable_tqdm=True
    args.num_train_epochs=1.0
    model=RobertaForSequenceClassification.from_pretrained(output/("seed-"+str(spec["scope"]["seeds"][0])+"-initial"),local_files_only=True)
    first_trainer=build_protocol_trainer(spec,model,args,datasets["A"],validation,collator)
    result=first_trainer.train()
    assert first_trainer.state.global_step==2
    assert set(first_trainer.eval_dataset)=={"validation"}
    evaluation_sizes=[len(batch["labels"]) for batch in first_trainer.get_eval_dataloader(validation)]
    assert evaluation_sizes==[5]
    first_trainer.create_optimizer()
    optimizer=first_trainer.optimizer
    actual_optimizer=getattr(optimizer,"optimizer",optimizer)
    assert actual_optimizer.defaults["foreach"] is False
    assert actual_optimizer.defaults["fused"] is False
    preserved=copy.deepcopy(first_trainer.state)
    first_trainer.state.best_metric=None
    flags=[first_trainer._determine_best_metric({"eval_validation_macro_f1":x},None) for x in (.8,.8,.7)]
    assert flags==[True,False,False]
    first_trainer.state=preserved
    assert len({case_id(spec,"test",i) for i in range(1,479)})==478
    try:
        make_arguments(spec,"A",spec["scope"]["seeds"][0])
    except RuntimeError:
        activation_blocked=True
    else:raise AssertionError("Research activation was not blocked")
    # Resolve full-size parameter count on meta device, not a full-model execution.
    with torch.device("meta"):
        structural_model=RobertaForSequenceClassification(model_config(ROOT,spec))
    parameter_count=sum(p.numel() for p in structural_model.parameters())
    trainer_steps=first_trainer.set_initial_training_values(
        make_arguments(spec,"A",spec["scope"]["seeds"][0],cpu_check=True),
        type("SizedLoader",(),{"__len__":lambda self:plan["train_microbatches_per_epoch"],"dataset":range(8497)})(),32)
    assert trainer_steps[1]==266 and trainer_steps[-1]==3990
    for pair in checks:assert pair["epoch_orders_equal"]
    assert len({x["initial_head_sha256"] for x in checks}) == 5
    reports=ROOT/"reports"
    (reports/"experimental_run_plan.json").write_text(json.dumps(plan,indent=2)+"\n",encoding="utf-8")
    (reports/"resolved_protocol_arguments.json").write_text(json.dumps(dict(
        note="Actual installed TrainingArguments defaults resolved on CPU; CUDA target not validated. Relative run paths.",runs=resolved),indent=2)+"\n",encoding="utf-8")
    report=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),protocol_id=spec["protocol_id"],
        protocol_sha256=sha256(ROOT/"configs/experimental_protocol.v1.json"),
        scope="CPU synthetic/tiny only; research data files hashed but not tokenized or trained; no Test predictions.",
        paired_seed_checks=checks,synthetic_optimizer_steps=first_trainer.state.global_step,
        technical_overrides=dict(device="cpu",model="random tiny",examples=35,epochs=1,output="ignored artifacts/protocol_dry_run",progress_display=False),
        exact_tie_keeps_earliest=flags==[True,False,False],evaluation_partial_batch_sizes=evaluation_sizes,
        optimizer_defaults=actual_optimizer.defaults,scheduler=type(first_trainer.lr_scheduler).__name__,
        trainer_predicted_updates_per_epoch=trainer_steps[1],trainer_predicted_updates_per_run=trainer_steps[-1],
        full_model_structural_parameter_count=parameter_count,full_model_execution_validated=False,
        research_activation_blocked=activation_blocked,gpu_validated=False,pretrained_weights_downloaded=False,
        case_ids_unique_478=True,synthetic_token_overflow_rejected=overflow_rejected,
        different_seeds_initialize_different_heads=True,old_adapter_files_modified=False,
        script_sha256=sha256(Path(__file__)),bridge_sha256=sha256(ROOT/"src/cultural_codes/protocol.py"))
    (reports/"protocol_validation.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("paired_seed_checks","optimizer_defaults")},indent=2))


if __name__=="__main__":
    main()

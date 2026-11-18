"""Measure short full-model CUDA blocks at synthetic A/B-like sequence lengths."""
import copy
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, RobertaForSequenceClassification, TrainingArguments
from transformers.trainer_utils import enable_full_determinism
from cultural_codes.protocol import load_protocol, model_config, build_protocol_trainer
from preflight_full_model_gpu import Observed, file_hash


def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    spec=load_protocol(ROOT)
    weight=ROOT/spec['model']['local_config']/spec['model']['pretrained_file']
    if not weight.exists() or file_hash(weight)!=spec['model']['pretrained_sha256']:
        raise RuntimeError('Pinned pretrained weights required')
    if not torch.cuda.is_available() or torch.__version__!=spec['execution']['target_versions']['torch']:
        raise RuntimeError('Verified CUDA runtime required')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    tokenizer=AutoTokenizer.from_pretrained(ROOT/spec['model']['local_config'],local_files_only=True,use_fast=True)
    rows=[]
    for condition,length in [('A',100),('B',170)]:
        enable_full_determinism(13)
        ids=tokenizer(' invented'*2000,truncation=True,max_length=length)['input_ids']
        assert len(ids)==length
        train=Observed(ids,160);validation=Observed(ids,9)
        args_values=copy.deepcopy(spec['training_arguments'])
        benchmark_dir=ROOT/'artifacts'/f"gpu_benchmark_{condition}_{file_hash(ROOT/'configs/experimental_protocol.v1.json')[:10]}"
        args_values.update(output_dir=str(benchmark_dir),
            logging_dir=str(benchmark_dir/'logs'),
            run_name=f'invented-benchmark-{condition}',seed=13,data_seed=13,
            max_steps=5,num_train_epochs=1,report_to=[],disable_tqdm=True)
        args=TrainingArguments(**args_values)
        model=RobertaForSequenceClassification.from_pretrained(ROOT/spec['model']['local_config'],
            config=model_config(ROOT,spec),local_files_only=True,dtype=torch.float32,attn_implementation='eager')
        torch.cuda.reset_peak_memory_stats()
        trainer=build_protocol_trainer(spec,model,args,train,validation,DataCollatorWithPadding(tokenizer,padding='longest'))
        started=perf_counter()
        result=trainer.train()
        torch.cuda.synchronize()
        elapsed=perf_counter()-started
        assert trainer.state.global_step==5 and sorted(train.seen)==list(range(160))
        assert len(validation.seen)==9
        rows.append(dict(condition=condition,synthetic_token_length=length,
            training_examples=160,optimizer_steps=5,elapsed_train_eval_save_seconds=round(elapsed,3),
            trainer_reported_runtime_seconds=result.metrics['train_runtime'],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        print(json.dumps(rows[-1]),flush=True)
        del trainer,model
        torch.cuda.empty_cache()
    report=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Five updates each of invented fixed-length token IDs at 100 and 170, full-size pretrained model, same optimizer/config; technical timing only.',
        protocol_id=spec['protocol_id'],protocol_sha256=file_hash(ROOT/'configs/experimental_protocol.v1.json'),
        script_sha256=file_hash(Path(__file__)),
        source_weight_sha256=file_hash(weight),environment=dict(os=platform.system(),torch=torch.__version__,gpu=torch.cuda.get_device_name(0)),
        blocks=rows,selection_performance_claim=False,actual_research_training=False)
    (ROOT/'reports/gpu_synthetic_benchmark.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()

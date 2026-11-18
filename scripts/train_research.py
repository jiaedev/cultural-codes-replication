"""Prepare paired starts or execute one frozen research run in a fresh process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, RobertaForSequenceClassification, TrainerCallback
from transformers.trainer_utils import enable_full_determinism
from cultural_codes.data import read_rows
from cultural_codes.protocol import load_protocol, make_arguments, model_config, make_dataset, build_protocol_trainer, state_hash

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_new(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2);f.write('\n')
def append(path,obj):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(obj)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def root_for(spec):return ROOT/spec['outputs']['root']
def check_weight(spec):
    p=ROOT/spec['model']['local_config']/spec['model']['pretrained_file']
    if not p.is_file() or p.stat().st_size!=spec['model']['pretrained_bytes'] or sha(p)!=spec['model']['pretrained_sha256']:
        raise ValueError('Pinned pretrained weight bytes are missing or changed')
    return p

def prepare(spec,seed):
    if seed not in spec['scope']['seeds']:raise ValueError('Unplanned seed')
    check_weight(spec)
    p=ROOT/spec['outputs']['initializations'].format(seed=seed)
    if p.exists():
        m=json.loads((p/'initialization_hashes.json').read_text())
        if sha(p/'model.safetensors')!=m['snapshot_sha256']:raise ValueError('Existing initial snapshot changed')
        print(json.dumps({'seed':seed,'status':'already verified','snapshot_sha256':m['snapshot_sha256']}));return
    enable_full_determinism(seed)
    model=RobertaForSequenceClassification.from_pretrained(ROOT/spec['model']['local_config'],
        config=model_config(ROOT,spec),local_files_only=True,dtype=torch.float32,attn_implementation='eager')
    tmp=p.with_name(p.name+'.partial')
    if tmp.exists():raise RuntimeError('Incomplete initialization snapshot requires manual inspection')
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp,safe_serialization=True)
    snapshot=sha(tmp/'model.safetensors')
    record={'seed':seed,'created_at_utc':now(),'protocol_sha256':sha(ROOT/'configs/experimental_protocol.v1.json'),
        'source_weight_sha256':spec['model']['pretrained_sha256'],'snapshot_sha256':snapshot,
        'full_parameter_sha256':state_hash(model),'head_parameter_sha256':state_hash(model,'classifier.')}
    write_new(tmp/'initialization_hashes.json',record)
    tmp.rename(p)
    print(json.dumps({'seed':seed,'snapshot_sha256':snapshot,'head_parameter_sha256':record['head_parameter_sha256']}))

def activation(spec):
    p=root_for(spec)/'activation.json'
    if not spec['execution_ready'] or not p.is_file():raise RuntimeError('Execution activation is not frozen')
    a=json.loads(p.read_text(encoding='utf-8'))
    if a['protocol_sha256']!=sha(ROOT/'configs/experimental_protocol.v1.json'):
        raise RuntimeError('Protocol changed after activation')
    for name,digest in a['code_sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Execution code changed after activation: '+name)
    if a['gpu_lock_sha256']!=sha(ROOT/'environment/requirements-win-gpu.lock.txt'):
        raise RuntimeError('GPU lock changed after activation')
    if a['budget_sha256']!=sha(ROOT/'configs/execution_budget.json'):
        raise RuntimeError('Execution budget changed after activation')
    if not torch.cuda.is_available() or torch.version.cuda is None or torch.cuda.device_count()!=1:
        raise RuntimeError('Single CUDA device unavailable')
    if torch.__version__!=spec['execution']['target_versions']['torch']:
        raise RuntimeError('Torch version changed')
    return a

class BudgetStop(TrainerCallback):
    """Stop at a step boundary if the predeclared total GPU time is exhausted."""
    def __init__(self, started, already_used, limit):
        self.started, self.already_used, self.limit = started, already_used, limit
    def on_step_end(self, args, state, control, **kwargs):
        if self.already_used + perf_counter() - self.started >= self.limit:
            control.should_training_stop = True
        return control

class SeenDataset:
    def __init__(self,base):self.base,self.seen=base,[]
    def __len__(self):return len(self.base)
    def __getitem__(self,index):self.seen.append(index);return self.base[index]

class RecordEpochs(TrainerCallback):
    def __init__(self,dataset,out,starting_epoch=0):self.dataset,self.out,self.epoch=dataset,out,starting_epoch
    def on_epoch_begin(self,args,state,control,**kwargs):self.dataset.seen.clear()
    def on_epoch_end(self,args,state,control,**kwargs):
        order=self.dataset.seen
        if len(order)!=len(self.dataset) or sorted(order)!=list(range(len(self.dataset))):
            raise RuntimeError('Train coverage mismatch')
        append(self.out,{'epoch_1based':self.epoch+1,'count':len(order),
            'ordered_source_row_sha256':hashlib.sha256(json.dumps(order).encode()).hexdigest()})
        self.epoch+=1

def train(spec,condition,seed,resume):
    if condition not in ('A','B') or seed not in spec['scope']['seeds']:raise ValueError('Unplanned run')
    os.chdir(ROOT)
    frozen=activation(spec)
    check_weight(spec)
    run_id=spec['outputs']['run_id'].format(condition=condition,seed=seed)
    plan=json.loads((ROOT/'reports/experimental_run_plan.json').read_text())
    if run_id not in [r['run_id'] for r in plan['runs']]:raise RuntimeError('Run absent from plan')
    out=root_for(spec)/'runs'/run_id
    if (out/'selected_checkpoint.json').exists():raise RuntimeError('Completed run cannot be overwritten')
    attempts=out/'attempts.jsonl'
    previous=[json.loads(line) for line in attempts.read_text(encoding='utf-8').splitlines()] if attempts.exists() else []
    started=[entry for entry in previous if entry['status']=='running']
    if len(started)>=2:raise RuntimeError('Recovery limit reached')
    if bool(started)!=bool(resume):raise RuntimeError('First attempt or resume sequence mismatch')
    if started and not resume:raise RuntimeError('Recovery requires an explicit complete checkpoint')
    init=ROOT/spec['outputs']['initializations'].format(seed=seed)
    im=json.loads((init/'initialization_hashes.json').read_text())
    if sha(init/'model.safetensors')!=im['snapshot_sha256']:raise RuntimeError('Initial snapshot hash mismatch')
    if im['protocol_sha256']!=frozen['pre_activation_protocol_sha256'] and im['protocol_sha256']!=frozen['protocol_sha256']:
        raise RuntimeError('Initial snapshot protocol provenance mismatch')
    if resume:
        cp=out/resume
        if not cp.is_dir() or not all((cp/name).exists() for name in ('model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json')):
            raise RuntimeError('Only a complete local checkpoint can be resumed')
    else:cp=None
    source=ROOT/spec['data']['directory']
    rows={name:read_rows(source/filename) for name,filename in [('train','train.csv'),('validation','val.csv')]}
    for split,items in rows.items():
        if len(items)!=spec['data']['splits'][split]['rows']:raise RuntimeError('Split row count mismatch')
        if any(not row['personality_only'].endswith(row['base_text']) for row in items):
            raise RuntimeError('A/B base content differs')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/spec['model']['local_config'],local_files_only=True,use_fast=True)
    tokenizer.padding_side=tokenizer.truncation_side='right'
    col=spec['scope']['conditions'][condition]
    datasets={name:SeenDataset(make_dataset(spec,[r[col] for r in items],[r['output'] for r in items],tokenizer))
              for name,items in rows.items()}
    enable_full_determinism(seed)
    model=RobertaForSequenceClassification.from_pretrained(init,local_files_only=True)
    if state_hash(model)!=im['full_parameter_sha256'] or state_hash(model,'classifier.')!=im['head_parameter_sha256']:
        raise RuntimeError('Loaded start differs from paired initialization')
    args=make_arguments(spec,condition,seed)
    if args.output_dir!=str(out) and Path(args.output_dir).resolve()!=out.resolve():
        raise RuntimeError('Resolved output path differs from plan')
    epoch_log=out/f'attempt-{len(started)+1}-epoch_case_order_hashes.jsonl'
    if epoch_log.exists():raise RuntimeError('Attempt epoch-order log already exists')
    starting_epoch=int(json.loads((cp/'trainer_state.json').read_text())['global_step']//266) if cp else 0
    callback=RecordEpochs(datasets['train'],epoch_log,starting_epoch)
    trainer=build_protocol_trainer(spec,model,args,datasets['train'],datasets['validation'],
        DataCollatorWithPadding(tokenizer,padding='longest'))
    trainer.add_callback(callback)
    budget=json.loads((ROOT/'configs/execution_budget.json').read_text(encoding='utf-8'))
    used=0.0
    for other in (root_for(spec)/'runs').glob('*/attempts.jsonl'):
        used+=sum(float(x.get('elapsed_seconds',0)) for x in (json.loads(line) for line in other.read_text(encoding='utf-8').splitlines()) if x['status'] in ('complete','failed'))
    if used>=budget['total_gpu_wall_time_limit_seconds']:
        raise RuntimeError('Declared total GPU time ceiling already reached')
    out.mkdir(parents=True,exist_ok=True)
    if not previous:
        for filename,obj in [('config.json',spec),('environment.json',frozen['environment']),
            ('source_hashes.json',{k:v['sha256'] for k,v in spec['data']['splits'].items()}),
            ('initialization_hashes.json',im)]:write_new(out/filename,obj)
    attempt={'attempt':len(started)+1,'start_utc':now(),'resume_checkpoint':resume,'status':'running'}
    append(attempts,attempt)
    start=perf_counter()
    trainer.add_callback(BudgetStop(start,used,budget['total_gpu_wall_time_limit_seconds']))
    try:
        result=trainer.train(resume_from_checkpoint=str(cp) if cp else None)
        if trainer.state.global_step!=3990 or callback.epoch!=15:
            raise RuntimeError('Run ended before 15 full epochs / 3990 updates')
        selected=Path(trainer.state.best_model_checkpoint)
        if not selected.is_dir():raise RuntimeError('Selected checkpoint missing')
        epoch=trainer.state.best_global_step//266
        if trainer.state.best_global_step!=epoch*266 or not 1<=epoch<=15:
            raise RuntimeError('Selected checkpoint is not a full epoch')
        selected_record={'run_id':run_id,'condition':condition,'seed':seed,'selected_epoch':epoch,
            'selected_step':trainer.state.best_global_step,'validation_macro_f1':trainer.state.best_metric,
            'checkpoint':str(selected.resolve().relative_to(out.resolve())).replace('\\','/'),
            'checkpoint_weight_sha256':sha(selected/'model.safetensors'),
            'initial_snapshot_sha256':im['snapshot_sha256'],
            'protocol_sha256':frozen['protocol_sha256'],'environment_lock_sha256':frozen['gpu_lock_sha256'],
            'train_examples_per_epoch':8497,'validation_examples_per_epoch':544,
            'optimizer_steps':trainer.state.global_step,'elapsed_seconds':round(perf_counter()-start,3),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        (out/'train_log.json').write_text(json.dumps(trainer.state.log_history,indent=2)+'\n',encoding='utf-8')
        write_new(out/'selected_checkpoint.json',selected_record)
        attempt.update(end_utc=now(),status='complete',elapsed_seconds=selected_record['elapsed_seconds'])
        append(attempts,attempt)
        print(json.dumps(selected_record,indent=2))
    except Exception as error:
        attempt.update(end_utc=now(),status='failed',error_type=type(error).__name__,
                       error=str(error),elapsed_seconds=round(perf_counter()-start,3))
        append(attempts,attempt)
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--seed',type=int,required=True)
    p=sub.add_parser('train');p.add_argument('--condition',choices=['A','B'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--resume')
    opt=parser.parse_args();spec=load_protocol(ROOT)
    if opt.command=='prepare':prepare(spec,opt.seed)
    else:train(spec,opt.condition,opt.seed,opt.resume)
if __name__=='__main__':main()

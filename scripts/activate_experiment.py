"""Verify all pre-execution gates and write an immutable local activation record."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from transformers import AutoTokenizer
from cultural_codes.data import read_rows
from cultural_codes.protocol import load_protocol, make_arguments, make_dataset
from train_research import sha, check_weight, root_for, write_new

def main():
    os.chdir(ROOT)
    spec=load_protocol(ROOT)
    if not spec['execution_ready'] or spec['protocol_id']!='cc-fixed-v1.0.1':
        raise RuntimeError('Frozen protocol configuration missing')
    root=root_for(spec)
    if (root/'activation.json').exists():raise RuntimeError('Activation already exists; do not overwrite it')
    if any((root/'runs').glob('*/attempts.jsonl')):
        raise RuntimeError('Research attempts already exist; activation cannot be replaced')
    budget=json.loads((ROOT/'configs/execution_budget.json').read_text(encoding='utf-8'))
    if (budget['protocol_id']!=spec['protocol_id'] or budget['external_service_cost_limit_krw']!=0
        or budget['total_gpu_wall_time_limit_seconds']!=86400):
        raise RuntimeError('User-selected local 24-hour budget differs')
    if torch.__version__!=spec['execution']['target_versions']['torch'] or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Verified one-device CUDA environment required')
    if platform.system()!='Windows' or platform.python_version()!='3.11.17':
        raise RuntimeError('GPU environment differs from the verified local target')
    for package,key in [('transformers','transformers'),('accelerate','accelerate'),
                        ('scikit-learn','scikit_learn'),('tokenizers','tokenizers'),('regex','regex')]:
        if version(package)!=spec['execution']['target_versions'][key]:
            raise RuntimeError('GPU dependency changed: '+package)
    lock=ROOT/'environment/requirements-win-gpu.lock.txt'
    actual=subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True)
    locked=[x.lower() for x in lock.read_text(encoding='utf-8').splitlines() if x and not x.startswith(('#','--'))]
    if sorted(x.replace('_','-') for x in locked)!=sorted(x.lower().replace('_','-') for x in actual.splitlines() if x):
        raise RuntimeError('Installed packages differ from GPU lock')
    weight=check_weight(spec)
    preflight=json.loads((ROOT/'reports/full_model_gpu_preflight.json').read_text(encoding='utf-8'))
    if (preflight['status']!='passed' or preflight['protocol_sha256']!=sha(ROOT/'configs/experimental_protocol.v1.json')
        or preflight['script_sha256']!=sha(ROOT/'scripts/preflight_full_model_gpu.py')
        or preflight['bridge_sha256']!=sha(ROOT/'src/cultural_codes/protocol.py')
        or preflight['source_weight_sha256']!=sha(weight)):
        raise RuntimeError('Full-size CUDA preflight does not match active inputs')
    if not all(preflight[k] for k in ('all_parameters_trainable','each_example_seen_once','best_checkpoint_exists',
                                      'selection_metric_key_present','source_saved_reload_equal')):
        raise RuntimeError('Full-size CUDA preflight is incomplete')
    if preflight['optimizer_steps']!=2 or preflight['evaluation_batch_sizes']!=[8,1]:
        raise RuntimeError('Full-size CUDA preflight coverage differs')
    for name,script in [('protocol_validation.json','validate_protocol.py'),
                        ('accumulation_validation.json','validate_accumulation.py')]:
        report=json.loads((ROOT/'reports'/name).read_text(encoding='utf-8'))
        if (report['protocol_sha256']!=sha(ROOT/'configs/experimental_protocol.v1.json')
            or report['bridge_sha256']!=sha(ROOT/'src/cultural_codes/protocol.py')
            or report['script_sha256']!=sha(ROOT/'scripts'/script)):
            raise RuntimeError('Recorded CPU regression is stale: '+name)
    old_hash=sha(ROOT/'reports/history/cc-fixed-v1.0.1-preactivation/configs/experimental_protocol.v1.json')
    initialization={}
    for seed in spec['scope']['seeds']:
        path=ROOT/spec['outputs']['initializations'].format(seed=seed)
        m=json.loads((path/'initialization_hashes.json').read_text(encoding='utf-8'))
        if m['protocol_sha256']!=old_hash or sha(path/'model.safetensors')!=m['snapshot_sha256']:
            raise RuntimeError('Paired initial snapshot mismatched: '+str(seed))
        initialization[str(seed)]=m['snapshot_sha256']
    if len(set(initialization.values()))!=5:raise RuntimeError('Initial snapshots unexpectedly equal')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/spec['model']['local_config'],local_files_only=True,use_fast=True)
    tokenizer.padding_side=tokenizer.truncation_side='right'
    lengths={}
    for split,filename in [('train','train.csv'),('validation','val.csv')]:
        rows=read_rows(ROOT/spec['data']['directory']/filename)
        if len(rows)!=spec['data']['splits'][split]['rows']:raise RuntimeError('Data row count changed')
        if any(not r['personality_only'].endswith(r['base_text']) for r in rows):
            raise RuntimeError('A/B base suffix differs')
        for condition,column in spec['scope']['conditions'].items():
            dataset=make_dataset(spec,[r[column] for r in rows],[r['output'] for r in rows],tokenizer)
            lengths[split+'_'+condition]={'rows':len(dataset),'max_tokens':max(len(x) for x in dataset.encodings['input_ids'])}
    plan=json.loads((ROOT/'reports/experimental_run_plan.json').read_text(encoding='utf-8'))
    if len(plan['runs'])!=10 or plan['protocol_id']!=spec['protocol_id']:
        raise RuntimeError('Ten-run plan changed')
    resolved={}
    for run in plan['runs']:
        args=make_arguments(spec,run['condition'],run['seed'])
        value=args.to_dict()
        for key in ('output_dir','logging_dir'):
            value[key]=str(Path(value[key])).replace('\\','/')
        if value['use_cpu'] or value['seed']!=run['seed'] or value['data_seed']!=run['seed']:
            raise RuntimeError('Resolved GPU arguments changed')
        resolved[run['run_id']]=value
    (ROOT/'reports/resolved_gpu_arguments.json').write_text(json.dumps({'note':'Resolved from the active single-CUDA environment; run paths relative to project root.','runs':resolved},indent=2)+'\n',encoding='utf-8')
    code=['src/cultural_codes/protocol.py','src/cultural_codes/trainer.py','src/cultural_codes/data.py',
          'src/cultural_codes/metrics.py','scripts/train_research.py','scripts/execute_plan.py',
          'scripts/evaluate_final.py','scripts/activate_experiment.py']
    for name in code:
        compile((ROOT/name).read_text(encoding='utf-8-sig'),name,'exec')
    environment={'os':'Windows','python':platform.python_version(),'torch':torch.__version__,
        'transformers':version('transformers'),'accelerate':version('accelerate'),
        'scikit_learn':version('scikit-learn'),'tokenizers':version('tokenizers'),
        'gpu':torch.cuda.get_device_name(0),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,
        'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()}
    record={'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_id':spec['protocol_id'],'protocol_sha256':sha(ROOT/'configs/experimental_protocol.v1.json'),
        'pre_activation_protocol_sha256':old_hash,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        'code_sha256':{name:sha(ROOT/name) for name in code},
        'gpu_lock_sha256':sha(lock),'budget_sha256':sha(ROOT/'configs/execution_budget.json'),
        'source_weights_sha256':sha(weight),'data_sha256':{k:v['sha256'] for k,v in spec['data']['splits'].items()},
        'tokenizer_manifest_sha256':sha(ROOT/'reports/tokenizer_manifest.json'),
        'gpu_preflight_report_sha256':sha(ROOT/'reports/full_model_gpu_preflight.json'),
        'cpu_protocol_report_sha256':sha(ROOT/'reports/protocol_validation.json'),
        'cpu_accumulation_report_sha256':sha(ROOT/'reports/accumulation_validation.json'),
        'paired_initial_snapshot_sha256':initialization,'train_validation_token_checks':lengths,
        'resolved_gpu_arguments_sha256':sha(ROOT/'reports/resolved_gpu_arguments.json'),
        'run_ids_in_order':[x['run_id'] for x in plan['runs']],
        'environment':environment,'budget_seconds':budget['total_gpu_wall_time_limit_seconds'],
        'external_cost_limit_krw':0,'stage5_training_started':False,
        'test_length_audit_after_all_selections':True}
    write_new(root/'activation.json',record)
    print(json.dumps({'activation':'passed','protocol_sha256':record['protocol_sha256'],
        'run_count':len(plan['runs']),'gpu':environment['gpu'],'max_tokens':lengths},indent=2))
if __name__=='__main__':main()

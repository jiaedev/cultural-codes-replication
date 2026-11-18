"""Execute the ten planned fits in fresh processes within the selected GPU budget."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    budget=json.loads((ROOT/'configs/execution_budget.json').read_text(encoding='utf-8'))
    plan=json.loads((ROOT/'reports/experimental_run_plan.json').read_text(encoding='utf-8'))
    spec=json.loads((ROOT/'configs/experimental_protocol.v1.json').read_text(encoding='utf-8'))
    if not spec['execution_ready'] or budget['protocol_id']!=plan['protocol_id'] or plan['protocol_id']!=spec['protocol_id']:
        raise RuntimeError('Plan, budget or activation not frozen')
    root=ROOT/spec['outputs']['root']
    if not (root/'activation.json').is_file():raise RuntimeError('Activation manifest absent')
    elapsed=0.0
    for run in plan['runs']:
        out=root/'runs'/run['run_id']
        attempts=out/'attempts.jsonl'
        if attempts.exists():
            entries=[json.loads(x) for x in attempts.read_text(encoding='utf-8').splitlines()]
            elapsed+=sum(float(x.get('elapsed_seconds',0)) for x in entries if x['status'] in ('complete','failed'))
    for run in plan['runs']:
        out=root/'runs'/run['run_id']
        if (out/'selected_checkpoint.json').exists():continue
        if elapsed>=budget['total_gpu_wall_time_limit_seconds']:
            raise RuntimeError('Declared GPU time ceiling reached; remaining runs stay incomplete')
        if (out/'attempts.jsonl').exists():
            raise RuntimeError('Interrupted run requires checkpoint inspection and explicit recovery: '+run['run_id'])
        command=[sys.executable,'-B',str(ROOT/'scripts/train_research.py'),'train',
                 '--condition',run['condition'],'--seed',str(run['seed'])]
        log=root/'logs'/f"{run['run_id']}.log"
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('x',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Run {run['run_id']} failed; inspect its attempt record and log")
        selected=json.loads((out/'selected_checkpoint.json').read_text(encoding='utf-8'))
        elapsed+=selected['elapsed_seconds']
        print(json.dumps({'run_id':run['run_id'],'status':'complete','elapsed_training_seconds_total':elapsed}),flush=True)
    print(json.dumps({'status':'all_ten_training_runs_complete','elapsed_training_seconds_total':elapsed}))
if __name__=='__main__':main()

"""Evaluate ten locked Validation selections on complete Test rows once."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from datetime import datetime, timezone
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from transformers import AutoTokenizer, DataCollatorWithPadding, RobertaForSequenceClassification
from cultural_codes.data import read_rows, convert_label
from cultural_codes.protocol import load_protocol, make_arguments, make_dataset, case_id
from cultural_codes.trainer import final_evaluate
from train_research import activation, sha, write_new

def lock_selections(spec,frozen,plan):
    base=ROOT/spec['outputs']['root']
    selected=[]
    for item in plan['runs']:
        out=base/'runs'/item['run_id']
        manifest=out/'selected_checkpoint.json'
        if not manifest.is_file():raise RuntimeError('All ten Validation selections must be complete before Test access')
        record=json.loads(manifest.read_text(encoding='utf-8'))
        if (record['run_id'],record['seed'],record['condition'])!=(item['run_id'],item['seed'],item['condition']):
            raise RuntimeError('Selection identity mismatch')
        if record['protocol_sha256']!=frozen['protocol_sha256']:
            raise RuntimeError('Selection protocol hash mismatch')
        checkpoint=out/record['checkpoint']
        if not checkpoint.is_dir() or sha(checkpoint/'model.safetensors')!=record['checkpoint_weight_sha256']:
            raise RuntimeError('Selected checkpoint changed')
        selected.append({'run_id':item['run_id'],'selection_manifest_sha256':sha(manifest),
                         'checkpoint_weight_sha256':record['checkpoint_weight_sha256']})
    lock={'created_at_utc':datetime.now(timezone.utc).isoformat(),
          'protocol_sha256':frozen['protocol_sha256'],'activation_sha256':sha(base/'activation.json'),
          'selected':selected}
    path=base/'selection_lock.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        for k in ('protocol_sha256','activation_sha256','selected'):
            if old[k]!=lock[k]:raise RuntimeError('Locked selections changed')
    else:write_new(path,lock)
    return path,selected

def main():
    spec=load_protocol(ROOT)
    frozen=activation(spec)
    plan=json.loads((ROOT/'reports/experimental_run_plan.json').read_text(encoding='utf-8'))
    if len(plan['runs'])!=10 or plan['protocol_id']!=spec['protocol_id']:
        raise RuntimeError('Unexpected run plan')
    root=ROOT/spec['outputs']['root']
    lock,selections=lock_selections(spec,frozen,plan)
    source=ROOT/spec['data']['directory']/'test.csv'
    if sha(source)!=spec['data']['splits']['test']['sha256']:
        raise RuntimeError('Test source bytes changed')
    rows=read_rows(source)
    if len(rows)!=478:raise RuntimeError('Test row count mismatch')
    labels=[convert_label(r['output']) for r in rows]
    identifiers=[case_id(spec,'test',i) for i in range(1,479)]
    if len(set(identifiers))!=478:raise RuntimeError('Test IDs collided')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/spec['model']['local_config'],local_files_only=True,use_fast=True)
    tokenizer.padding_side=tokenizer.truncation_side='right'
    datasets={condition:make_dataset(spec,[r[column] for r in rows],labels,tokenizer)
              for condition,column in spec['scope']['conditions'].items()}
    # Both full Test inputs are length-audited under the frozen rule before any model scores them.
    results=[]
    for item in plan['runs']:
        out=root/'runs'/item['run_id']
        selected=json.loads((out/'selected_checkpoint.json').read_text(encoding='utf-8'))
        pred_path=out/'final_test_predictions.jsonl'
        metric_path=out/'final_test_metrics.json'
        if pred_path.exists() or metric_path.exists():
            raise RuntimeError('Existing final evaluation must be audited; no automatic overwrite: '+item['run_id'])
        checkpoint=out/selected['checkpoint']
        model=RobertaForSequenceClassification.from_pretrained(checkpoint,local_files_only=True)
        args=make_arguments(spec,item['condition'],item['seed'])
        started=perf_counter()
        result=final_evaluate(model,args,datasets[item['condition']],
                              DataCollatorWithPadding(tokenizer,padding='longest'))
        logits=np.asarray(result.predictions)
        references=np.asarray(result.label_ids)
        if logits.shape!=(478,2) or not np.isfinite(logits).all() or not np.array_equal(references,labels):
            raise RuntimeError('Prediction count, finite logits or reference order failed')
        predictions=np.argmax(logits,axis=1)
        if not np.isin(predictions,[0,1]).all():raise RuntimeError('Unexpected prediction class')
        f1=float(f1_score(labels,predictions,labels=[0,1],average='macro',zero_division=0))
        acc=float(accuracy_score(labels,predictions))
        precision,recall,class_f1,support=precision_recall_fscore_support(labels,predictions,labels=[0,1],zero_division=0)
        matrix=confusion_matrix(labels,predictions,labels=[0,1])
        elapsed=round(perf_counter()-started,3)
        record={'run_id':item['run_id'],'condition':item['condition'],'seed':item['seed'],
                'status':'complete','selected_epoch':selected['selected_epoch'],
                'selected_checkpoint':selected['checkpoint'],
                'selected_checkpoint_weight_sha256':selected['checkpoint_weight_sha256'],
                'validation_macro_f1':selected['validation_macro_f1'],
                'test_macro_f1':f1,'test_accuracy':acc,'evaluated_rows':478,
                'runtime_seconds':elapsed,'protocol_sha256':frozen['protocol_sha256'],
                'environment_lock_sha256':frozen['gpu_lock_sha256'],
                'selection_lock_sha256':sha(lock),
                'per_class':{name:{'precision':float(precision[k]),'recall':float(recall[k]),
                     'f1':float(class_f1[k]),'support':int(support[k])} for k,name in enumerate(('False','True'))},
                'confusion_matrix_labels_0_1':matrix.tolist(),
                'prediction_rule':'argmax over finite logits; exact tie selects 0'}
        with pred_path.open('x',encoding='utf-8') as stream:
            for i,(row,pred,score) in enumerate(zip(rows,predictions,logits),1):
                entry={'run_id':item['run_id'],'protocol_sha256':frozen['protocol_sha256'],
                       'seed':item['seed'],'condition':item['condition'],'case_id':identifiers[i-1],
                       'data_row_1based':i,'source_export_index':row[''],
                       'input_sha256':hashlib.sha256(row[item['input_column']].encode()).hexdigest(),
                       'label_id':labels[i-1],'prediction_id':int(pred),
                       'logit_0':float(score[0]),'logit_1':float(score[1])}
                stream.write(json.dumps(entry)+'\n')
        write_new(metric_path,record)
        results.append(record)
        del model
        torch.cuda.empty_cache()
        print(json.dumps({'run_id':item['run_id'],'test_macro_f1':f1,'evaluated_rows':478}),flush=True)
    by={(r['seed'],r['condition']):r for r in results}
    pairs=[{'seed':seed,'A_macro_f1':by[seed,'A']['test_macro_f1'],
            'B_macro_f1':by[seed,'B']['test_macro_f1'],
            'B_minus_A':by[seed,'B']['test_macro_f1']-by[seed,'A']['test_macro_f1']}
            for seed in spec['scope']['seeds']]
    summary={'protocol_id':spec['protocol_id'],'protocol_sha256':frozen['protocol_sha256'],
             'runs_complete':10,'test_evaluations_complete':10,
             'conditions':{c:{'mean_macro_f1':statistics.mean(by[s,c]['test_macro_f1'] for s in spec['scope']['seeds']),
                'sd_macro_f1_ddof1':statistics.stdev(by[s,c]['test_macro_f1'] for s in spec['scope']['seeds'])} for c in ('A','B')},
             'paired_B_minus_A':{'values':pairs,'mean':statistics.mean(x['B_minus_A'] for x in pairs),
                 'sd_ddof1':statistics.stdev(x['B_minus_A'] for x in pairs)},
             'interpretation_scope':'descriptive paired-seed comparison; seed SD is not Test-sample uncertainty'}
    for filename,fields,data in [
        ('per_run.csv',['run_id','condition','seed','status','selected_epoch','selected_checkpoint','validation_macro_f1','test_macro_f1','test_accuracy','evaluated_rows','runtime_seconds','protocol_sha256','environment_lock_sha256'],results),
        ('paired_differences.csv',['seed','A_macro_f1','B_macro_f1','B_minus_A'],pairs)]:
        with (root/filename).open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(data)
    write_new(root/'summary.json',summary)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

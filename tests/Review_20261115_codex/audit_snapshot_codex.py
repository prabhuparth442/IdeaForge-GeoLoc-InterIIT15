"""Read-only source inventory and independent rescore of delivered T10/T11 decisions."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
WORKSPACE=ROOT.parent
OUT=HERE/'results'


def main():
    old=list(csv.DictReader((ROOT/'tests/Review_20261113_codex/results/source_manifest_codex.csv').open()))
    changed=[]
    for r in old:
        p=WORKSPACE/r['tree']/r['path']
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:
            changed.append(str(p.relative_to(WORKSPACE)))
    sources=[]
    folders=[ROOT/'src',ROOT/'train',WORKSPACE/'Main_Pipeline_Headless',WORKSPACE/'Submission_Files/myteam/code']
    folders += [ROOT/'tests'/f'Test_Iteration_{i}' for i in range(8,12)]
    for folder in folders:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix in ('.py','.sh','.md','.ipynb','.json'):
                sources.append(dict(path=str(p.relative_to(WORKSPACE)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
    for p in ROOT.glob('*.py'):
        sources.append(dict(path=str(p.relative_to(WORKSPACE)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
    (OUT/'current_source_manifest_codex.json').write_text(json.dumps(sources,indent=2))
    state={}
    for p in [ROOT,WORKSPACE/'Main_Pipeline_Headless']:
        state[p.name]={cmd:subprocess.run(['git','-C',str(p)]+args,capture_output=True,text=True).stdout.strip()
                       for cmd,args in [('branch',['branch','--show-current']),('head',['rev-parse','HEAD']),('status',['status','--short'])]}
    (OUT/'state_codex.json').write_text(json.dumps(dict(git=state,previous_manifest_files=len(old),changed_since_previous=changed),indent=2))
    scores=[]
    for iteration,prefix in [(10,'it6_808'),(11,'it2_909')]:
        parent=ROOT/'tests'/f'Test_Iteration_{iteration}'/'results'
        final=pd.read_csv(parent/(prefix+'_final.csv'))
        for _,r in final.iterrows():
            path=parent/f"{prefix}_{r['var']}_{r['rule']}.csv"
            table=pd.read_csv(path)
            assert not table.duplicated(['data','photo']).any()
            calculated=dict(score=float(table.pts.mean()*6),wrong=int((table.pts<0).sum()),
                            answered=int((table.stage!='skip').sum()),queries=float(table.n_queries.mean()))
            assert len(table)==300
            assert abs(calculated['score']-r['score'])<.011
            assert calculated['wrong']==r['wrong'] and calculated['answered']==r['answered']
            scores.append(dict(iteration=iteration,variant=r['var'],rule=r['rule'],n=len(table),**calculated,
                               manifest=str(path.relative_to(ROOT))+'.manifest.json',sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (OUT/'independent_rescore_codex.json').write_text(json.dumps(scores,indent=2))
    print('Previous source manifest:',len(old),'files;',len(changed),'changed since previous review')
    print('Re-scored',len(scores),'saved policy outputs; all summary scores/answer counts/wrong counts agree; 300 photos each')
    print('Current snapshot:',len(sources),'source/report/notebook/manifest files')
    # Count assertion output, not every PASS occurrence in warning/summary prose.
    log=(OUT/'aggregate_codex.log').read_text()
    passed=sum(line.startswith('PASS ') for line in log.splitlines())
    failed=[line for line in log.splitlines() if line.startswith('FAIL ')]
    print('Aggregate currently recorded:',passed,'PASS lines; failures:',failed)


if __name__=='__main__':
    main()

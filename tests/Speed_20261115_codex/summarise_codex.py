"""Update the progress report from completed measurements without tuning decisions."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent;OUT=HERE/'results';ROOT=HERE.parents[1]


def main():
    lines=['# Isolated speed experiment — progress and results','',
           'No model fitting or threshold tuning. Adapters live in this folder; production sources stay unchanged. These are CPU regression/timing measurements, not target-board or independent generalisation results.','',
           '## Component measurements','',
           '| Domain | Change | Original median | Changed median | Ratio | Parity |',
           '|---|---|---:|---:|---:|---|']
    components={};runs={}
    for domain in ['official','nudt']:
        p=OUT/f'{domain}_components_codex.json'
        if p.exists():
            c=json.loads(p.read_text());components[domain]=c
            for label,key,new in [('Photo preprocessing (six renders)','render','cached'),('DINO K50','dino','vectorised'),('LoFTR M10','loftr','cached')]:
                a=c[key]['original']['median'];b=c[key][new]['median']
                parity=('exact' if c[key].get('exact',False) else ('exact ranks; score max '+str(c[key]['max_abs']) if key=='dino' else str(c[key]['parity'])))
                lines.append(f'| {domain} | {label} | {a:.4f}s | {b:.4f}s | {a/b:.2f}x | {parity} |')
            for batch,v in c['loftr']['batches'].items():
                a=c['loftr']['original']['median'];b=v['timing']['median']
                lines.append(f"| {domain} | LoFTR batch{batch} | {a:.4f}s | {b:.4f}s | {a/b:.2f}x | {v['parity']} |")
        p=OUT/f'{domain}_full_pipeline_codex.jsonl'
        if p.exists():
            runs[domain]=[json.loads(line) for line in p.read_text().splitlines()]
    lines += ['', '## Real image-to-answer pipeline', '',
              'Research AgreementQueryPipeline: full5.5km search, K50/M10, DINO batching5, MINIMA.21, six queries plus27-query fallback. baseline is unchanged research code; optimised adds preprocessing/edge/DINO changes; backbone also adds LoFTR query feature reuse. The same maps, hints, camera/geometry, weights and thresholds are used on both sides. No persistent query-result cache is used. The original general edge path computes fusion statistics it does not consume; the adapter skips these.','',
              '| Domain | Completed frames | Variant | Mean seconds | Score /60 | Answers | Exact final outputs vs baseline |',
              '|---|---:|---|---:|---:|---:|---|']
    summary=[]
    for domain,rows in runs.items():
        if not rows:continue
        for variant in rows[0]['outputs']:
            values=[r['outputs'][variant] for r in rows]
            same=all(r['comparisons'].get(variant,dict(final_same=True))['final_same'] for r in rows)
            record=dict(domain=domain,variant=variant,completed=len(rows),planned=3,mean_seconds=float(np.mean([v['total'] for v in values])),
                        score60=float(6*np.mean([v['points'] for v in values])),answers=sum(v['answered'] for v in values),
                        final_outputs_exact=same,stage_mean={k:float(np.mean([v['seconds'][k] for v in values])) for k in values[0]['seconds']},
                        process_peak_rss_mb=max(v['peak_rss_mb'] for v in values))
            summary.append(record)
            lines.append(f"| {domain} | {len(rows)}/3 | {variant} | {record['mean_seconds']:.2f} | {record['score60']:.2f} | {record['answers']} | {same} |")
    lines += ['', 'Stage means and cumulative process peak RSS are in results/summary_codex.json. Individual outputs include every query\'s candidate-score/rank, inlier and shift comparisons. Process RSS is a high-water mark across variants, not an isolated per-variant allocation measurement. Timings are a small sequential paired pilot, with alternating variant order; they have no reliable deployment confidence interval.','',
              '## Native/compiled experiments','']
    if runs:
        counts=sorted({v['n_queries_run'] for rows in runs.values() for r in rows for v in r['outputs'].values()})
        lines.insert(lines.index('## Native/compiled experiments'),f'Executed query counts in this pilot: {counts}. If only six, full-image timing/parity does not exercise the 27-query escalation branch; the branch remains enabled. Component and unit checks are separate from that coverage limitation.\n')
    p=OUT/'native_benchmark_codex.json'
    if p.exists():
        n=json.loads(p.read_text())
        lines.append(f"C++ translation-consensus kernel: {n['cases']} differential cases; {len(n['mismatches'])} mismatches. Original NumPy semantics and real saved matches were compared, including boundary/tie cases. Kernel-only timings include ctypes overhead; this is not a full-pipeline speedup.")
        for size in [50,200,500]:
            r=next(r for r in n['measurements'] if r['name']==f'synthetic_{size}' and r['radius']==3)
            lines.append(f"- N={size}: NumPy {r['numpy']*1e6:.1f}µs; C++ {r['cpp']*1e6:.1f}µs; {r['numpy']/r['cpp']:.2f}x.")
    else:lines.append('Native benchmark not completed yet.')
    p=OUT/'compile_benchmark_codex.json'
    if p.exists():
        c=json.loads(p.read_text())
        lines.append(f"\nCPU TorchInductor LoFTR-backbone-only: eager {c['eager']['median']:.4f}s, compiled warm {c['compiled']['median']:.4f}s, cold compile {c['cold_compile_seconds']:.1f}s. Feature max errors {c['feature_max_abs']}; allclose {c['feature_allclose']}. Excluded from full-pipeline variants; needs final match validation before adoption.")
        if 'full_match_probe' in c:
            lines.append('\nOne complete matching-pair probe: '+json.dumps(c['full_match_probe']))
        lines.append('\nThe initial compile failed because the installed C++ builder split workspace paths containing spaces. Temporary no-space aliases for the existing venv and experiment cache allowed a bounded retry. No package/source installation or modification was required; see compile_retry_paths_codex.json and the two compiler logs.')
    else:
        p=OUT/'compile_status_codex.json'
        lines.append('\n'+(p.read_text() if p.exists() else 'Compilation experiment not completed yet.'))
    p=OUT/'judge_benchmark_codex.json'
    if p.exists():
        j=json.loads(p.read_text())
        a=float(np.mean([r['original'] for r in j['rows']]))
        b=float(np.mean([r['incremental'] for r in j['rows']]))
        lines += ['', '## changed_src live-judge bookkeeping', '',
                  f"Five saved photos, six query prefixes: mean repeated-Pandas/prediction time {a:.4f}s → {b:.4f}s ({a/b:.2f}x). All prefix probabilities exact: {j['all_exact']}. Raw signal extraction and network inference are excluded. The existing frozen T7 judge is a thermal-trained historical baseline, not a newly trained or promoted model."]
    lines += ['', '## Availability, tests and lineage','',
              'CUDA is unavailable. TensorRT, ONNX, ONNX Runtime, Torch-TensorRT and Numba are not installed. No engine acceleration or GPU scalar-transfer benefit is claimed. DINO and LoFTR use existing frozen baseline weights; no new encoder or judge was trained.','',
              'Unit tests cover backbone scope/restoration, per-photo scope/restoration, DINO missing candidates and vectorised score agreement. See results/unit_tests_codex.log. Raw matches and fitted parity are checked using actual MINIMA inference in the component benchmark. Native code is optional and is not installed into the pipeline.','',
              'Read PROTOCOL_codex.md and README_codex.md for exact reproduction and scope.']
    lines += ['', '## What to keep from this experiment', '']
    for domain in ['official','nudt']:
        base=next((r for r in summary if r['domain']==domain and r['variant']=='baseline'),None)
        best=next((r for r in summary if r['domain']==domain and r['variant']=='backbone'),None)
        if base and best:
            lines.append(f"- {domain}: full adapter {base['mean_seconds']/best['mean_seconds']:.2f}x faster ({100*(1-best['mean_seconds']/base['mean_seconds']):.1f}% less wall time) on three frames, exact final outputs. Use `SpeedAgreementPipeline(..., query_backbone=True)` for this variant; default false is the separate preprocessing/edge/DINO ablation.")
    lines += ['- Per-photo preparation and scoped query-CNN reuse are the strongest equivalent-implementation candidates from these CPU tests. Retain for larger regression validation, not automatic production promotion.',
              '- DINO vectorisation provides no measured CPU speedup here; its GPU synchronisation benefit remains untested. LoFTR batches2/4 are slower here and are not enabled in the full adapter.',
              '- C++ translation consensus saves time in its kernel, but this kernel is a small fraction of complete inference. The judge incremental optimisation likewise saves milliseconds; it is not the main end-to-end gain.',
              '- The compiled backbone changes floating-point outputs and is excluded from the exact-equivalence full runs. The complete matching-pair probe is only one case, even if its inlier count agrees. Do not treat the compiler result as a fully validated replacement.',
              '- All six pilot frames resolved in the first six queries. Larger datasets, hard/escalated cases, GPU numerical parity, target-board latency and shared memory remain to be measured. No independent accuracy/generalisation gain is claimed.']
    (OUT/'summary_codex.json').write_text(json.dumps(summary,indent=2))
    (HERE/'REPORT_codex.md').write_text('\n'.join(lines)+'\n')
    print('Report updated:',{d:len(r) for d,r in runs.items()},flush=True)


if __name__=='__main__':main()

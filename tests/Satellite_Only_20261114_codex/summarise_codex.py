"""Summarise every scheduled frame, including skips/failures; never retune."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
WORKSPACE=HERE.parents[2]
OUT=HERE/'results'


def main():
    summaries=[]
    details=[]
    diagnostics=[]
    rng=np.random.default_rng(141128)
    for domain in ['official','nudt']:
        manifest=json.loads((OUT/f'{domain}_evaluation_manifest_codex.json').read_text())
        rows=[json.loads(line) for line in (OUT/f'{domain}_pipeline_codex.jsonl').read_text().splitlines()]
        assert [r['id'] for r in rows]==[r['id'] for r in manifest['samples']], 'Incomplete or duplicated evaluation'
        base=np.array([r['scores']['baseline']['points'] for r in rows],float)
        n=len(rows)
        resamples=rng.integers(0,n,size=(10000,n))
        for policy in ['baseline','sat_veto','sat_select']:
            scores=[r['scores'][policy] for r in rows]
            points=np.array([s['points'] for s in scores],float)
            delta=6*(points-base)
            bootstrap=delta[resamples].mean(axis=1)
            summary=dict(domain=domain,policy=policy,photos=n,answers=sum(s['answered'] for s in scores),
                         wrong_over4sigma=sum(s['wrong_over4sigma'] for s in scores),penalised_over8sigma=sum(s['penalised_over8sigma'] for s in scores),
                         failures=sum('failure' in r for r in rows),score60=float(max(0,6*points.mean())),
                         paired_delta=float(delta.mean()),paired_photo_se=float(delta.std(ddof=1)/np.sqrt(n)),
                         paired_photo_bootstrap95=np.percentile(bootstrap,[2.5,97.5]).tolist(),
                         wall_seconds_mean=float(np.mean([r['wall_seconds'] for r in rows])),
                         cpu_over30s=sum(r['wall_seconds']>30 for r in rows),
                         encoder_seconds_mean=float(np.mean([r.get('encoder_seconds',0) for r in rows])),
                         warning='n=6 correlated regression frames; photo-level SE/bootstrap are descriptive, not cross-flight/domain uncertainty')
            summaries.append(summary)
        for r in rows:
            if r.get('candidates'):
                truth=next(s['truth_xy'] for s in manifest['samples'] if s['id']==r['id'])
                best_cosine=max(r['candidates'],key=lambda c:c['cosine'])
                err=math.hypot(best_cosine['x']-truth[0],best_cosine['y']-truth[1])
                diagnostics.append(dict(domain=domain,id=r['id'],highest_cosine=best_cosine['cosine'],
                                        highest_cosine_error_m=err,highest_cosine_wrong_over4sigma=err>4*r['sigma_m']))
            details.append(dict(domain=domain,id=r['id'],flight=r['flight'],
                                baseline_points=r['scores']['baseline']['points'],veto_points=r['scores']['sat_veto']['points'],
                                select_points=r['scores']['sat_select']['points'],seconds=r['wall_seconds'],
                                queries=r.get('baseline',{}).get('n_queries',0),failure=r.get('failure','')))
    changed=[]
    source_manifest=REPO/'tests/Review_20261113_codex/results/source_manifest_codex.csv'
    originals=list(csv.DictReader(source_manifest.open()))
    for row in originals:
        p=WORKSPACE/row['tree']/row['path']
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            changed.append(str(p))
    integrity=dict(checked=len(originals),changed=changed)
    (OUT/'source_integrity_codex.json').write_text(json.dumps(integrity,indent=2))
    (OUT/'summary_codex.json').write_text(json.dumps(summaries,indent=2))
    (OUT/'ranking_diagnostics_codex.json').write_text(json.dumps(diagnostics,indent=2))
    with (OUT/'per_photo_codex.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    training=json.loads((OUT/'FROZEN_TRAINING_MANIFEST_codex.json').read_text())
    lines=['# Satellite-only encoder: real end-to-end pilot', '',
           'This is an actual raw-drone-image → answer run, not a replay of previously saved candidate tables. All work is isolated in this folder. No production changes were promoted.', '',
           '## Results', '',
           '| Dataset | Decision | Photos | Answers | >4σ wrong | >8σ penalised | Score / 60 | Δ score |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['domain']} | {s['policy']} | {s['photos']} | {s['answers']} | {s['wrong_over4sigma']} | {s['penalised_over8sigma']} | {s['score60']:.2f} | {s['paired_delta']:+.2f} |")
    lines += ['', 'Scores include every scheduled frame; skips and failures receive zero. Wrong here means error >4 times the altitude/focal-length scoring sigma; >8 sigma receives −20 raw points. The scoring sigma (metres) is unrelated to statistical uncertainty in mean score.', '',
              '## Decision', '',
              'Do not promote these gates. The experiment demonstrates successful satellite-task learning but poor transfer of its candidate scores and calibrated acceptance gate to actual thermal views. Keeping a low wrong-answer count by discarding correct answers is not an improvement. No threshold was loosened after examining drone results. This rejects this specific small-model/gate configuration, not all possible satellite-only training approaches.', '',
              'Highest encoder-cosine candidate diagnostics (before the retained geometric safeguard):', '']
    for domain in ['official','nudt']:
        ds=[d for d in diagnostics if d['domain']==domain]
        lines.append(f"- {domain}: {sum(d['highest_cosine_wrong_over4sigma'] for d in ds)}/{len(ds)} highest-cosine candidates were >4σ wrong; maximum error {max(d['highest_cosine_error_m'] for d in ds):.1f} m.")
    lines += ['',
              '## What was trained', '',
              f"A {training['parameters']:,}-parameter CNN, randomly initialised. Only raw satellite scene 268234931 was read by training. Geographic stripes contained 512 training, 128 validation and 256 calibration crops with buffered, non-overlapping split footprints. No NUDT map was used to train or calibrate, so NUDT is a cross-map test for these new weights. Official is same-map adaptation, not an unseen-map claim.", '',
              f"Validation contrastive loss: {training['initial_validation_loss']:.4f} → {min(e['validation_loss'] for e in training['epochs']):.4f}. Selected checkpoint: lowest satellite validation loss over 12 epochs. Synthetic calibration retained {training['calibration']['retained_correct']}/256 correct positive sets with no accepted incorrect top selections or null sets among the 256 null sets. Frozen gate: cosine ≥ {training['calibration']['cosine']}, margin ≥ {training['calibration']['margin']}. This is finite-sample empirical calibration, not a certified thermal risk bound.", '',
              'The encoder uses polarity-invariant gradient channels; fixed synthetic appearance/geometric augmentations are not real thermal rendering. The full pipeline still contains existing frozen DINO/MINIMA weights with separate, not-certified-satellite-only pretraining. No F4/T7/old synthetic judge weights were used.', '',
              '## What was run', '',
              'The current headless pipeline was imported from Main_Pipeline_Headless. It ran projection → full 5.5 km edge search → DINO K50/M5 → MINIMA LoFTR (.21) → RANSAC → decision. Standard six-query search and the existing twelve additional-query fallback were enabled. The new decision variants shared the exact same candidate computations and baseline query scheduling. This is the headless bar8 baseline, not the research AgreementQueryPipeline K50/M10/27-query baseline.', '',
              'Official uses its existing memory-mapped map store. NUDT uses a new full-reference-map store at levels3/4, generated without drone locations. The adapter fixes NUDT camera width/height to 1280/1024 in this process and uses the historically established NUDT scale .8/no-attitude projection. No geometry was fitted in this experiment.', '',
              'The selected six images in each domain and synthetic 5 km priors were generated with fixed seeds, independent of outcomes. Only the evaluator sees truth; localisation receives image, camera, attitude, altitude and perturbed hint. These are regression datasets previously used by the project, not a newly blind flight acquisition.', '',
              '## Limits and timing', '',
              'CUDA was unavailable. The CPU accuracy run disabled the 22-second guard and used DINO batches of five to bound memory. This is not a deployment-time or 30-second-scored benchmark. The timing includes the small extra encoder; stage timing and encoder timing are separately logged. The host was shared with preprocessing/other work, and the negative-control run overlapped the final NUDT frame; latency is descriptive, not an isolated speed benchmark. Six frames cannot establish zero wrong-answer risk, optimal score, or meaningful cross-flight ±σ uncertainty.', '']
    for domain in ['official','nudt']:
        s=next(s for s in summaries if s['domain']==domain)
        lines.append(f"- {domain}: {s['wall_seconds_mean']:.2f} seconds/frame mean, added encoder {s['encoder_seconds_mean']:.3f} seconds/frame, {s['cpu_over30s']}/6 over 30 seconds, {s['failures']} execution failures.")
    lines += ['', 'Paired photo-level uncertainty (descriptive only, correlated frames; not a guarantee):', '']
    for s in summaries:
        if s['policy']=='baseline': continue
        lo,hi=s['paired_photo_bootstrap95']
        lines.append(f"- {s['domain']} {s['policy']}: Δ {s['paired_delta']:+.2f} ± {s['paired_photo_se']:.2f} points (one photo-level SE); percentile bootstrap 95% [{lo:+.2f}, {hi:+.2f}].")
    negative_file=OUT/'negative_control_codex.json'
    if negative_file.exists():
        negative=json.loads(negative_file.read_text())
        lines += ['', '## Separate negative control', '',
                  'Added after the valid-hint pilot had started, as a diagnostic only. The first preselected official frame was evaluated with its hint at the farthest inset map corner, placing its truth outside the search disc. This deliberately violates the normal ≤5 km hint assumption; it is excluded from the accuracy table and never used for fitting.', '',
                  f"Truth-to-hint distance: {negative['manifest']['truth_distance_m']:.1f} m. Queries executed: {negative['baseline']['n_queries']}; fallback used: {negative['baseline']['second_pass']}. CPU seconds: {negative['baseline']['total_seconds']:.1f}.", '']
        for policy,s in negative['scores'].items():
            lines.append(f"- {policy}: {'ANSWERED' if s['answered'] else 'abstained'}; >4σ wrong: {s['wrong_over4sigma']}.")
        lines.append('\nOne absent-truth control exercises rejection behavior but cannot establish a false-accept probability.')
    lines += ['', '## Integrity and reproduction', '',f"Original source hashes checked: {integrity['checked']}; changed: {len(changed)}. Seven isolated unit tests cover split-footprint separation, frozen hashes, polarity invariance, conservative gates, ambiguity, independent-query agreement and failure scoring. Weights, thresholds and training source hashes remain frozen.", '',
              'See README_codex.md for commands; results/ contains per-photo candidates, stage timings, manifests, training log, calibration tasks, test log and summary JSON. The large generated NUDT .npy cache is ignored by git and can be rebuilt or removed with this experiment folder.']
    (HERE/'REPORT_codex.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summaries,indent=2))
    print('original source integrity',integrity)


if __name__=='__main__':
    main()

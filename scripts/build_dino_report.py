"""Build committed, screenshot-free report figures from curated aggregate metrics.

Default: regenerate PNGs using the committed JSON only.
To refresh the curated snapshot, supply --bundle EXTRACTED_ROOT --audit AUDIT_JSON.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO/'docs/results/dino'
SNAPSHOT = ROOT/'metrics-v1.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def import_metrics(bundle, audit_path):
    audit = read(audit_path)
    heads = bundle/'outputs/dino_heads'
    manifest_path = heads/'training_windows.json'
    manifest = read(manifest_path)
    baseline = {name: read(bundle/f'outputs/fullpage/validation_metrics/{name}.metrics.json')
                for name in ('dinov2', 'dinov3', 'pixel')}
    assert audit['recomputed_heads'] == 36 and manifest['version'] == 1
    assert all(v['n'] == 1033 for v in baseline.values())
    snapshot = dict(format_version=1, experiment='Design2Code DINO A/B/C v1', recorded='2026-11-14',
        interpretation='Validation-selected checkpoints and thresholds; no independent test estimate',
        counts=dict(train_pages=390, val_pages=69, train_pairs=5909, val_pairs=1033,
                    windows=11818, seeds=[490,491,492], steps=[500,1000,2000]),
        fingerprints={k: manifest[k] for k in ('metadata_sha256','split_sha256')},
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        baselines=baseline, models=audit['models'], selected_steps=audit['selected'],
        breakdowns=audit['breakdowns'], checkpoint_curves={}, learning={}, runtime={}, provenance={}, selected_scorer_metrics={})
    for model in ('dinov2','dinov3'):
        selection=read(heads/f'{model}_validation/selection.json')
        for condition in ('B','C'):
            snapshot['selected_scorer_metrics'][f'{model}_{condition}']={
                seed: entry['metrics'] for seed,entry in selection['selected'][condition]['seeds'].items()}
        snapshot['checkpoint_curves'][model]=[{k:r[k] for k in
            ('condition','step','auroc_mean','auroc_std','benign_fpr95_mean')}
            for r in audit['checkpoint_curves'][model]]
        for condition in ('B','C'):
            for seed in (490,491,492):
                root=heads/f'{model}_runs/{condition}/seed{seed}'
                key=f'{model}_{condition}_{seed}'
                snapshot['learning'][key]=[json.loads(line) for line in (root/'learning.jsonl').read_text().splitlines()]
                snapshot['runtime'][key]=read(root/'complete.json')
        config=read(heads/f'{model}_runs/B/seed490/config.json')
        snapshot['provenance'][model]=dict(revision=config['cache']['model_revision'],
            backbone=config['cache']['backbone'], packages=config['provenance']['packages'],
            git_commit=config['provenance']['git_commit'], dirty=config['provenance']['dirty'],
            sources=config['provenance']['sources'])
    SNAPSHOT.write_text(json.dumps(snapshot,indent=2,allow_nan=False)+'\n', encoding='utf-8')


def scorer_tables(data):
    """Publish every scalar metric and recall breakdown from the shared scorer."""
    entries=[(name,'pixel' if name=='pixel' else 'A',[data['baselines'][name]])
             for name in ('dinov2','dinov3','pixel')]
    entries += [(name.split('_')[0],name.split('_')[1],
                 [values[str(seed)] for seed in (490,491,492)])
                for name,values in data['selected_scorer_metrics'].items()]
    def get(row,path):
        for key in path.split('/'):
            if not isinstance(row,dict): return None
            row=row.get(key)
        return row
    metrics=[('Pairs (n)','n'),('AUROC: regression / benign','auroc/regression_vs_benign'),
        ('AUROC: regression / all negatives','auroc/regression_vs_all_negatives'),
        ('Benign FPR at 95% recall','benign_fpr_at_95_recall/fpr'),
        ('95%-recall operating threshold','benign_fpr_at_95_recall/threshold'),
        ('Achieved recall at that threshold','benign_fpr_at_95_recall/recall'),
        ('Max-F1 decision threshold','threshold'),('Precision','detection/precision'),
        ('Recall','detection/recall'),('F1','detection/f1'),('Accuracy','detection/accuracy'),
        ('Benign FPR at max-F1 threshold','false_positive_rate/benign'),
        ('No-change FPR at max-F1 threshold','false_positive_rate/no_change'),
        ('Type macro-F1','type/macro_f1'),('Pointing accuracy','localization/pointing_accuracy'),
        ('Mean box IoU','localization/mean_iou'),('Regressions with point predictions','localization/n_point'),
        ('Regressions with box predictions','localization/n_box')]
    groups={group:sorted({key for _,_,runs in entries for r in runs
                         for key in r.get('recall_by',{}).get(group,{})})
            for group in ('family','magnitude','reflow')}
    def fmt(value,path):
        if value is None: return '—'
        if path in ('n','localization/n_point','localization/n_box'): return str(int(value))
        return f'{value:.9g}' if 'threshold' in path else f'{value:.4f}'
    exported=[]
    def table(spec):
        lines=['| Metric | '+' | '.join('Pixel' if m=='pixel' else f'{m} {c}' for m,c,_ in entries)+' |',
               '|---|'+'---:|'*len(entries)]
        for title,path in spec:
            cells=[]
            for model,condition,runs in entries:
                vals=[get(r,path) for r in runs]
                if any(v is None for v in vals):
                    mean,sd=None,None
                    cell='—'
                else:
                    mean=float(np.mean(vals));sd=float(np.std(vals,ddof=1)) if len(vals)>1 else None
                    cell=fmt(mean,path)
                    if sd is not None and path not in ('n','localization/n_point','localization/n_box'):
                        cell+=' ± '+fmt(sd,path)
                cells.append(cell)
                exported.append(dict(model=model,condition=condition,metric=path,
                    mean='' if mean is None else mean,seed_sd='' if sd is None else sd,runs=len(runs)))
            lines.append('| '+title+' | '+' | '.join(cells)+' |')
        return '\n'.join(lines)
    content=['## Complete shared-scorer metrics',
        'All seven configurations below use the same 1,033 validation pairs: '
        '826 regressions, 138 benign pairs and 69 no-change pairs. B/C select update '
        '2,000 and show **mean ± sample SD across seeds 490/491/492**; A/pixel are '
        'single runs. Counts are per run, not totals pooled over seeds. Scorer '
        'format is version 1; decision thresholds maximize F1 on validation.',
        table(metrics),
        'Detection precision/recall/F1/accuracy and class-specific FPRs use each '
        'run\'s **max-F1 threshold**. FPR@95R uses its own threshold; ties can make '
        'achieved recall exceed 95%. Pointing is evaluated on regressions independently '
        'of either threshold. Dashes mean unavailable, not zero: these models do '
        'not predict mutation classes or boxes. AUROC includes both requested '
        'negative definitions; identity controls make the all-negative version easier.',
        '**Threshold summaries are descriptive, not deployable thresholds.** '
        'Each seed has its own selected value; exact values are below and in '
        '`metrics-v1.json`. Extra significant digits prevent near-zero/near-one '
        'thresholds from appearing equal after rounding.',
        '### Exact per-seed operating thresholds',
        '| Model | Condition | Seed | Max-F1 threshold | 95%-recall threshold | Achieved recall at 95%-recall threshold |',
        '|---|---|---:|---:|---:|---:|']
    for model,condition,runs in entries:
        for index,r in enumerate(runs):
            seed=str((490,491,492)[index]) if len(runs)>1 else '—'
            content.append(f"| {model} | {condition} | {seed} | {r['threshold']:.17g} | {r['benign_fpr_at_95_recall']['threshold']:.17g} | {r['benign_fpr_at_95_recall']['recall']:.6f} |")
    content += ['### Shared-scorer recall breakdowns',
        'These are regression recalls at each run\'s max-F1 threshold. They are '
        'not operator AUROCs or localization scores. `reflow=True` means the mutation '
        'propagates layout changes; `False` denotes non-reflow examples.',
        table([(f'Recall: {group} / {key}',f'recall_by/{group}/{key}')
               for group,keys in groups.items() for key in keys])]
    with (ROOT/'shared-scorer-v1.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(exported[0]));writer.writeheader();writer.writerows(exported)
    content += ['[Download all shared-scorer metrics and breakdowns as CSV](shared-scorer-v1.csv).']
    report=ROOT/'trained-validation-results.md'
    start,end='<!-- shared-scorer:start -->','<!-- shared-scorer:end -->'
    # Keep the exact-threshold header, separator and data rows contiguous.
    threshold_end=6+2+sum(len(runs) for _,_,runs in entries)
    block=(start+'\n\n'+'\n\n'.join(content[:6])+'\n\n'
           +'\n'.join(content[6:threshold_end])+'\n\n'
           +'\n\n'.join(content[threshold_end:])+'\n\n'+end+'\n\n')
    text=report.read_text(encoding='utf-8')
    if start in text:
        before,rest=text.split(start,1);_,after=rest.split(end,1)
        text=before+block+after.lstrip('\n')
    else:
        text=text.replace('## Artifact and metric verification',block+'## Artifact and metric verification',1)
    report.write_text(text,encoding='utf-8')


def plots(data):
    out=ROOT/'figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    labels=['v2 A','v3 A','Pixel','v2 B','v2 C','v3 B','v3 C']
    base=[data['baselines'][m] for m in ('dinov2','dinov3','pixel')]
    learned=[data['models'][m] for m in ('dinov2_B','dinov2_C','dinov3_B','dinov3_C')]
    colors=['#aeb6bf']*3+['#2166ac','#91bfdb','#d95f02','#fdb863']
    rows=[]
    for model,b in zip(('dinov2','dinov3','pixel'),base):
        rows.append(dict(model=model,condition='pixel' if model=='pixel' else 'A',
            auroc_mean=b['auroc']['regression_vs_benign'],auroc_seed_sd='',
            benign_fpr95_mean=b['benign_fpr_at_95_recall']['fpr'],
            pointing_mean=b['localization']['pointing_accuracy'],pointing_seed_sd='',
            f1_mean=b['detection']['f1'],recall_mean=b['detection']['recall'],
            benign_fpr_at_max_f1_mean=b['false_positive_rate']['benign']))
    for name in ('dinov2_B','dinov2_C','dinov3_B','dinov3_C'):
        model,condition=name.split('_');v=data['models'][name]
        rows.append(dict(model=model,condition=condition,auroc_mean=v['auroc']['mean'],auroc_seed_sd=v['auroc']['sd'],
            benign_fpr95_mean=v['benign_fpr95']['mean'],pointing_mean=v['pointing']['mean'],pointing_seed_sd=v['pointing']['sd'],
            f1_mean=v['f1']['mean'],recall_mean=v['recall']['mean'],benign_fpr_at_max_f1_mean=v['benign_fpr_f1']['mean']))
    with (ROOT/'metrics-v1.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    specs=[('auroc',lambda v:v['auroc']['regression_vs_benign'],'Reg/benign AUROC (higher is better)'),
           ('benign_fpr95',lambda v:v['benign_fpr_at_95_recall']['fpr'],'Benign FPR at 95% recall (lower is better)'),
           ('pointing',lambda v:v['localization']['pointing_accuracy'],'Pointing accuracy (higher is better)')]
    for ax,(key,getter,title) in zip(axes,specs):
        values=[getter(v) for v in base]+[v[key]['mean'] for v in learned]
        sd=[0]*3+[v[key]['sd'] for v in learned]
        ax.bar(labels,values,color=colors,yerr=sd,capsize=3)
        ax.set(ylim=(0,1.13),title=title)
        ax.tick_params(axis='x',rotation=40)
        for i,v in enumerate(values):
            ax.text(i,v+sd[i]+.025,f'{v:.3f}',ha='center',fontsize=8)
        if key=='auroc': ax.axhline(.5,color='grey',ls=':',lw=1)
    fig.suptitle('A/B/C v1: 69 validation pages, 1,033 pairs; B/C bars = 3-seed mean ± sample SD\nA and pixel are single training-free runs; all results are validation-selected',fontsize=12)
    fig.savefig(out/'comparison-v1.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,model in enumerate(('dinov2','dinov3')):
        for condition,color in (('B','#2166ac'),('C','#d95f02')):
            for seed,style in zip((490,491,492),('-','--',':')):
                rows=data['learning'][f'{model}_{condition}_{seed}']
                axes[0,col].plot([r['step'] for r in rows],[r['fixed_training_loss'] for r in rows],style,
                                  color=color,label=f'{condition}/{seed}',alpha=.8)
            rows=sorted([r for r in data['checkpoint_curves'][model] if r['condition']==condition],key=lambda r:r['step'])
            axes[1,col].errorbar([r['step'] for r in rows],[r['auroc_mean'] for r in rows],
                yerr=[r['auroc_std'] for r in rows],label=condition,color=color,capsize=4)
        axes[0,col].set(title=model.upper(),ylabel='Fixed training-window loss',xlabel='Update')
        axes[1,col].set(ylabel='Full-page validation AUROC',xlabel='Saved checkpoint update',ylim=(.3,1))
        axes[1,col].axhline(.5,color='grey',ls=':',lw=1)
        for ax in axes[:,col]: ax.legend(fontsize=8)
    fig.suptitle('Learning progress: lower training loss in C does not imply better benign rejection\nTraining-window diagnostics and full-page validation measure different distributions',fontsize=12)
    fig.savefig(out/'learning-v1.png',dpi=160);plt.close(fig)

    groups=data['breakdowns']['dinov2_B']
    ops=sorted(k for k in groups if k.startswith('operator/'))
    values=np.array([[data['breakdowns'][m][k]['auroc_mean'] for m in ('dinov2_B','dinov3_B')] for k in ops])
    fig,ax=plt.subplots(figsize=(7.5,7),layout='constrained')
    im=ax.imshow(values,vmin=.5,vmax=1,cmap='YlGnBu',aspect='auto')
    ax.set(xticks=[0,1],xticklabels=['DINOv2 B','DINOv3 B'],yticks=range(len(ops)),
        yticklabels=[f"{k.split('/')[1]} (n={groups[k]['n_regressions']})" for k in ops])
    for y in range(len(ops)):
        for x in range(2):
            ax.text(x,y,f'{values[y,x]:.3f}',ha='center',va='center',color='white' if values[y,x]>.9 else 'black')
    ax.set_title('Operator-specific regression/benign AUROC\nEach operator versus the same 138 benign pairs; 3-seed mean')
    fig.colorbar(im,ax=ax,shrink=.7,label='AUROC (shared scale)')
    fig.savefig(out/'operators-v1.png',dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(12,6),layout='constrained');ax.set(xlim=(0,12),ylim=(0,6));ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=color,edgecolor='#34495e'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    box(.3,4.4,11.4,1,'Paired screenshot windows -> frozen DINOv2 or DINOv3 patch features\nExisting split: 390 training pages / 69 validation pages','#edf2f7')
    box(.3,2.4,3.4,1.3,'A: training-free\nCorresponding-patch distances\nNo head-weight updates','#d5d8dc')
    box(4.3,2.4,3.4,1.3,'B: trained spatial head\n50% regression / 25% benign\n25% no-change','#c6dbef')
    box(8.3,2.4,3.4,1.3,'C: same head, no benign\n50% regression\n50% no-change','#fdd0a2')
    box(.3,.3,11.4,1.1,'Full-page validation: label-blind 448px windows, stride 224px\nDetection confidence + localization point | checkpoint/threshold selection on validation\nB/C: seeds 490, 491, 492; 2,000 updates; checkpoints 500 / 1,000 / 2,000','#e8f5e9')
    for x in (2,6,10):
        ax.annotate('',xy=(x,3.8),xytext=(x,4.3),arrowprops=dict(arrowstyle='->',color='#34495e'))
        ax.annotate('',xy=(x,1.5),xytext=(x,2.3),arrowprops=dict(arrowstyle='->',color='#34495e'))
    ax.set_title('Six DINO configurations: A/B/C for each backbone\nOnly B/C train on the training pages; pixel difference is an additional baseline',fontsize=13)
    fig.savefig(out/'experiment-design-v1.png',dpi=160);plt.close(fig)
    print(f'Generated 4 committed report figures in {out}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path)
    p.add_argument('--audit',type=Path)
    a=p.parse_args()
    if bool(a.bundle)!=bool(a.audit): p.error('--bundle and --audit must be supplied together')
    if a.bundle: import_metrics(a.bundle,a.audit)
    data=read(SNAPSHOT)
    scorer_tables(data)
    plots(data)


if __name__=='__main__': main()

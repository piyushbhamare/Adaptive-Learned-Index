#!/usr/bin/env python3
"""generate_report.py -- Regenerate report.md and all figures from CSV results."""
import sys, os, datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else 'results'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)

def load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        print(f"[WARN] {path} not found"); return pd.DataFrame()
    return pd.read_csv(path)

df    = load('benchmark_results.csv')
abl   = load('ablation_results.csv')
drift = load('drift_results.csv')

if df.empty:
    print("[ERROR] benchmark_results.csv missing or empty"); sys.exit(1)

IDX_ORDER = ['B-Tree','ALEX','PGM','RMI','NLI']
datasets  = ['Books','Facebook','WikiTS']
COLORS    = {'B-Tree':'#4e79a7','ALEX':'#f28e2b','PGM':'#59a14f','RMI':'#e15759','NLI':'#b07aa1'}

def q(ds, idx, n, col):
    r = df[(df['dataset']==ds)&(df['index']==idx)&(df['n_keys']==n)]
    return float(r[col].values[0]) if not r.empty else float('nan')

def safe(v, fmt='.1f'):
    return '—' if v!=v else format(v, fmt)

# ── Figures 1-4 ──────────────────────────────────────────────────────────────
# Fig 1: Mean query latency
fig, axes = plt.subplots(1,3,figsize=(14,4.5))
for ax, ds in zip(axes, datasets):
    sub = df[(df['dataset']==ds)&(df['index'].isin(IDX_ORDER))]
    x = range(len(IDX_ORDER)); w = 0.35
    for i, n in enumerate([100_000, 1_000_000]):
        vals = [float(sub[(sub['index']==idx)&(sub['n_keys']==n)]['mean_ns'].values[0])
                if not sub[(sub['index']==idx)&(sub['n_keys']==n)].empty else 0 for idx in IDX_ORDER]
        offset = (i-0.5)*w
        ax.bar([xi+offset for xi in x], vals, w, label=f'n={n//1000}K',
               color=[COLORS[i] for i in IDX_ORDER], alpha=0.85 if i==0 else 1.0, edgecolor='white')
    ax.set_title(ds, fontsize=11, fontweight='bold')
    ax.set_xticks(list(x)); ax.set_xticklabels(IDX_ORDER, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Mean Latency (ns)' if ds=='Books' else ''); ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('Point Query Mean Latency', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig1_query_latency.png'), dpi=150, bbox_inches='tight'); plt.close()

# Fig 2: Percentile profiles
fig, axes = plt.subplots(1,3,figsize=(14,4.5))
for ax, ds in zip(axes, datasets):
    sub = df[(df['dataset']==ds)&(df['n_keys']==1_000_000)&(df['index'].isin(IDX_ORDER))]
    for idx in IDX_ORDER:
        r = sub[sub['index']==idx]
        if r.empty: continue
        vals = [r['median_ns'].values[0], r['p95_ns'].values[0], r['p99_ns'].values[0], r['p999_ns'].values[0]]
        ax.plot([0,1,2,3], vals, marker='o', label=idx, color=COLORS[idx], linewidth=2, markersize=5)
    ax.set_title(f'{ds} n=1M', fontsize=11, fontweight='bold')
    ax.set_xticks([0,1,2,3]); ax.set_xticklabels(['p50','p95','p99','p99.9'])
    ax.set_yscale('log'); ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_ylabel('Latency (ns)' if ds=='Books' else ''); ax.legend(fontsize=7)
    ax.grid(alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('Latency Percentile Profiles (log scale)', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig2_latency_percentiles.png'), dpi=150, bbox_inches='tight'); plt.close()

# Fig 3: Insert latency
fig, axes = plt.subplots(1,3,figsize=(14,4.5))
for ax, ds in zip(axes, datasets):
    sub = df[(df['dataset']==ds)&(df['n_keys']==1_000_000)&(df['index'].isin(IDX_ORDER))]
    vals = [float(sub[sub['index']==idx]['insert_mean_ns'].values[0]) if not sub[sub['index']==idx].empty else 0 for idx in IDX_ORDER]
    bars = ax.bar(IDX_ORDER, vals, color=[COLORS[i] for i in IDX_ORDER], edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v*1.05, f'{v:.0f}', ha='center', va='bottom', fontsize=8)
    ax.set_title(ds, fontsize=11, fontweight='bold')
    ax.set_ylabel('Insert Mean Latency (ns)' if ds=='Books' else '')
    ax.set_yscale('log'); ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.tick_params(axis='x', rotation=30); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('Insert Latency (n=1M, log scale)', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig3_insert_latency.png'), dpi=150, bbox_inches='tight'); plt.close()

# Fig 4: Build time
fig, axes = plt.subplots(1,3,figsize=(14,4.5))
for ax, ds in zip(axes, datasets):
    sub = df[(df['dataset']==ds)&(df['n_keys']==1_000_000)&(df['index'].isin(IDX_ORDER))]
    vals = [float(sub[sub['index']==idx]['build_ms'].values[0]) if not sub[sub['index']==idx].empty else 0 for idx in IDX_ORDER]
    bars = ax.bar(IDX_ORDER, vals, color=[COLORS[i] for i in IDX_ORDER], edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.2, f'{v:.1f}ms', ha='center', va='bottom', fontsize=8)
    ax.set_title(ds, fontsize=11, fontweight='bold')
    ax.set_ylabel('Build Time (ms)' if ds=='Books' else ''); ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('Index Build Time (n=1M)', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig4_build_time.png'), dpi=150, bbox_inches='tight'); plt.close()

# Fig 5: Ablation
if not abl.empty:
    abl_cols = {'PGM(ref)':'#59a14f','NLI-Full':'#b07aa1','NLI-Linear':'#9c755f',
                'NLI-NoDrift':'#bab0ac','NLI-NoRepair':'#76b7b2','NLI-NoPW':'#ff9da7'}
    abl_cfgs = list(abl_cols.keys())
    fig, axes = plt.subplots(1,3,figsize=(14,4.5))
    for ax, ds in zip(axes, datasets):
        sub = abl[(abl['dataset']==ds)&(abl['n_keys']==1_000_000)]
        vals = [float(sub[sub['index']==c]['mean_ns'].values[0]) if not sub[sub['index']==c].empty else 0 for c in abl_cfgs]
        bars = ax.bar(range(len(abl_cfgs)), vals, color=[abl_cols[c] for c in abl_cfgs], edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+2, f'{v:.0f}', ha='center', va='bottom', fontsize=7, rotation=45)
        ax.set_title(ds, fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(abl_cfgs))); ax.set_xticklabels(abl_cfgs, rotation=35, ha='right', fontsize=7)
        ax.set_ylabel('Mean Latency (ns)' if ds=='Books' else '')
        ax.grid(axis='y', alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle('Ablation Study — NLI Component Contributions (n=1M)', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig5_ablation.png'), dpi=150, bbox_inches='tight'); plt.close()

# Fig 6: Memory
fig, axes = plt.subplots(1,3,figsize=(14,4.5))
for ax, ds in zip(axes, datasets):
    sub = df[(df['dataset']==ds)&(df['n_keys']==1_000_000)&(df['index'].isin(IDX_ORDER))]
    vals = [float(sub[sub['index']==idx]['memory_bytes'].values[0])/1e6 if not sub[sub['index']==idx].empty else 0 for idx in IDX_ORDER]
    bars = ax.bar(IDX_ORDER, vals, color=[COLORS[i] for i in IDX_ORDER], edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.1, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    ax.set_title(ds, fontsize=11, fontweight='bold'); ax.set_ylabel('Memory (MB)' if ds=='Books' else '')
    ax.tick_params(axis='x', rotation=30); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('Memory Footprint (n=1M)', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig6_memory.png'), dpi=150, bbox_inches='tight'); plt.close()

# Fig 7: Throughput heatmap
fig, ax = plt.subplots(figsize=(10,4))
sub = df[df['n_keys']==1_000_000]
mat = np.array([[float(sub[(sub['index']==idx)&(sub['dataset']==ds)]['throughput_mops'].values[0])
                 if not sub[(sub['index']==idx)&(sub['dataset']==ds)].empty else 0
                 for ds in datasets] for idx in IDX_ORDER])
im = ax.imshow(mat, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets, fontsize=11)
ax.set_yticks(range(len(IDX_ORDER))); ax.set_yticklabels(IDX_ORDER, fontsize=11)
for i in range(len(IDX_ORDER)):
    for j in range(len(datasets)):
        ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=10,
                color='black' if mat[i,j]<mat.max()*0.7 else 'white')
plt.colorbar(im, ax=ax, label='Mops/s')
ax.set_title('Throughput Heatmap (n=1M, Mops/s)', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig7_throughput_heatmap.png'), dpi=150, bbox_inches='tight'); plt.close()

print(f"[OK] {len(os.listdir(FIG))} figures in {FIG}")

# ── Write report ──────────────────────────────────────────────────────────────
lines = [
    "# NLI Benchmark Report",
    f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
    "**Indexes:** B-Tree, ALEX (microsoft/ALEX), PGM (gvinciguerra/PGM-index ε=64), RMI (1024 models), NLI v3.0  ",
    "**Datasets:** Books, Facebook, WikiTS (SOSD, 1M keys per run)  ",
    "**Workload:** 300K queries (80% hit), 50K inserts  ",
    "",
    "---",
    "",
    "## 1. Point Query Mean Latency (ns)",
    "",
]
for n, label in [(100_000,'100K'),(1_000_000,'1M')]:
    lines += [f"### n = {label}", "",
              "| Index | Books mean | Books p99 | FB mean | FB p99 | WikiTS mean | WikiTS p99 |",
              "|-------|--------:|--------:|-------:|-------:|----------:|----------:|"]
    for idx in IDX_ORDER:
        row = f"| {idx}"
        for ds in datasets:
            row += f" | {safe(q(ds,idx,n,'mean_ns'))} | {safe(q(ds,idx,n,'p99_ns'))}"
        lines.append(row + " |")
    lines.append("")

lines += ["---","","## 2. Throughput (Mops/s) — n=1M","",
          "| Index | Books | Facebook | WikiTS |",
          "|-------|------:|---------:|-------:|"]
for idx in IDX_ORDER:
    row = f"| {idx}"
    for ds in datasets: row += f" | {safe(q(ds,idx,1_000_000,'throughput_mops'),'.2f')}"
    lines.append(row + " |")

lines += ["","---","","## 3. Insert Latency (ns) — n=1M","",
          "| Index | Books | Facebook | WikiTS |",
          "|-------|------:|---------:|-------:|"]
for idx in IDX_ORDER:
    row = f"| {idx}"
    for ds in datasets: row += f" | {safe(q(ds,idx,1_000_000,'insert_mean_ns'))}"
    lines.append(row + " |")

lines += ["","---","","## 4. Build Time (ms) — n=1M","",
          "| Index | Books | Facebook | WikiTS |",
          "|-------|------:|---------:|-------:|"]
for idx in IDX_ORDER:
    row = f"| {idx}"
    for ds in datasets: row += f" | {safe(q(ds,idx,1_000_000,'build_ms'),'.1f')}"
    lines.append(row + " |")

if not abl.empty:
    lines += ["","---","","## 5. Ablation Study — n=1M","",
              "| Config | Books (ns) | Facebook (ns) | WikiTS (ns) |",
              "|--------|----------:|--------------:|------------:|"]
    for cfg in ['PGM(ref)','NLI-Full','NLI-Linear','NLI-NoDrift','NLI-NoRepair','NLI-NoPW']:
        row = f"| {cfg}"
        for ds in datasets:
            r = abl[(abl['index']==cfg)&(abl['dataset']==ds)&(abl['n_keys']==1_000_000)]
            v = float(r['mean_ns'].values[0]) if not r.empty else float('nan')
            row += f" | {safe(v,'.1f')}"
        lines.append(row + " |")

if not drift.empty:
    lines += ["","---","","## 6. Drift Detection","",
              "| Dataset | Drift Type | Detected | Delay | FP | FN |",
              "|---------|-----------|:--------:|------:|:--:|:--:|"]
    for _, r in drift.iterrows():
        det   = 'YES' if r['drift_detected']==1 else 'no'
        delay = str(r['detection_delay']) if r['drift_detected']==1 else '—'
        lines.append(f"| {r['dataset']} | {r['drift_type']} | {det} | {delay} | {int(r['is_false_positive'])} | {int(r['is_false_negative'])} |")
    tp = len(drift[(drift['drift_detected']==1)&(drift['is_false_positive']==0)])
    fp = len(drift[(drift['drift_detected']==1)&(drift['is_false_positive']==1)])
    fn = len(drift[(drift['drift_detected']==0)&(drift['is_false_negative']==1)])
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    lines += ["", f"**Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}**"]

lines += ["","---","","## 7. Figures","",
          "| File | Description |",
          "|------|-------------|",
          "| fig1_query_latency.png | Mean latency grouped bar chart |",
          "| fig2_latency_percentiles.png | Percentile profiles (log scale) |",
          "| fig3_insert_latency.png | Insert latency (log scale) |",
          "| fig4_build_time.png | Build time comparison |",
          "| fig5_ablation.png | Ablation component analysis |",
          "| fig6_memory.png | Memory footprint |",
          "| fig7_throughput_heatmap.png | Throughput heatmap |",
          "","---",
          "_All results from actual runtime measurements. ALEX: microsoft/ALEX (MIT). PGM: gvinciguerra/PGM-index (Apache 2.0)._"]

with open(os.path.join(OUT,'report.md'),'w') as f:
    f.write('\n'.join(lines))
print(f"[OK] report.md written ({len(lines)} lines)")

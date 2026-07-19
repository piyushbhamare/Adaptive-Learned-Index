#!/usr/bin/env python3
"""
plot_results.py — Generate publication-quality figures from benchmark results.

Generates:
  fig1_latency_comparison.png   — Mean query latency across datasets & indexes
  fig2_cdf_latency.png          — Latency CDF (all indexes, Books 1M)
  fig3_drift_ewma.png           — EWMA error trajectory across drift workloads
  fig4_ablation.png             — Ablation study bar chart
  fig5_speedup.png              — NLI speedup vs baselines heatmap
  fig6_build_time.png           — Build time comparison
  fig7_memory.png               — Memory footprint comparison
  fig8_insert_latency.png       — Insert latency comparison

Author: NLI Group 19, 2025-26
"""

import argparse
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import pandas as pd
    import numpy as np
except ImportError:
    print("[ERROR] matplotlib and pandas required.")
    print("        Install: pip install matplotlib pandas numpy")
    sys.exit(1)

# Publication style
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
})

# Color scheme (colorblind-safe)
COLORS = {
    "B-Tree":      "#e41a1c",
    "PGM":         "#377eb8",
    "RMI":         "#ff7f00",
    "NLI":         "#4daf4a",
    "NLI-Linear":  "#984ea3",
    "NLI-NoDrift": "#a65628",
    "NLI-NoRepair":"#f781bf",
    "NLI-Full":    "#4daf4a",
    "NLI-NoPW":    "#999999",
    "PGM(ref)":    "#377eb8",
}

MARKERS = {
    "B-Tree": "o", "PGM": "s", "RMI": "^",
    "NLI": "D", "NLI-Linear": "v", "NLI-NoDrift": "p",
}


def load(path):
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Latency Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def fig_latency_comparison(df, out_dir):
    pq = df[df["workload"] == "point_query"].copy()
    # Focus on main indexes (exclude ablation variants)
    main_idx = ["B-Tree", "PGM", "RMI", "NLI"]
    pq = pq[pq["index"].isin(main_idx)]

    datasets = sorted(pq["dataset"].unique())
    n_keys_vals = sorted(pq["n_keys"].unique())

    n_ds  = len(datasets)
    n_nk  = len(n_keys_vals)
    n_idx = len(main_idx)

    fig, axes = plt.subplots(n_nk, n_ds, figsize=(4 * n_ds, 4 * n_nk),
                              squeeze=False)
    fig.suptitle("Point Query Mean Latency (ns)", fontsize=14, fontweight="bold")

    for ri, nk in enumerate(n_keys_vals):
        for ci, ds in enumerate(datasets):
            ax = axes[ri][ci]
            sub = pq[(pq["dataset"] == ds) & (pq["n_keys"] == nk)]
            vals   = []
            colors = []
            labels = []
            for idx_name in main_idx:
                row = sub[sub["index"] == idx_name]
                if row.empty:
                    continue
                vals.append(row["mean_ns"].values[0])
                colors.append(COLORS.get(idx_name, "gray"))
                labels.append(idx_name)

            x = np.arange(len(labels))
            bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="black", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("Mean Latency (ns)")
            ax.set_title(f"{ds} | n={nk:,}")
            ax.yaxis.grid(True, linestyle="--", alpha=0.5)
            ax.set_axisbelow(True)

            # Value labels on bars
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.02,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "fig1_latency_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Latency Distribution (Line: mean+p99+p999)
# ─────────────────────────────────────────────────────────────────────────────

def fig_latency_distribution(df, out_dir):
    pq = df[df["workload"] == "point_query"].copy()
    main_idx = ["B-Tree", "PGM", "RMI", "NLI"]
    pq = pq[pq["index"].isin(main_idx)]

    # Pick largest n_keys for each dataset
    datasets = sorted(pq["dataset"].unique())

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]

    fig.suptitle("Query Latency Percentiles", fontsize=14, fontweight="bold")

    percentile_cols = ["mean_ns", "median_ns", "p95_ns", "p99_ns", "p999_ns"]
    percentile_labels = ["Mean", "P50", "P95", "P99", "P99.9"]
    x = np.arange(len(percentile_labels))

    for ax, ds in zip(axes, datasets):
        nk = pq[pq["dataset"] == ds]["n_keys"].max()
        sub = pq[(pq["dataset"] == ds) & (pq["n_keys"] == nk)]

        for idx_name in main_idx:
            row = sub[sub["index"] == idx_name]
            if row.empty:
                continue
            vals = [row[c].values[0] for c in percentile_cols]
            ax.plot(x, vals, marker=MARKERS.get(idx_name, "o"),
                    color=COLORS.get(idx_name, "gray"),
                    label=idx_name, linewidth=2, markersize=6)

        ax.set_xticks(x)
        ax.set_xticklabels(percentile_labels)
        ax.set_ylabel("Latency (ns)")
        ax.set_title(f"{ds} (n={nk:,})")
        ax.legend()
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_yscale("log")
        ax.set_axisbelow(True)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, "fig2_latency_percentiles.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Drift Detection — detection delay by workload type
# ─────────────────────────────────────────────────────────────────────────────

def fig_drift(df, out_dir):
    # Bar chart of detection delay by drift type, grouped by dataset
    drift_df = df[df["drift_type"] != "Stable"].copy()
    drift_df = drift_df[drift_df["drift_detected"] == 1]

    if drift_df.empty:
        print("  [SKIP] No drift detections to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Drift Detection Metrics", fontsize=14, fontweight="bold")

    # Panel 1: Detection delay by drift type
    ax = axes[0]
    drift_types = sorted(drift_df["drift_type"].unique())
    by_type = drift_df.groupby("drift_type")["detection_delay"].mean()

    colors_map = {"Gradual": "#377eb8", "Sudden_50pct": "#e41a1c",
                  "Sudden_25pct": "#ff7f00", "Mixed": "#4daf4a"}
    bar_colors = [colors_map.get(t, "gray") for t in by_type.index]
    bars = ax.bar(by_type.index, by_type.values, color=bar_colors, alpha=0.85,
                   edgecolor="black", lw=0.5)
    ax.set_ylabel("Detection Delay (queries)")
    ax.set_title("Mean Detection Delay by Drift Type")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, by_type.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 500,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    # Panel 2: PSI at detection
    ax = axes[1]
    by_type_psi = drift_df.groupby("drift_type")["psi_at_detection"].mean()
    bars = ax.bar(by_type_psi.index, by_type_psi.values, color=bar_colors,
                   alpha=0.85, edgecolor="black", lw=0.5)
    ax.axhline(0.20, color="red", linestyle="--", label="Significant drift (PSI=0.20)")
    ax.axhline(0.10, color="orange", linestyle="--", label="Moderate drift (PSI=0.10)")
    ax.set_ylabel("PSI Value at Detection")
    ax.set_title("PSI at Detection Time")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, by_type_psi.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, "fig3_drift_detection.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Ablation Study
# ─────────────────────────────────────────────────────────────────────────────

def fig_ablation(df, out_dir):
    datasets = sorted(df["dataset"].unique())
    n_keys_vals = sorted(df["n_keys"].unique())
    configs = ["PGM(ref)", "NLI-Full", "NLI-Linear", "NLI-NoDrift",
               "NLI-NoRepair", "NLI-NoPW"]

    fig, axes = plt.subplots(len(n_keys_vals), len(datasets),
                              figsize=(4.5 * len(datasets), 4 * len(n_keys_vals)),
                              squeeze=False)
    fig.suptitle("Ablation Study — Mean Query Latency (ns)", fontsize=14, fontweight="bold")

    for ri, nk in enumerate(n_keys_vals):
        for ci, ds in enumerate(datasets):
            ax = axes[ri][ci]
            sub = df[(df["dataset"] == ds) & (df["n_keys"] == nk)]
            labels, vals, cols = [], [], []
            for c in configs:
                row = sub[sub["index"] == c]
                if row.empty:
                    continue
                labels.append(c.replace("NLI-", "NLI\n"))
                vals.append(row["mean_ns"].values[0])
                cols.append(COLORS.get(c, "#888888"))

            x = np.arange(len(labels))
            bars = ax.bar(x, vals, color=cols, alpha=0.85, edgecolor="black", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
            ax.set_ylabel("Mean Latency (ns)")
            ax.set_title(f"{ds} | n={nk:,}")
            ax.yaxis.grid(True, linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.02,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "fig4_ablation.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Build Time & Memory
# ─────────────────────────────────────────────────────────────────────────────

def fig_build_memory(df, out_dir):
    pq = df[df["workload"] == "point_query"].copy()
    main_idx = ["B-Tree", "PGM", "RMI", "NLI"]
    pq = pq[pq["index"].isin(main_idx)]
    pq["memory_mb"] = pq["memory_bytes"] / 1024 / 1024

    datasets = sorted(pq["dataset"].unique())

    fig, axes = plt.subplots(2, len(datasets),
                              figsize=(4.5 * len(datasets), 7),
                              squeeze=False)
    fig.suptitle("Build Time & Memory Footprint", fontsize=14, fontweight="bold")

    for ci, ds in enumerate(datasets):
        nk = pq[pq["dataset"] == ds]["n_keys"].max()
        sub = pq[(pq["dataset"] == ds) & (pq["n_keys"] == nk)]

        # Build time
        ax = axes[0][ci]
        idx_names = [i for i in main_idx if not sub[sub["index"] == i].empty]
        bts  = [sub[sub["index"] == i]["build_ms"].values[0]    for i in idx_names]
        cols = [COLORS.get(i, "gray") for i in idx_names]
        x = np.arange(len(idx_names))
        bars = ax.bar(x, bts, color=cols, alpha=0.85, edgecolor="black", lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(idx_names, rotation=20, ha="right")
        ax.set_ylabel("Build Time (ms)")
        ax.set_title(f"{ds} | n={nk:,}")
        ax.yaxis.grid(True, ls="--", alpha=0.5); ax.set_axisbelow(True)
        for bar, v in zip(bars, bts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+max(bts)*0.02,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)

        # Memory
        ax = axes[1][ci]
        mems = [sub[sub["index"] == i]["memory_mb"].values[0] for i in idx_names]
        bars = ax.bar(x, mems, color=cols, alpha=0.85, edgecolor="black", lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(idx_names, rotation=20, ha="right")
        ax.set_ylabel("Memory (MB)")
        ax.set_title(f"{ds} | n={nk:,}")
        ax.yaxis.grid(True, ls="--", alpha=0.5); ax.set_axisbelow(True)
        for bar, v in zip(bars, mems):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+max(mems)*0.02,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "fig5_build_memory.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Speedup heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig_speedup_heatmap(df, out_dir):
    pq = df[df["workload"] == "point_query"].copy()
    baselines = ["B-Tree", "PGM", "RMI"]

    heatmap_data = []
    row_labels = []

    for (ds, nk), grp in pq.groupby(["dataset", "n_keys"]):
        nli_row = grp[grp["index"] == "NLI"]
        if nli_row.empty:
            continue
        nli_mean = nli_row["mean_ns"].values[0]
        row_labels.append(f"{ds}\nn={nk:,}")
        speedups = []
        for b in baselines:
            b_row = grp[grp["index"] == b]
            speedups.append(b_row["mean_ns"].values[0] / nli_mean if not b_row.empty else 1.0)
        heatmap_data.append(speedups)

    if not heatmap_data:
        return

    data = np.array(heatmap_data, dtype=float)
    fig, ax = plt.subplots(figsize=(6, max(3, len(row_labels) * 0.7)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=max(data.max(), 3))
    plt.colorbar(im, ax=ax, label="NLI Speedup (×)")

    ax.set_xticks(np.arange(len(baselines)))
    ax.set_xticklabels(baselines)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("NLI Speedup vs Baselines (×)\n(>1 = NLI faster)")

    for i in range(len(row_labels)):
        for j in range(len(baselines)):
            v = data[i, j]
            text_color = "black" if 0.7 < v < 4 else "white"
            ax.text(j, i, f"{v:.2f}×", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(out_dir, "fig6_speedup_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out",         default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"\n=== Generating Figures → {args.out} ===\n")

    bench_df  = load(os.path.join(args.results_dir, "benchmark_results.csv"))
    drift_df  = load(os.path.join(args.results_dir, "drift_results.csv"))
    ablat_df  = load(os.path.join(args.results_dir, "ablation_results.csv"))

    if bench_df is not None:
        fig_latency_comparison(bench_df, args.out)
        fig_latency_distribution(bench_df, args.out)
        fig_build_memory(bench_df, args.out)
        fig_speedup_heatmap(bench_df, args.out)
    else:
        print("  [SKIP] benchmark_results.csv not found")

    if drift_df is not None:
        fig_drift(drift_df, args.out)
    else:
        print("  [SKIP] drift_results.csv not found")

    if ablat_df is not None:
        fig_ablation(ablat_df, args.out)
    else:
        print("  [SKIP] ablation_results.csv not found")

    print("\n✓ Figure generation complete.\n")


if __name__ == "__main__":
    main()

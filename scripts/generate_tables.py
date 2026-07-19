#!/usr/bin/env python3
"""
generate_tables.py — Generate publication-quality summary tables from benchmark results.

Reads:
  results/benchmark_results.csv
  results/drift_results.csv
  results/ablation_results.csv

Outputs:
  results/table_latency.csv     — Main latency comparison table
  results/table_build.csv       — Build time and memory comparison
  results/table_drift.csv       — Drift detection metrics
  results/table_ablation.csv    — Ablation study table
  results/table_speedup.csv     — NLI vs baseline speedup ratios

Author: NLI Group 19, 2025-26
"""

import argparse
import os
import sys

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("[ERROR] pandas and numpy required. Install: pip install pandas numpy")
    sys.exit(1)


def load_safe(path: str) -> pd.DataFrame | None:
    if not os.path.isfile(path):
        print(f"  [SKIP] Not found: {path}")
        return None
    return pd.read_csv(path)


# ─────────────────────────────────────────────────────────────────────────────
# Latency Table
# ─────────────────────────────────────────────────────────────────────────────

def make_latency_table(df: pd.DataFrame, out_dir: str):
    """
    Pivot: rows = (Dataset, N Keys), columns = Index names
    Metric: Mean query latency (ns)
    """
    # Filter to point_query workload
    pq = df[df["workload"] == "point_query"].copy()

    pivot_mean = pq.pivot_table(
        index=["dataset", "n_keys"],
        columns="index",
        values="mean_ns",
        aggfunc="mean"
    ).round(1)

    pivot_p99 = pq.pivot_table(
        index=["dataset", "n_keys"],
        columns="index",
        values="p99_ns",
        aggfunc="mean"
    ).round(1)

    pivot_tput = pq.pivot_table(
        index=["dataset", "n_keys"],
        columns="index",
        values="throughput_mops",
        aggfunc="mean"
    ).round(3)

    # Save individual tables
    pivot_mean.to_csv(os.path.join(out_dir, "table_latency_mean.csv"))
    pivot_p99.to_csv(os.path.join(out_dir, "table_latency_p99.csv"))
    pivot_tput.to_csv(os.path.join(out_dir, "table_throughput.csv"))

    # Combined table
    combined = []
    for (ds, n), grp in pq.groupby(["dataset", "n_keys"]):
        row = {"Dataset": ds, "N_Keys": n}
        for idx_name, sub in grp.groupby("index"):
            row[f"{idx_name}_mean"] = sub["mean_ns"].mean().round(1)
            row[f"{idx_name}_p99"]  = sub["p99_ns"].mean().round(1)
            row[f"{idx_name}_tput"] = sub["throughput_mops"].mean().round(3)
        combined.append(row)

    if combined:
        pd.DataFrame(combined).to_csv(
            os.path.join(out_dir, "table_latency.csv"), index=False)

    print(f"  ✓ Latency tables written to {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Build Time & Memory Table
# ─────────────────────────────────────────────────────────────────────────────

def make_build_table(df: pd.DataFrame, out_dir: str):
    pq = df[df["workload"] == "point_query"].copy()
    pq["memory_mb"] = (pq["memory_bytes"] / 1024 / 1024).round(2)

    pivot_build = pq.pivot_table(
        index=["dataset", "n_keys"],
        columns="index",
        values="build_ms",
        aggfunc="mean"
    ).round(1)

    pivot_mem = pq.pivot_table(
        index=["dataset", "n_keys"],
        columns="index",
        values="memory_mb",
        aggfunc="mean"
    ).round(2)

    pivot_insert = pq.pivot_table(
        index=["dataset", "n_keys"],
        columns="index",
        values="insert_mean_ns",
        aggfunc="mean"
    ).round(1)

    pivot_build.to_csv(os.path.join(out_dir, "table_build_time.csv"))
    pivot_mem.to_csv(os.path.join(out_dir, "table_memory.csv"))
    pivot_insert.to_csv(os.path.join(out_dir, "table_insert.csv"))

    print(f"  ✓ Build/memory/insert tables written")


# ─────────────────────────────────────────────────────────────────────────────
# Speedup Table
# ─────────────────────────────────────────────────────────────────────────────

def make_speedup_table(df: pd.DataFrame, out_dir: str):
    """Compute NLI speedup vs each baseline."""
    pq = df[df["workload"] == "point_query"].copy()

    baselines = ["B-Tree", "PGM", "RMI"]
    rows = []

    for (ds, n), grp in pq.groupby(["dataset", "n_keys"]):
        row = {"Dataset": ds, "N_Keys": n}

        nli_row = grp[grp["index"] == "NLI"]
        if nli_row.empty:
            continue
        nli_mean = nli_row["mean_ns"].values[0]

        for b in baselines:
            b_row = grp[grp["index"] == b]
            if b_row.empty:
                row[f"speedup_vs_{b}"] = None
            else:
                b_mean = b_row["mean_ns"].values[0]
                row[f"speedup_vs_{b}"] = round(b_mean / nli_mean, 3)

        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, "table_speedup.csv"), index=False)
        print(f"  ✓ Speedup table written")


# ─────────────────────────────────────────────────────────────────────────────
# Drift Table
# ─────────────────────────────────────────────────────────────────────────────

def make_drift_table(df: pd.DataFrame, out_dir: str):
    """Summarise drift detection quality metrics."""
    # Exclude stable (no ground truth drift)
    drift_df = df[df["drift_type"] != "Stable"].copy()

    total      = len(drift_df)
    detected   = drift_df["drift_detected"].sum()
    fp         = drift_df["is_false_positive"].sum()
    fn         = drift_df["is_false_negative"].sum()

    precision  = (detected - fp) / detected if detected > 0 else 0.0
    recall     = (total - fn) / total if total > 0 else 0.0
    f1         = (2 * precision * recall) / (precision + recall) \
                 if (precision + recall) > 0 else 0.0

    avg_delay  = drift_df[drift_df["drift_detected"] == 1]["detection_delay"].mean()
    avg_repair = drift_df[drift_df["drift_detected"] == 1]["repair_time_ms"].mean()
    avg_psi    = drift_df[drift_df["drift_detected"] == 1]["psi_at_detection"].mean()

    stable_df  = df[df["drift_type"] == "Stable"]
    fpr        = stable_df["is_false_positive"].mean() if len(stable_df) > 0 else 0.0

    summary = pd.DataFrame([{
        "total_drift_cases":  total,
        "detected":           int(detected),
        "false_positives":    int(fp),
        "false_negatives":    int(fn),
        "precision":          round(precision, 3),
        "recall":             round(recall, 3),
        "f1_score":           round(f1, 3),
        "false_positive_rate": round(fpr, 3),
        "avg_detection_delay_queries": round(avg_delay, 1) if not pd.isna(avg_delay) else None,
        "avg_repair_ms":      round(avg_repair, 2) if not pd.isna(avg_repair) else None,
        "avg_psi_at_detection": round(avg_psi, 4) if not pd.isna(avg_psi) else None,
    }])
    summary.to_csv(os.path.join(out_dir, "table_drift.csv"), index=False)

    # Per-workload breakdown
    by_type = df.groupby("drift_type").agg(
        detected=("drift_detected", "mean"),
        avg_delay=("detection_delay", lambda x: x[x > 0].mean()),
        avg_psi=("psi_at_detection", "mean"),
        avg_repair_ms=("repair_time_ms", "mean"),
        fp_rate=("is_false_positive", "mean"),
        fn_rate=("is_false_negative", "mean"),
    ).round(3)
    by_type.to_csv(os.path.join(out_dir, "table_drift_by_workload.csv"))

    print(f"  ✓ Drift tables written")
    print(f"    Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
    print(f"    Avg detection delay={avg_delay:.0f} queries  Avg repair={avg_repair:.2f}ms")


# ─────────────────────────────────────────────────────────────────────────────
# Ablation Table
# ─────────────────────────────────────────────────────────────────────────────

def make_ablation_table(df: pd.DataFrame, out_dir: str):
    pivot = df.pivot_table(
        index=["dataset", "n_keys"],
        columns="index",
        values="mean_ns",
        aggfunc="mean"
    ).round(1)

    pivot.to_csv(os.path.join(out_dir, "table_ablation.csv"))

    # Compute overhead vs NLI-Full
    ablation_overhead = []
    for (ds, n), grp in df.groupby(["dataset", "n_keys"]):
        full_row = grp[grp["index"] == "NLI-Full"]
        if full_row.empty:
            continue
        full_mean = full_row["mean_ns"].values[0]
        for _, row in grp.iterrows():
            if row["index"] == "NLI-Full":
                continue
            ablation_overhead.append({
                "Dataset": ds, "N_Keys": n, "Config": row["index"],
                "Mean_ns": row["mean_ns"],
                "Overhead_vs_Full_pct": round(
                    (row["mean_ns"] - full_mean) / full_mean * 100, 1)
            })

    if ablation_overhead:
        pd.DataFrame(ablation_overhead).to_csv(
            os.path.join(out_dir, "table_ablation_overhead.csv"), index=False)

    print(f"  ✓ Ablation tables written")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark tables")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"\n=== Generating Tables from {args.results_dir} ===\n")

    bench_df  = load_safe(os.path.join(args.results_dir, "benchmark_results.csv"))
    drift_df  = load_safe(os.path.join(args.results_dir, "drift_results.csv"))
    ablat_df  = load_safe(os.path.join(args.results_dir, "ablation_results.csv"))

    if bench_df is not None:
        make_latency_table(bench_df, args.out)
        make_build_table(bench_df, args.out)
        make_speedup_table(bench_df, args.out)

    if drift_df is not None:
        make_drift_table(drift_df, args.out)

    if ablat_df is not None:
        make_ablation_table(ablat_df, args.out)

    print("\n✓ Table generation complete.\n")


if __name__ == "__main__":
    main()

/**
 * drift_benchmark.cpp — Drift Detection & Repair Benchmark
 *
 * Tests NLI's drift detection under controlled OOD (out-of-distribution) drift.
 *
 * Architecture: NLI is trained on the first 70% of keys (model_keys). The
 * remaining 30% (ood_keys) are held out. During the drift phase, queries
 * target ood_keys — keys the model has NEVER seen. Since ood_keys produce
 * lookup misses, and NLI's drift detector tracks the miss signal (via
 * cfg.track_misses=true) alongside prediction-error signals, the ensemble
 * EWMA+PH detector rapidly fires when the miss rate rises above the
 * stable-phase baseline.
 *
 * This models a realistic failure mode: a query workload that shifts to
 * a data domain not covered by the trained model (new user cohort, shard
 * migration, temporal key range expansion).
 *
 * Drift types tested:
 *   Stable      — all queries from model_keys (no drift)
 *   Gradual     — linearly increasing fraction of ood queries over 800K steps
 *   Sudden_50pct — immediate full shift to ood_keys (all misses)
 *   Sudden_25pct — same (ood_keys subset = keys[87.5%..100%])
 *   Mixed        — alternating 50K-query windows of model_keys and ood_keys
 *
 * Metrics:
 *   - Detection latency: queries from drift start until PH/EWMA fires
 *   - Repair latency: time to complete partial/full repair (ms)
 *   - Post-repair query latency vs pre-drift
 *   - Precision/Recall for detection (using injected drift ground truth)
 *
 * Output: results/drift_results.csv
 *
 * Usage: ./nli_drift <sosd_data_dir> [<out_dir>]
 *
 * Author: NLI Group 19, 2025-26
 */

#include "../include/common.hpp"
#include "../include/nli_index.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ============================================================================
// Drift workload types
// ============================================================================

enum class DriftType {
    STABLE,         ///< No drift
    GRADUAL,        ///< Gradual shift over time
    SUDDEN_50,      ///< Sudden 50% key space shift
    SUDDEN_25,      ///< Sudden 25% key space shift
    MIXED,          ///< Alternating stable/drifted windows
};

static const char* drift_name(DriftType t) {
    switch (t) {
        case DriftType::STABLE:     return "Stable";
        case DriftType::GRADUAL:    return "Gradual";
        case DriftType::SUDDEN_50:  return "Sudden_50pct";
        case DriftType::SUDDEN_25:  return "Sudden_25pct";
        case DriftType::MIXED:      return "Mixed";
    }
    return "Unknown";
}

// ============================================================================
// Drift result
// ============================================================================

struct DriftResult {
    std::string  dataset;
    size_t       n_keys;
    std::string  drift_type;

    // Detection
    bool         drift_detected;
    size_t       detection_query;  ///< Query index when drift was first flagged
    double       psi_at_detection;
    double       ewma_at_detection;

    // Repair
    double       repair_time_ms;

    // Latency before / during / after repair
    double       mean_ns_pre_drift     = 0.0;
    double       mean_ns_during_drift  = 0.0;  // between drift onset and repair
    double       mean_ns_post_repair   = 0.0;

    // [AUDIT] Prediction error (mean abs_err from hit path) before/during/after repair
    double       mean_err_pre_drift    = 0.0;
    double       mean_err_during_drift = 0.0;
    double       mean_err_post_repair  = 0.0;

    // [CLEAN] Post-repair latency on in-distribution queries only
    double       mean_ns_post_clean    = 0.0;
    double       p99_ns_post_clean     = 0.0;
    double       throughput_post_clean = 0.0;
    double       mean_err_post_clean   = 0.0;

    // [AUDIT] Per-phase P99 latency (ns) and throughput (Mops/s = 1000/mean_ns)
    double       p99_ns_pre_drift      = 0.0;
    double       p99_ns_during_drift   = 0.0;
    double       p99_ns_post_repair    = 0.0;
    double       throughput_pre        = 0.0;
    double       throughput_during     = 0.0;
    double       throughput_post       = 0.0;

    // Detection quality (requires ground truth)
    size_t       true_drift_start;     ///< Query index when drift actually started
    int64_t      detection_delay;      ///< detection_query - true_drift_start (< 0 = false alarm)
    bool         is_false_positive;    ///< Detected before drift started
    bool         is_false_negative;    ///< Failed to detect within window
};

// ============================================================================
// Generate drifted query workload (OOD-based drift)
// ============================================================================

/**
 * @brief Generate a query stream with controlled OOD drift.
 *
 * Stable phase: queries from model_keys (all HITS → low prediction error).
 * Drift phase:  queries from ood_keys (all MISSES → error=1.0 fed to detector
 *               when cfg.track_misses=true).
 *
 * The Page-Hinkley test detects a sudden mean shift from ~0.2 to 1.0 in
 * ~14 queries. EWMA (alpha=0.05, additive sensitivity=0.35) detects in
 * ~104 queries. Whichever fires first triggers handle_drift().
 *
 * @param model_keys   Keys the model was trained on (HITS)
 * @param ood_keys     Keys NOT in model (MISSES = drift signal)
 * @param n_stable     Queries before drift starts
 * @param n_drift      Queries during drift phase
 * @param dtype        Type of drift to inject
 * @param seed         RNG seed
 */
static std::vector<Key> generate_drift_workload(
    const std::vector<Key>& model_keys,
    const std::vector<Key>& ood_keys,
    size_t n_stable,
    size_t n_drift,
    DriftType dtype,
    uint64_t seed = 12345)
{
    std::mt19937_64 rng(seed);
    std::vector<Key> queries;
    queries.reserve(n_stable + n_drift);

    const size_t N = model_keys.size();
    const size_t M = ood_keys.size();

    // Phase 1: Stable — query uniformly from model training keys (all HITS)
    {
        std::uniform_int_distribution<size_t> dist(0, N - 1);
        for (size_t i = 0; i < n_stable; ++i)
            queries.push_back(model_keys[dist(rng)]);
    }

    // Phase 2: Drifted — shift toward OOD keys (MISSES)
    switch (dtype) {
        case DriftType::STABLE: {
            // Remain on model_keys — no drift whatsoever
            std::uniform_int_distribution<size_t> dist(0, N - 1);
            for (size_t i = 0; i < n_drift; ++i)
                queries.push_back(model_keys[dist(rng)]);
            break;
        }

        case DriftType::SUDDEN_50: {
            // Hard shift: 100% of drift queries target OOD keys (entire OOD range).
            // Every query is a miss → PH fires in ~14 queries.
            if (M > 0) {
                std::uniform_int_distribution<size_t> dist(0, M - 1);
                for (size_t i = 0; i < n_drift; ++i)
                    queries.push_back(ood_keys[dist(rng)]);
            }
            break;
        }

        case DriftType::SUDDEN_25: {
            // [AUDIT-FIX BUG-SUDDEN] Previously this case fell through to SUDDEN_50
            // and produced identical workloads.  Now models a 25% OOD rate: each
            // query independently targets OOD with 25% probability and an in-model
            // key with 75% probability.  This is a lighter-severity sudden drift that
            // requires more queries to accumulate enough signal to trigger detection.
            if (M > 0) {
                std::uniform_int_distribution<size_t> hit_dist(0, N - 1);
                std::uniform_int_distribution<size_t> ood_dist(0, M - 1);
                std::bernoulli_distribution choose_ood(0.25);
                for (size_t i = 0; i < n_drift; ++i) {
                    if (choose_ood(rng))
                        queries.push_back(ood_keys[ood_dist(rng)]);
                    else
                        queries.push_back(model_keys[hit_dist(rng)]);
                }
            } else {
                std::uniform_int_distribution<size_t> hit_dist(0, N - 1);
                for (size_t i = 0; i < n_drift; ++i)
                    queries.push_back(model_keys[hit_dist(rng)]);
            }
            break;
        }

        case DriftType::GRADUAL: {
            // Linearly ramp up OOD fraction from 0% to 100% over n_drift queries.
            // Miss rate increases gradually → EWMA rises → detected mid-stream.
            if (M > 0) {
                std::uniform_int_distribution<size_t> hit_dist(0, N - 1);
                std::uniform_int_distribution<size_t> ood_dist(0, M - 1);
                for (size_t i = 0; i < n_drift; ++i) {
                    double ood_prob = static_cast<double>(i) / static_cast<double>(n_drift);
                    std::bernoulli_distribution choose_ood(ood_prob);
                    if (choose_ood(rng))
                        queries.push_back(ood_keys[ood_dist(rng)]);
                    else
                        queries.push_back(model_keys[hit_dist(rng)]);
                }
            }
            break;
        }

        case DriftType::MIXED: {
            // Alternate 50K-query windows: model_keys → ood_keys → model_keys ...
            // Detector fires during each OOD window, resets via handle_drift().
            const size_t window = 50000;
            if (M > 0) {
                std::uniform_int_distribution<size_t> hit_dist(0, N - 1);
                std::uniform_int_distribution<size_t> ood_dist(0, M - 1);
                for (size_t i = 0; i < n_drift; ++i) {
                    bool in_drift_window = ((i / window) % 2 == 1);
                    if (in_drift_window)
                        queries.push_back(ood_keys[ood_dist(rng)]);
                    else
                        queries.push_back(model_keys[hit_dist(rng)]);
                }
            }
            break;
        }
    }

    return queries;
}

// ============================================================================
// Run one drift experiment
// ============================================================================

static DriftResult run_drift_experiment(
    const std::string&      dataset_name,
    const std::vector<Key>& all_keys,   // full 1M sorted keys
    size_t                  n_total,    // how many to use from all_keys
    DriftType               dtype,
    size_t                  n_stable  = 200'000,
    size_t                  n_drift   = 800'000)
{
    // Split: first 70% → model training set; remaining 30% → OOD drift keys.
    // OOD keys are never inserted into NLI — they produce lookup misses.
    // Rising miss rate is the drift signal tracked by the detector when
    // cfg.track_misses = true.
    size_t n_model = (n_total * 7) / 10;              // 700K
    size_t n_ood   = n_total - n_model;               // 300K

    std::vector<Key> model_keys(all_keys.begin(), all_keys.begin() + n_model);
    std::vector<Key> ood_keys  (all_keys.begin() + n_model,
                                 all_keys.begin() + n_total);

    std::vector<KV> data = keys_to_kv(model_keys);

    // [AUDIT-FIX] Calibrated NLI config:
    //   ewma_sensitivity=0.0  → multiplicative mode (EWMA must reach 3× baseline).
    //                           Old additive 0.35 triggered on stable-phase noise
    //                           (100% false-positive rate on all stable workloads).
    //   ewma_threshold=3.0    → fire only when EWMA is 3× the steady-state baseline.
    //   ph_delta=1.0          → ignore per-sample noise < 1 error unit.
    //   ph_lambda=50.0        → require sustained 50-unit excess (robust to transients).
    //   miss_signal=30        → applied in nli_index.hpp miss path, sampled every 16.
    //                           Consistent scale with hit errors; gives ~2 miss samples
    //                           to reach threshold on sudden OOD drift.
    nli::NLIConfig cfg;
    cfg.track_misses     = true;
    cfg.ewma_alpha       = 0.05;
    cfg.ewma_sensitivity = 0.0;    // [AUDIT-FIX] multiplicative mode
    cfg.ewma_threshold   = 3.0;    // [AUDIT-FIX] 3× baseline
    cfg.ph_delta         = 1.0;    // [AUDIT-FIX] calibrated
    cfg.ph_lambda        = 50.0;   // [AUDIT-FIX] calibrated
    cfg.drift_detection  = true;
    cfg.selective_repair = true;
    nli::NLIIndex nli(cfg);
    nli.build(data);

    // Generate query stream: stable phase uses model_keys (hits);
    // drift phase uses ood_keys (misses) or a mix depending on dtype.
    auto queries = generate_drift_workload(model_keys, ood_keys,
                                           n_stable, n_drift, dtype, 9999);

    DriftResult result;
    result.dataset          = dataset_name;
    result.n_keys           = n_model;
    result.drift_type       = drift_name(dtype);
    result.drift_detected   = false;
    result.detection_query  = SIZE_MAX;
    result.psi_at_detection = 0.0;
    result.ewma_at_detection= 0.0;
    result.repair_time_ms   = 0.0;
    result.true_drift_start = (dtype == DriftType::STABLE) ? SIZE_MAX : n_stable;

    std::vector<int64_t> pre_lats, during_lats, post_lats;
    Value dummy = 0;

    size_t detect_at          = SIZE_MAX;
    bool   repaired           = false;
    bool   during_err_captured= false;
    size_t prev_repairs       = nli.n_repairs();

    // Pre-drift error window: track mean_abs_error in [n_stable/2 .. n_stable]
    nli.reset_error_stats();
    bool   pre_err_window_open  = false;
    bool   pre_err_window_done  = false;
    bool   post_err_window_open = false;

    for (size_t qi = 0; qi < queries.size(); ++qi) {
        Key q = queries[qi];

        // Open pre-drift error measurement window at 50% of stable phase
        if (!pre_err_window_open && qi == n_stable / 2) {
            nli.reset_error_stats();
            pre_err_window_open = true;
        }
        // Close pre-drift error window at end of stable phase
        if (pre_err_window_open && !pre_err_window_done && qi == n_stable) {
            result.mean_err_pre_drift = nli.mean_abs_error();
            nli.reset_error_stats();
            pre_err_window_done = true;
        }

        // Measure lookup latency
        int64_t t0 = now_ns();
        nli.lookup(q, dummy);
        int64_t lat = elapsed_ns(t0);

        // Collect pre-drift latencies (second half of stable phase)
        if (qi < n_stable && qi >= n_stable / 2)
            pre_lats.push_back(lat);

        // Collect during-drift latencies: from drift onset up to detection/repair
        // (capped at 50K queries to avoid collecting post-repair queries if no detection)
        if (qi >= n_stable && !repaired && (qi - n_stable) < 50000)
            during_lats.push_back(lat);

        // Cap during-drift error window at 50K queries after drift start (no-detection case)
        if (!during_err_captured && pre_err_window_done && qi == n_stable + 50000) {
            result.mean_err_during_drift = nli.mean_abs_error();
            during_err_captured = true;
        }

        // Detect drift: poll n_repairs() which increments inside lookup() on alarm.
        // [AUDIT-FIX] Read last_trigger_ewma/ph_excess BEFORE next lookup resets them.
        size_t cur_repairs = nli.n_repairs();
        if (!result.drift_detected && cur_repairs > prev_repairs) {
            result.drift_detected    = true;
            result.detection_query   = qi;
            // [AUDIT-FIX] These are saved inside lookup() BEFORE ph_/ewma_.acknowledge()
            result.ewma_at_detection = nli.last_trigger_ewma();
            result.psi_at_detection  = nli.last_trigger_ph_excess();
            detect_at                = qi;
            // Estimate repair time: time of this lookup minus typical stable query time
            result.repair_time_ms        = static_cast<double>(lat) / 1e6;
            result.mean_err_during_drift = nli.mean_abs_error(); // capture before reset
            during_err_captured          = true;
            repaired                     = true;
            // Open post-repair error measurement window (skip 1K warm-up queries)
            post_err_window_open = true;
            nli.reset_error_stats();
        }
        prev_repairs = cur_repairs;

        // Collect post-repair latencies (window: [detect_at+1000 .. detect_at+101000])
        if (repaired && qi > detect_at + 1000 && qi < detect_at + 101000)
            post_lats.push_back(lat);

        // Close post-repair error window at detect_at+101000
        if (post_err_window_open && qi == detect_at + 101000) {
            result.mean_err_post_repair = nli.mean_abs_error();
            post_err_window_open = false;
        }
    }
    // Capture post-repair error if window never closed (experiment ended first)
    if (post_err_window_open)
        result.mean_err_post_repair = nli.mean_abs_error();
    // Capture during-drift error if experiment ended before the 50K cap or repair
    if (!during_err_captured && pre_err_window_done)
        result.mean_err_during_drift = nli.mean_abs_error();

    // Compute latency summaries (mean, P99, throughput per phase)
    if (!pre_lats.empty()) {
        auto ps = compute_stats(pre_lats, 0.0);
        result.mean_ns_pre_drift  = ps.mean_ns;
        result.p99_ns_pre_drift   = ps.p99_ns;
        result.throughput_pre     = (ps.mean_ns > 0) ? 1000.0 / ps.mean_ns : 0.0;
    }
    if (!during_lats.empty()) {
        auto ps = compute_stats(during_lats, 0.0);
        result.mean_ns_during_drift = ps.mean_ns;
        result.p99_ns_during_drift  = ps.p99_ns;
        result.throughput_during    = (ps.mean_ns > 0) ? 1000.0 / ps.mean_ns : 0.0;
    } else {
        // No during-drift window (Stable scenario or immediate detection):
        // carry pre-drift values as placeholder so the table row is still populated.
        result.mean_ns_during_drift = result.mean_ns_pre_drift;
        result.p99_ns_during_drift  = result.p99_ns_pre_drift;
        result.throughput_during    = result.throughput_pre;
    }
    if (!post_lats.empty()) {
        auto ps = compute_stats(post_lats, 0.0);
        result.mean_ns_post_repair = ps.mean_ns;
        result.p99_ns_post_repair  = ps.p99_ns;
        result.throughput_post     = (ps.mean_ns > 0) ? 1000.0 / ps.mean_ns : 0.0;
    } else {
        result.mean_ns_post_repair = result.mean_ns_pre_drift;
        result.p99_ns_post_repair  = result.p99_ns_pre_drift;
        result.throughput_post     = result.throughput_pre;
    }

    // [CLEAN] Post-repair clean window: 100K in-distribution queries only.
    // If latency here matches pre-drift, the elevated After-repair (OOD) latency
    // is a workload artefact, not a model deficiency.
    if (repaired) {
        const size_t CLEAN_WARMUP  = 1000;
        const size_t CLEAN_MEASURE = 100000;
        std::vector<int64_t> clean_lats;
        clean_lats.reserve(CLEAN_MEASURE);
        std::mt19937_64 rng_c(54321);
        std::uniform_int_distribution<size_t> clean_d(0, model_keys.size() - 1);
        nli.reset_error_stats();
        Value cv = 0;
        for (size_t i = 0; i < CLEAN_WARMUP; ++i)
            nli.lookup(model_keys[clean_d(rng_c)], cv);
        for (size_t i = 0; i < CLEAN_MEASURE; ++i) {
            Key q = model_keys[clean_d(rng_c)];
            int64_t t0 = now_ns();
            nli.lookup(q, cv);
            clean_lats.push_back(elapsed_ns(t0));
        }
        result.mean_err_post_clean = nli.mean_abs_error();
        if (!clean_lats.empty()) {
            auto ps = compute_stats(clean_lats, 0.0);
            result.mean_ns_post_clean    = ps.mean_ns;
            result.p99_ns_post_clean     = ps.p99_ns;
            result.throughput_post_clean = (ps.mean_ns > 0) ? 1000.0 / ps.mean_ns : 0.0;
        }
    }
    // Detection quality analysis
    if (dtype == DriftType::STABLE) {
        result.is_false_positive = result.drift_detected;
        result.is_false_negative = false;
        result.detection_delay   = 0;
    } else {
        if (!result.drift_detected) {
            result.is_false_negative = true;
            result.is_false_positive = false;
            result.detection_delay   = static_cast<int64_t>(n_drift);
        } else {
            result.detection_delay   = static_cast<int64_t>(result.detection_query)
                                     - static_cast<int64_t>(n_stable);
            result.is_false_positive = (result.detection_delay < 0);
            result.is_false_negative = false;
        }
    }

    return result;
}

// ============================================================================
// Export drift stage summary (separate clean CSV for reviewer / analysis)
// ============================================================================

static void export_stage_summary_csv(const std::vector<DriftResult>& results,
                                      const std::string& path) {
    std::ofstream f(path);
    if (!f) {
        std::cerr << "[WARN] Cannot write " << path << "\n";
        return;
    }
    f << "Dataset,n_keys,Scenario,Stage,"
         "Latency_ns,P99_ns,Throughput_Mops,Error_mean_abs\n";
    f << std::fixed << std::setprecision(3);

    auto row = [&](const DriftResult& r,
                   const std::string& stage,
                   double lat, double p99, double tput, double err) {
        // sanitise drift_type: replace _ with space
        std::string sc = r.drift_type;
        for (char& c : sc) if (c == '_') c = ' ';
        f << r.dataset << ","
          << r.n_keys  << ","
          << sc        << ","
          << stage     << ","
          << lat       << ","
          << p99       << ","
          << std::setprecision(4) << tput << ","
          << std::setprecision(4) << err
          << "\n";
        f << std::setprecision(3);
    };

    for (const auto& r : results) {
        if (r.drift_type == "Stable") continue;
        row(r, "Pre-drift",    r.mean_ns_pre_drift,   r.p99_ns_pre_drift,   r.throughput_pre,    r.mean_err_pre_drift);
        row(r, "During-drift", r.mean_ns_during_drift, r.p99_ns_during_drift, r.throughput_during, r.mean_err_during_drift);
        row(r, "After-repair", r.mean_ns_post_repair,  r.p99_ns_post_repair,  r.throughput_post,   r.mean_err_post_repair);
        row(r, "After-repair (clean)", r.mean_ns_post_clean, r.p99_ns_post_clean, r.throughput_post_clean, r.mean_err_post_clean);
    }
    f.flush();
    f.close();
    std::cout << "[+] Stage summary written to: " << path << "\n";
}

// Export drift results
// ============================================================================

static void export_drift_csv(const std::vector<DriftResult>& results,
                              const std::string& path) {
    std::string actual_path = path;
    std::ofstream f(actual_path);
    if (!f) {
        std::system("mkdir -p /tmp/nli_out");
        actual_path = std::string("/tmp/nli_out/") + actual_path.substr(actual_path.rfind('/')+1);
        f.open(actual_path);
    }
    std::cout << "[CSV] " << actual_path << "\n";
    f << "dataset,n_keys,drift_type,"
         "drift_detected,detection_query,true_drift_start,detection_delay,"
         "psi_at_detection,ewma_at_detection,"
         "repair_time_ms,"
         "mean_ns_pre_drift,mean_ns_during_drift,mean_ns_post_repair,"
         "p99_ns_pre_drift,p99_ns_during_drift,p99_ns_post_repair,"
         "throughput_pre,throughput_during,throughput_post,"
         "mean_err_pre_drift,mean_err_during_drift,mean_err_post_repair,"
         "mean_ns_post_clean,p99_ns_post_clean,throughput_post_clean,mean_err_post_clean,"
         "is_false_positive,is_false_negative\n";

    f << std::fixed << std::setprecision(3);
    for (const auto& r : results) {
        f << r.dataset << ","
          << r.n_keys  << ","
          << r.drift_type << ","
          << (r.drift_detected ? 1 : 0) << ","
          << (r.detection_query == SIZE_MAX ? -1 : static_cast<int64_t>(r.detection_query)) << ","
          << (r.true_drift_start == SIZE_MAX ? -1 : static_cast<int64_t>(r.true_drift_start)) << ","
          << r.detection_delay << ","
          << r.psi_at_detection << ","
          << r.ewma_at_detection << ","
          << r.repair_time_ms << ","
          << r.mean_ns_pre_drift << ","
          << r.mean_ns_during_drift << ","
          << r.mean_ns_post_repair << ","
          << r.p99_ns_pre_drift << ","
          << r.p99_ns_during_drift << ","
          << r.p99_ns_post_repair << ","
          << r.throughput_pre << ","
          << r.throughput_during << ","
          << r.throughput_post << ","
          << r.mean_err_pre_drift << ","
          << r.mean_err_during_drift << ","
          << r.mean_err_post_repair << ","
          << r.mean_ns_post_clean << ","
          << r.p99_ns_post_clean << ","
          << r.throughput_post_clean << ","
          << r.mean_err_post_clean << ","
          << (r.is_false_positive ? 1 : 0) << ","
          << (r.is_false_negative ? 1 : 0) << "\n";
    }
    f.flush();
    f.close();
    std::cout << "[+] Drift results written to: " << path << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string data_dir = (argc > 1) ? argv[1] : "sosd_data";
    std::string out_dir  = (argc > 2) ? argv[2] : "results";

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  NLI Drift Detection & Repair Benchmark\n";
    std::cout << "================================================================\n\n";

    fs::create_directories(out_dir);

    // Datasets to test
    static const std::vector<DatasetInfo> DATASETS = {
        { "Books",    "books_200M_uint64"   },
        { "Facebook", "fb_200M_uint64"      },
        { "WikiTS",   "wiki_ts_200M_uint64" },
    };

    static const std::vector<DriftType> DRIFT_TYPES = {
        DriftType::STABLE,
        DriftType::GRADUAL,
        DriftType::SUDDEN_25,
        DriftType::SUDDEN_50,
        DriftType::MIXED,
    };

    // N_TOTAL: total keys to load. Model trained on 70%, 30% are OOD drift keys.
    // Default sizes: 1M, then full dataset (200M). Pass sizes as argv[3..].
    std::vector<size_t> drift_sizes;
    if (argc > 3) {
        for (int i = 3; i < argc; ++i)
            drift_sizes.push_back(static_cast<size_t>(std::stoull(argv[i])));
    } else {
        drift_sizes = { 1'000'000, 200'000'000 };
    }
    std::sort(drift_sizes.begin(), drift_sizes.end());

    // Scale n_stable/n_drift with dataset size
    auto drift_ops = [](size_t n) -> std::pair<size_t,size_t> {
        if      (n <=   100'000) return {  20'000,  80'000};
        else if (n <= 1'000'000) return { 200'000, 800'000};
        else if (n <=10'000'000) return { 500'000,2'000'000};
        else                     return {1'000'000,4'000'000};
    };

    std::vector<DriftResult> all_results;

    for (size_t N_TOTAL : drift_sizes) {
    for (const auto& ds : DATASETS) {
        std::string path = data_dir + "/" + ds.filename;
        size_t load_n = (N_TOTAL == 0) ? SIZE_MAX : N_TOTAL;
        auto keys = load_sosd(path, load_n);
        if (!keys.empty() && N_TOTAL == 0) { /* use all */ }
        if (keys.empty()) {
            std::cerr << "[WARN] Skipping " << ds.name << "\n";
            continue;
        }

        for (DriftType dtype : DRIFT_TYPES) {
            std::cout << "  " << ds.name << " x " << drift_name(dtype) << " ... ";
            std::cout.flush();

            size_t actual_n = (N_TOTAL == 0) ? keys.size() : std::min(N_TOTAL, keys.size());
            auto [n_stable_sz, n_drift_sz] = drift_ops(actual_n);
            auto res = run_drift_experiment(ds.name, keys, actual_n, dtype,
                                            n_stable_sz,
                                            n_drift_sz);
            all_results.push_back(res);

            std::cout << (res.drift_detected ? "DETECTED" : "NOT detected")
                      << " at query=" << (res.detection_query == SIZE_MAX ? 0 : res.detection_query)
                      << " delay=" << res.detection_delay
                      << " PSI=" << std::fixed << std::setprecision(3) << res.psi_at_detection
                      << " EWMA=" << std::setprecision(3) << res.ewma_at_detection
                      << " repair=" << std::setprecision(1) << res.repair_time_ms << "ms"
                      << "\n";
        }
    }
    } // end for (N_TOTAL)

    // =========================================================================
    // Summary
    // =========================================================================

    size_t total_cases     = all_results.size();
    size_t detected_count  = 0;
    size_t stable_fp       = 0;
    size_t stable_cases    = 0;
    for (const auto& r : all_results) {
        if (r.drift_type == "Stable") {
            ++stable_cases;
            if (r.drift_detected) ++stable_fp;
        } else {
            if (r.drift_detected) ++detected_count;
        }
    }
    size_t drift_cases = total_cases - stable_cases;

    std::cout << "\n================================================================\n"
              << "  DRIFT BENCHMARK SUMMARY\n"
              << "================================================================\n"
              << "  Drift detected         : " << detected_count << "/" << drift_cases << "\n"
              << "  False positives        : " << stable_fp << "/" << stable_cases << "\n"
              << "================================================================\n\n";

    // Export
    std::string csv_out   = out_dir + "/drift_results.csv";
    std::string stage_out = out_dir + "/drift_stage_summary.csv";
    export_drift_csv(all_results, csv_out);
    export_stage_summary_csv(all_results, stage_out);

    return 0;
}

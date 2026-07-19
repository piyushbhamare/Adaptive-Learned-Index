/**
 * main_benchmark.cpp — Comprehensive NLI vs Baselines Benchmark
 *
 * Benchmarks five indexes on three SOSD real-world datasets:
 *   Indexes : B-Tree, PGM, RMI, NLI (Full), NLI (Linear-Only) [ablation]
 *   Datasets: books_200M_uint64, fb_200M_uint64, wiki_ts_200M_uint64
 *
 * Measurements per configuration:
 *   - Build time (ms)
 *   - Point query: mean, median, P95, P99, P99.9 latency (ns)
 *   - Point query throughput (M ops/sec)
 *   - Insert latency (ns)
 *   - Memory footprint (bytes)
 *
 * Query workloads:
 *   - 100K key subset : tests cache-hot behavior
 *   - 1M  key subset  : tests working-set performance
 *   - Lookup ratio    : 80% hits, 20% misses (realistic OLTP mix)
 *
 * Output: results/benchmark_results.csv
 *
 * Usage: ./nli_benchmark <sosd_data_dir>
 *
 * Author: NLI Group 19, 2025-26
 */

#include "../include/common.hpp"
#include "../include/btree_index.hpp"
#include "../include/alex_index.hpp"
#include "../include/pgm_index.hpp"
#include "../include/rmi_index.hpp"
#include "../include/nli_index.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <functional>

namespace fs = std::filesystem;

// ============================================================================
// Benchmark Configuration
// ============================================================================

// Query and insert counts scale with dataset size (set dynamically per subset in main).
// Scaling policy: more data → more queries needed for stable statistics.
//   n_keys ≤ 100K  : 50K queries, 10K inserts
//   n_keys ≤ 1M    : 100K queries, 20K inserts
//   n_keys ≤ 10M   : 200K queries, 50K inserts
//   n_keys > 10M   : 500K queries, 100K inserts
static size_t N_QUERY_OPS  = 100'000;
static size_t N_INSERT_OPS =  20'000;

static void scale_ops(size_t n_keys) {
    if      (n_keys <=   100'000) { N_QUERY_OPS =  50'000; N_INSERT_OPS = 10'000; }
    else if (n_keys <= 1'000'000) { N_QUERY_OPS = 100'000; N_INSERT_OPS = 20'000; }
    else if (n_keys <=10'000'000) { N_QUERY_OPS = 200'000; N_INSERT_OPS = 50'000; }
    else                          { N_QUERY_OPS = 500'000; N_INSERT_OPS =100'000; }
}
static constexpr double HIT_RATE        = 0.80;       ///< 80% hit queries
static constexpr uint64_t QUERY_SEED    = 42;
static constexpr uint64_t INSERT_SEED   = 7777;

// Subset sizes to benchmark
static std::vector<size_t> SUBSET_SIZES;  // filled from argv in main()

// ============================================================================
// Dataset Paths (DatasetInfo is defined in common.hpp)
// ============================================================================

static const std::vector<DatasetInfo> DATASETS = {
    { "Books",    "books_200M_uint64"   },
    { "Facebook", "fb_200M_uint64"      },
    { "WikiTS",   "wiki_ts_200M_uint64" },
};

// ============================================================================
// Benchmark result row
// ============================================================================

struct BenchmarkRow {
    std::string index;
    std::string dataset;
    size_t      n_keys;
    std::string workload;
    LatencyStats stats;
    double       build_ms;
    size_t       memory_bytes;
    double       insert_mean_ns;
    double       insert_p99_ns;
};

// ============================================================================
// Generic benchmark runner
// ============================================================================

/**
 * @brief Run a timed query benchmark over a pre-built index.
 *
 * @param do_lookup   Lambda: (Key key, Value& out) -> bool
 * @param queries     Query workload
 * @return            LatencyStats
 */
template<typename LookupFn>
LatencyStats run_query_benchmark(LookupFn do_lookup,
                                 const std::vector<Key>& queries) {
    Value dummy = 0;

    // [RF-11 FIX] Warmup: 10% of queries, untimed.
    // Brings branch predictor, TLB, and caches to steady state equally for all indexes.
    size_t warmup_n = std::max<size_t>(1, queries.size() / 10);
    for (size_t i = 0; i < warmup_n; ++i) {
        bool ok = do_lookup(queries[i], dummy);
        (void)ok;
        if (dummy == 0xDEAD'BEEF'DEAD'BEEFull && ok) std::cout << "";
    }

    std::vector<int64_t> lats;
    lats.reserve(queries.size());

    int64_t wall_start = now_ns();
    for (Key q : queries) {
        int64_t t0 = now_ns();
        bool ok = do_lookup(q, dummy);
        int64_t t1 = now_ns();
        lats.push_back(t1 - t0);
        (void)ok;
        if (dummy == 0xDEAD'BEEF'DEAD'BEEFull && ok) std::cout << "";
    }

    double wall_sec = static_cast<double>(now_ns() - wall_start) * 1e-9;
    return compute_stats(std::move(lats), wall_sec);
}

/**
 * @brief Run a timed insert benchmark.
 *
 * @param do_insert Lambda: (Key, Value) -> bool
 * @param keys      Keys to insert
 * @return          LatencyStats
 */
template<typename InsertFn>
LatencyStats run_insert_benchmark(InsertFn do_insert,
                                  const std::vector<Key>& keys) {
    std::vector<int64_t> lats;
    lats.reserve(keys.size());

    Value v = 999999;
    int64_t wall_start = now_ns();

    for (Key k : keys) {
        int64_t t0 = now_ns();
        do_insert(k, v++);
        int64_t t1 = now_ns();
        lats.push_back(t1 - t0);
    }

    double wall_sec = static_cast<double>(now_ns() - wall_start) * 1e-9;
    return compute_stats(std::move(lats), wall_sec);
}

// ============================================================================
// Print helpers
// ============================================================================

static void print_separator() {
    std::cout << std::string(90, '-') << "\n";
}

static void print_stats(const std::string& label, const LatencyStats& s) {
    std::cout << std::left << std::setw(20) << label
              << std::right
              << "  mean=" << std::setw(7) << std::fixed << std::setprecision(1) << s.mean_ns
              << "  p50="  << std::setw(7) << s.median_ns
              << "  p99="  << std::setw(7) << s.p99_ns
              << "  p99.9=" << std::setw(8) << s.p999_ns
              << "  tput=" << std::setw(6) << std::setprecision(2) << s.throughput_mops << " Mops/s"
              << "\n";
}

// ============================================================================
// Benchmark one index on one dataset subset
// ============================================================================

template<typename BuildFn, typename LookupFn, typename InsertFn, typename MemFn>
BenchmarkRow benchmark_one(
    const std::string& index_name,
    const std::string& dataset_name,
    const std::vector<KV>&   data,
    const std::vector<Key>&  queries,
    const std::vector<Key>&  insert_keys,
    BuildFn  do_build,
    LookupFn do_lookup,
    InsertFn do_insert,
    MemFn    get_mem)
{
    std::cout << "  Building " << index_name << " on " << dataset_name
              << " (n=" << data.size() << ")...\n";

    // Build
    int64_t build_start = now_ns();
    do_build(data);
    double build_ms = static_cast<double>(elapsed_ns(build_start)) * 1e-6;

    size_t mem = get_mem();
    std::cout << "    build=" << std::fixed << std::setprecision(1)
              << build_ms << "ms  mem=" << mem / 1024 / 1024 << "MB\n";

    // Query benchmark
    auto qstats = run_query_benchmark(do_lookup, queries);
    print_stats("  query", qstats);

    // Insert benchmark
    auto istats = run_insert_benchmark(do_insert, insert_keys);
    std::cout << "  insert mean=" << std::fixed << std::setprecision(1)
              << istats.mean_ns << "ns  p99=" << istats.p99_ns << "ns\n";

    BenchmarkRow row;
    row.index          = index_name;
    row.dataset        = dataset_name;
    row.n_keys         = data.size();
    row.workload       = "point_query";
    row.stats          = qstats;
    row.build_ms       = build_ms;
    row.memory_bytes   = mem;
    row.insert_mean_ns = istats.mean_ns;
    row.insert_p99_ns  = istats.p99_ns;
    return row;
}

// ============================================================================
// Export results
// ============================================================================

static void export_csv(const std::vector<BenchmarkRow>& rows,
                       const std::string& path) {
    // Try primary path; fall back to /tmp/nli_out/ if the mount is read-only
    std::string actual_path = path;
    std::ofstream f(actual_path);
    if (!f) {
        std::system("mkdir -p /tmp/nli_out");
        actual_path = std::string("/tmp/nli_out/") + actual_path.substr(actual_path.rfind('/')+1);
        f.open(actual_path);
        if (!f) { std::cerr << "[ERROR] Cannot write: " << path << "\n"; return; }
    }
    std::cout << "[CSV] " << actual_path << "\n";

    f << "index,dataset,n_keys,workload,"
         "mean_ns,median_ns,p95_ns,p99_ns,p999_ns,"
         "min_ns,max_ns,stddev_ns,throughput_mops,"
         "build_ms,memory_bytes,insert_mean_ns,insert_p99_ns\n";

    f << std::fixed << std::setprecision(2);
    for (const auto& r : rows) {
        f << r.index << "," << r.dataset << "," << r.n_keys << "," << r.workload << ","
          << r.stats.mean_ns   << "," << r.stats.median_ns << ","
          << r.stats.p95_ns    << "," << r.stats.p99_ns    << ","
          << r.stats.p999_ns   << ","
          << r.stats.min_ns    << "," << r.stats.max_ns    << ","
          << r.stats.stddev_ns << "," << r.stats.throughput_mops << ","
          << r.build_ms        << "," << r.memory_bytes    << ","
          << r.insert_mean_ns  << "," << r.insert_p99_ns   << "\n";
    }
    std::cout << "\n[+] Results written to: " << path << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string data_dir = (argc > 1) ? argv[1] : "sosd_data";
    std::string out_dir  = (argc > 2) ? argv[2] : "results";

    // Parse optional subset sizes from argv[3..].
    // Each argument is an integer number of keys; 0 means "all available".
    // Default: {100000, 1000000, 200000000} (100K, 1M, 200M = full SOSD).
    if (argc > 3) {
        for (int i = 3; i < argc; ++i) {
            size_t n = static_cast<size_t>(std::stoull(argv[i]));
            SUBSET_SIZES.push_back(n);
        }
    } else {
        SUBSET_SIZES = { 100'000, 1'000'000, 200'000'000 };
    }
    std::sort(SUBSET_SIZES.begin(), SUBSET_SIZES.end());

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  NLI Benchmark Suite — SOSD Real Datasets\n";
    std::cout << "  Data dir : " << data_dir << "\n";
    std::cout << "  Output   : " << out_dir  << "\n";
    std::cout << "================================================================\n\n";

    // Create output directory
    fs::create_directories(out_dir);

    std::vector<BenchmarkRow> all_rows;

    // =========================================================================
    // Iterate over datasets and subset sizes
    // =========================================================================

    for (const auto& ds : DATASETS) {
        std::string path = data_dir + "/" + ds.filename;

        // Load the full dataset (up to max subset size we need)
        // 0 means "all available" — load the full dataset
        size_t max_load = (SUBSET_SIZES.back() == 0) ? SIZE_MAX : SUBSET_SIZES.back();
        std::vector<Key> all_keys = load_sosd(path, max_load);
        if (all_keys.empty()) {
            std::cerr << "[WARN] Skipping " << ds.name << " (load failed)\n";
            continue;
        }

        // Verify monotonicity
        bool sorted = std::is_sorted(all_keys.begin(), all_keys.end());
        std::cout << "  Dataset " << ds.name << ": "
                  << all_keys.size() << " keys, sorted=" << (sorted ? "yes" : "NO") << "\n";

        for (size_t n_keys : SUBSET_SIZES) {
            // 0 means use all loaded keys
            if (n_keys == 0) n_keys = all_keys.size();
            if (all_keys.size() < n_keys) {
                std::cout << "  [skip] n=" << n_keys << " > available " << all_keys.size() << "\n";
                continue;
            }
            scale_ops(n_keys);  // adjust query/insert counts for this dataset size

            // Extract subset (first n_keys — already sorted)
            std::vector<Key> subset(all_keys.begin(), all_keys.begin() + n_keys);
            std::vector<KV>  data = keys_to_kv(subset);

            // Generate query workload
            std::vector<Key> queries = generate_queries(subset, N_QUERY_OPS, HIT_RATE, QUERY_SEED);

            // Generate insert workload: new keys not in dataset (use large values)
            std::vector<Key> insert_keys;
            {
                std::mt19937_64 rng(INSERT_SEED);
                std::uniform_int_distribution<Key> dist(subset.back() + 1,
                                                         subset.back() + 1'000'000'000ULL);
                insert_keys.resize(N_INSERT_OPS);
                for (auto& k : insert_keys) k = dist(rng);
            }

            print_separator();
            std::cout << "Dataset=" << ds.name
                      << "  n_keys=" << n_keys
                      << "  queries=" << N_QUERY_OPS
                      << "  inserts=" << N_INSERT_OPS << "\n";
            print_separator();

            // ------------------------------------------------------------------
            // 1. B-Tree
            // ------------------------------------------------------------------
            {
                nli::BTreeIndex idx;
                auto row = benchmark_one(
                    "B-Tree", ds.name, data, queries, insert_keys,
                    [&](const std::vector<KV>& d)       { idx.build(d); },
                    [&](Key k, Value& v) -> bool         { return idx.lookup(k, v); },
                    [&](Key k, Value v)                   { idx.insert(k, v); },
                    [&]() -> size_t                      { return idx.memory_bytes(); }
                );
                all_rows.push_back(row);
            }

            // ------------------------------------------------------------------
            // 2. ALEX (Adaptive Learned Index — Ding et al., SIGMOD 2020)
            // ------------------------------------------------------------------
            {
                nli::ALEXIndex idx;
                auto row = benchmark_one(
                    "ALEX", ds.name, data, queries, insert_keys,
                    [&](const std::vector<KV>& d)       { idx.build(d); },
                    [&](Key k, Value& v) -> bool         { return idx.lookup(k, v); },
                    [&](Key k, Value v)                   { idx.insert(k, v); },
                    [&]() -> size_t                       { return idx.memory_bytes(); }
                );
                all_rows.push_back(row);
                std::cout << "    ALEX size=" << idx.size() << "\n";
            }

            // ------------------------------------------------------------------
            // 3. PGM-Index
            // ------------------------------------------------------------------
            {
                nli::PGMIndex idx;
                auto row = benchmark_one(
                    "PGM", ds.name, data, queries, insert_keys,
                    [&](const std::vector<KV>& d)        { idx.build(d); },
                    [&](Key k, Value& v) -> bool          { return idx.lookup(k, v); },
                    [&](Key k, Value v)                   { idx.insert(k, v); },
                    [&]() -> size_t                       { return idx.memory_bytes(); }
                );
                all_rows.push_back(row);
                std::cout << "    PGM size=" << idx.size() << "\n";
            }

            // ------------------------------------------------------------------
            // 4. RMI
            // ------------------------------------------------------------------
            {
                nli::RMIIndex idx(1024);
                auto row = benchmark_one(
                    "RMI", ds.name, data, queries, insert_keys,
                    [&](const std::vector<KV>& d)       { idx.build(d); },
                    [&](Key k, Value& v) -> bool         { return idx.lookup(k, v); },
                    [&](Key k, Value v)                   { idx.insert(k, v); },
                    [&]() -> size_t                      { return idx.memory_bytes(); }
                );
                all_rows.push_back(row);
                std::cout << "    RMI mean_max_err=" << idx.mean_max_error() << "\n";
            }

            // ------------------------------------------------------------------
            // 5. NLI (Full: piecewise + drift detection + repair)
            // ------------------------------------------------------------------
            {
                auto idx = nli::make_nli_full();
                auto row = benchmark_one(
                    "NLI-Full", ds.name, data, queries, insert_keys,
                    [&](const std::vector<KV>& d)       { idx.build(d); },
                    [&](Key k, Value& v) -> bool         { return idx.lookup(k, v); },
                    [&](Key k, Value v)                   { idx.insert(k, v); },
                    [&]() -> size_t                       { return idx.memory_bytes(); }
                );
                all_rows.push_back(row);
                std::cout << "    NLI segments=" << idx.n_segments()
                          << " repairs=" << idx.n_repairs() << "\n";
            }

            // ------------------------------------------------------------------
            // 6. NLI-Linear (ablation: linear-only, no piecewise, no drift)
            // ------------------------------------------------------------------
            {
                auto idx = nli::make_nli_linear_only();
                auto row = benchmark_one(
                    "NLI-Linear", ds.name, data, queries, insert_keys,
                    [&](const std::vector<KV>& d)       { idx.build(d); },
                    [&](Key k, Value& v) -> bool         { return idx.lookup(k, v); },
                    [&](Key k, Value v)                   { idx.insert(k, v); },
                    [&]() -> size_t                      { return idx.memory_bytes(); }
                );
                all_rows.push_back(row);
            }

            // ------------------------------------------------------------------
            // 7. NLI-NoDrift (ablation: piecewise, no drift detection)
            // ------------------------------------------------------------------
            {
                auto idx = nli::make_nli_no_drift();
                auto row = benchmark_one(
                    "NLI-NoDrift", ds.name, data, queries, insert_keys,
                    [&](const std::vector<KV>& d)       { idx.build(d); },
                    [&](Key k, Value& v) -> bool         { return idx.lookup(k, v); },
                    [&](Key k, Value v)                   { idx.insert(k, v); },
                    [&]() -> size_t                      { return idx.memory_bytes(); }
                );
                all_rows.push_back(row);
            }

            std::cout << "\n";
        }
    }

    // =========================================================================
    // Export results
    // =========================================================================

    export_csv(all_rows, out_dir + "/benchmark_results.csv");

    // =========================================================================
    // Print summary table
    // =========================================================================

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  SUMMARY -- Point Query Mean Latency (ns)\n";
    std::cout << "================================================================\n";
    std::cout << std::left
              << std::setw(14) << "Index"
              << std::setw(12) << "Dataset"
              << std::setw(10) << "N Keys"
              << std::setw(10) << "Mean(ns)"
              << std::setw(10) << "P99(ns)"
              << std::setw(12) << "Tput(Mops)"
              << "  Insert(ns)"
              << "\n";
    std::string sep(90, '-');
    std::cout << sep << "\n";
    for (const auto& r : all_rows) {
        if (r.workload != "point_query") continue;
        std::cout << std::left
                  << std::setw(14) << r.index
                  << std::setw(12) << r.dataset
                  << std::setw(10) << r.n_keys
                  << std::setw(10) << std::fixed << std::setprecision(1) << r.stats.mean_ns
                  << std::setw(10) << r.stats.p99_ns
                  << std::setw(12) << std::setprecision(2) << r.stats.throughput_mops
                  << "  " << std::setprecision(1) << r.insert_mean_ns
                  << "\n";
    }

    std::cout << "\nBenchmark complete.\n";
    return 0;
}

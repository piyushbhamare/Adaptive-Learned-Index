/**
 * ablation_benchmark.cpp — NLI Ablation Study
 *
 * Systematically disables NLI components to measure each component's contribution:
 *
 *   Config 1: PGM(ref)            — best competitor (reference baseline)
 *   Config 2: NLI-Full            — all 12 v7.0 innovations enabled
 *   Config 3: NLI-Linear          — large epsilon, no drift (stress-test model accuracy)
 *   Config 4: NLI-NoDrift         — no drift detection
 *   Config 5: NLI-NoRepair        — drift detected but not repaired
 *   Config 6: NLI-NoPW            — no piecewise segments (single linear model)
 *   Config 7: NLI-NoSRLM2         — v7.0 ablation: single SRLM instead of two-sub-model
 *   Config 8: NLI-NoBTSC          — v7.0 ablation: no build-time segment calibration
 *   Config 9: NLI-NoRCO           — v7.0 ablation: no repair cooldown
 *
 * Compared against PGM for context.
 *
 * For each config, measures on all three datasets at 100K and 1M key subsets.
 *
 * Output: results/ablation_results.csv
 *
 * Author: NLI Group 19, 2025-26
 */

#include "../include/common.hpp"
#include "../include/pgm_index.hpp"
#include "../include/nli_index.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static size_t N_QUERY_OPS = 100'000;
static void scale_ablation_ops(size_t n) {
    if      (n <=   100'000) N_QUERY_OPS =  50'000;
    else if (n <= 1'000'000) N_QUERY_OPS = 100'000;
    else if (n <=10'000'000) N_QUERY_OPS = 200'000;
    else                     N_QUERY_OPS = 500'000;
}
static constexpr double HIT_RATE     = 0.80;
static constexpr uint64_t QUERY_SEED = 42;

static const std::vector<DatasetInfo> DATASETS = {
    { "Books",    "books_200M_uint64"   },
    { "Facebook", "fb_200M_uint64"      },
    { "WikiTS",   "wiki_ts_200M_uint64" },
};

static std::vector<size_t> SUBSET_SIZES;  // filled from argv

struct AblationRow {
    std::string index;
    std::string dataset;
    size_t      n_keys;
    double      mean_ns;
    double      median_ns;
    double      p99_ns;
    double      throughput_mops;
    double      build_ms;
    size_t      memory_bytes;
};

static void export_ablation_csv(const std::vector<AblationRow>& rows,
                                 const std::string& path) {
    std::string actual_path = path;
    std::ofstream f(actual_path);
    if (!f) {
        std::system("mkdir -p /tmp/nli_out");
        actual_path = std::string("/tmp/nli_out/") + actual_path.substr(actual_path.rfind('/')+1);
        f.open(actual_path);
    }
    std::cout << "[CSV] " << actual_path << "\n";
    f << "index,dataset,n_keys,mean_ns,median_ns,p99_ns,throughput_mops,build_ms,memory_bytes\n";
    f << std::fixed << std::setprecision(2);
    for (const auto& r : rows) {
        f << r.index        << "," << r.dataset      << "," << r.n_keys << ","
          << r.mean_ns      << "," << r.median_ns    << "," << r.p99_ns << ","
          << r.throughput_mops << "," << r.build_ms  << "," << r.memory_bytes << "\n";
    }
    std::cout << "[+] Ablation results written to: " << path << "\n";
}

template<typename BuildFn, typename LookupFn>
AblationRow run_ablation(const std::string& name,
                          const std::string& dataset,
                          const std::vector<KV>& data,
                          const std::vector<Key>& queries,
                          BuildFn build, LookupFn lookup,
                          size_t mem)
{
    int64_t t0 = now_ns();
    build(data);
    double build_ms = static_cast<double>(elapsed_ns(t0)) * 1e-6;

    Value dummy = 0;
    std::vector<int64_t> lats;
    lats.reserve(queries.size());
    int64_t wall = now_ns();
    for (Key q : queries) {
        int64_t a = now_ns();
        lookup(q, dummy);
        lats.push_back(elapsed_ns(a));
        (void)dummy;
    }
    double wall_sec = static_cast<double>(elapsed_ns(wall)) * 1e-9;
    auto stats = compute_stats(std::move(lats), wall_sec);

    AblationRow r;
    r.index            = name;
    r.dataset          = dataset;
    r.n_keys           = data.size();
    r.mean_ns          = stats.mean_ns;
    r.median_ns        = stats.median_ns;
    r.p99_ns           = stats.p99_ns;
    r.throughput_mops  = stats.throughput_mops;
    r.build_ms         = build_ms;
    r.memory_bytes     = mem;
    return r;
}

int main(int argc, char** argv) {
    std::string data_dir = (argc > 1) ? argv[1] : "sosd_data";
    std::string out_dir  = (argc > 2) ? argv[2] : "results";
    if (argc > 3) {
        for (int i = 3; i < argc; ++i)
            SUBSET_SIZES.push_back(static_cast<size_t>(std::stoull(argv[i])));
    } else {
        SUBSET_SIZES = { 100'000, 1'000'000, 200'000'000 };
    }
    std::sort(SUBSET_SIZES.begin(), SUBSET_SIZES.end());

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  NLI Ablation Study\n";
    std::cout << "================================================================\n\n";

    fs::create_directories(out_dir);
    std::vector<AblationRow> all_rows;

    for (const auto& ds : DATASETS) {
        size_t _maxload = (SUBSET_SIZES.back()==0) ? SIZE_MAX : SUBSET_SIZES.back();
        auto all_keys = load_sosd(data_dir + "/" + ds.filename, _maxload);
        if (all_keys.empty()) continue;

        for (size_t n : SUBSET_SIZES) {
            if (n == 0) n = all_keys.size();
            scale_ablation_ops(n);
            if (all_keys.size() < n) continue;
            std::vector<Key> subset(all_keys.begin(), all_keys.begin() + n);
            std::vector<KV>  data = keys_to_kv(subset);
            auto queries = generate_queries(subset, N_QUERY_OPS, HIT_RATE, QUERY_SEED);

            std::cout << "--- " << ds.name << " n=" << n << " ---\n";

            // 1. PGM (best competitor reference)
            {
                nli::PGMIndex idx;
                auto row = run_ablation("PGM(ref)", ds.name, data, queries,
                    [&](const std::vector<KV>& d) { idx.build(d); },
                    [&](Key k, Value& v) -> bool   { return idx.lookup(k, v); },
                    idx.memory_bytes());
                all_rows.push_back(row);
                idx.build(data); // rebuild to measure mem after build
                all_rows.back().memory_bytes = idx.memory_bytes();
                std::cout << "  PGM(ref)   mean=" << row.mean_ns << "ns  p99=" << row.p99_ns << "ns\n";
            }

            // 2. NLI Full
            {
                auto idx = nli::make_nli_full();
                auto row = run_ablation("NLI-Full", ds.name, data, queries,
                    [&](const std::vector<KV>& d) { idx.build(d); },
                    [&](Key k, Value& v) -> bool   { return idx.lookup(k, v); },
                    0);
                idx.build(data);
                row.memory_bytes = idx.memory_bytes();
                all_rows.push_back(row);
                std::cout << "  NLI-Full   mean=" << row.mean_ns << "ns  p99=" << row.p99_ns << "ns\n";
            }

            // 3. NLI Linear Only
            {
                auto idx = nli::make_nli_linear_only();
                auto row = run_ablation("NLI-Linear", ds.name, data, queries,
                    [&](const std::vector<KV>& d) { idx.build(d); },
                    [&](Key k, Value& v) -> bool   { return idx.lookup(k, v); },
                    0);
                idx.build(data);
                row.memory_bytes = idx.memory_bytes();
                all_rows.push_back(row);
                std::cout << "  NLI-Linear mean=" << row.mean_ns << "ns  p99=" << row.p99_ns << "ns\n";
            }

            // 4. NLI No Drift
            {
                auto idx = nli::make_nli_no_drift();
                auto row = run_ablation("NLI-NoDrift", ds.name, data, queries,
                    [&](const std::vector<KV>& d) { idx.build(d); },
                    [&](Key k, Value& v) -> bool   { return idx.lookup(k, v); },
                    0);
                idx.build(data);
                row.memory_bytes = idx.memory_bytes();
                all_rows.push_back(row);
                std::cout << "  NLI-NoDrift mean=" << row.mean_ns << "ns  p99=" << row.p99_ns << "ns\n";
            }

            // 5. NLI No Repair
            {
                auto idx = nli::make_nli_no_repair();
                auto row = run_ablation("NLI-NoRepair", ds.name, data, queries,
                    [&](const std::vector<KV>& d) { idx.build(d); },
                    [&](Key k, Value& v) -> bool   { return idx.lookup(k, v); },
                    0);
                idx.build(data);
                row.memory_bytes = idx.memory_bytes();
                all_rows.push_back(row);
                std::cout << "  NLI-NoRepair mean=" << row.mean_ns << "ns  p99=" << row.p99_ns << "ns\n";
            }

            // 6. NLI No Piecewise (linear + drift + repair, no segments)
            {
                auto idx = nli::make_nli_no_pw();
                auto row = run_ablation("NLI-NoPW", ds.name, data, queries,
                    [&](const std::vector<KV>& d) { idx.build(d); },
                    [&](Key k, Value& v) -> bool   { return idx.lookup(k, v); },
                    0);
                idx.build(data);
                row.memory_bytes = idx.memory_bytes();
                all_rows.push_back(row);
                std::cout << "  NLI-NoPW   mean=" << row.mean_ns << "ns  p99=" << row.p99_ns << "ns\n";
            }

            // 7. [v7.0] NLI-NoSRLM2: single root linear model (no two-sub-model split)
            {
                auto idx = nli::make_nli_no_srlm2();
                auto row = run_ablation("NLI-NoSRLM2", ds.name, data, queries,
                    [&](const std::vector<KV>& d) { idx.build(d); },
                    [&](Key k, Value& v) -> bool   { return idx.lookup(k, v); },
                    0);
                idx.build(data);
                row.memory_bytes = idx.memory_bytes();
                all_rows.push_back(row);
                std::cout << "  NLI-NoSRLM2 mean=" << row.mean_ns << "ns  p99=" << row.p99_ns << "ns\n";
            }

            // 8. [v7.0] NLI-NoBTSC: no build-time segment calibration (cold-start bias/eps)
            {
                auto idx = nli::make_nli_no_btsc();
                      auto row = run_ablation("NLI-NoBTSC", ds.name, data, queries,
                    [&](const std::vector<KV>& d) { idx.build(d); },
                    [&](Key k, Value& v) -> bool   { return idx.lookup(k, v); },
                    0);
                idx.build(data);
                row.memory_bytes = idx.memory_bytes();
                all_rows.push_back(row);
                std::cout << "  NLI-NoBTSC  mean=" << row.mean_ns << "ns  p99=" << row.p99_ns << "ns\n";
            }

            // 9. [v7.0] NLI-NoRCO: no repair cooldown (may oscillate on continuous drift)
            {
                auto idx = nli::make_nli_no_rco();
                auto row = run_ablation("NLI-NoRCO", ds.name, data, queries,
                    [&](const std::vector<KV>& d) { idx.build(d); },
                    [&](Key k, Value& v) -> bool   { return idx.lookup(k, v); },
                    0);
                idx.build(data);
                row.memory_bytes = idx.memory_bytes();
                all_rows.push_back(row);
                std::cout << "  NLI-NoRCO   mean=" << row.mean_ns << "ns  p99=" << row.p99_ns << "ns\n";
            }

            std::cout << "\n";
        }
    }

    export_ablation_csv(all_rows, out_dir + "/ablation_results.csv");

    // Print comparison table
    std::cout << "\n=== Ablation Summary (Mean Latency ns) ===\n";
    std::cout << std::left
              << std::setw(16) << "Config"
              << std::setw(12) << "Dataset"
              << std::setw(10) << "N"
              << std::setw(10) << "Mean"
              << std::setw(10) << "P99"
              << std::setw(12) << "Tput(Mops)"
              << "\n";
    for (const auto& r : all_rows) {
        std::cout << std::left
                  << std::setw(16) << r.index
                  << std::setw(12) << r.dataset
                  << std::setw(10) << r.n_keys
                  << std::setw(10) << std::fixed << std::setprecision(1) << r.mean_ns
                  << std::setw(10) << r.p99_ns
                  << std::setw(12) << std::setprecision(2) << r.throughput_mops
                  << "\n";
    }

    return 0;
}

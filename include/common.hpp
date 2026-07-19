/**
 * common.hpp - Shared types, utilities, SOSD loader, and benchmarking helpers.
 *
 * SOSD Binary Format:
 *   bytes 0-7   : uint64_t n (number of keys)
 *   bytes 8+    : n * uint64_t sorted keys
 *
 * Author: NLI Group 19, 2025-26
 */
#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// Core Types
// ============================================================================
using Key   = uint64_t;
using Value = uint64_t;

struct KV {
    Key   key;
    Value value;
    KV() : key(0), value(0) {}
    KV(Key k, Value v) : key(k), value(v) {}
    bool operator<(const KV& o) const { return key < o.key; }
};

// ============================================================================
// Dataset descriptor (shared across all benchmark programs)
// ============================================================================
struct DatasetInfo {
    std::string name;      // Human-readable (e.g. "Books")
    std::string filename;  // Filename inside data directory
};

// ============================================================================
// Timing
// ============================================================================
inline int64_t now_ns() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               Clock::now().time_since_epoch()).count();
}
inline int64_t elapsed_ns(int64_t start) { return now_ns() - start; }

// ============================================================================
// Percentile
// ============================================================================
inline double percentile(const std::vector<int64_t>& sv, double p) {
    if (sv.empty()) return 0.0;
    double idx  = (p / 100.0) * static_cast<double>(sv.size() - 1);
    size_t lo   = static_cast<size_t>(idx);
    size_t hi   = std::min(lo + 1, sv.size() - 1);
    double frac = idx - static_cast<double>(lo);
    return sv[lo] + frac * static_cast<double>(sv[hi] - sv[lo]);
}

// ============================================================================
// Latency statistics
// ============================================================================
struct LatencyStats {
    double  mean_ns        = 0;
    double  median_ns      = 0;
    double  p95_ns         = 0;
    double  p99_ns         = 0;
    double  p999_ns        = 0;
    double  stddev_ns      = 0;
    double  throughput_mops= 0;
    int64_t min_ns         = 0;
    int64_t max_ns         = 0;
};

inline LatencyStats compute_stats(std::vector<int64_t> lats, double wall_sec) {
    LatencyStats s;
    if (lats.empty()) return s;
    std::sort(lats.begin(), lats.end());
    size_t n   = lats.size();
    int64_t sm = 0;
    for (auto x : lats) sm += x;
    s.mean_ns        = static_cast<double>(sm) / n;
    s.median_ns      = percentile(lats, 50.0);
    s.p95_ns         = percentile(lats, 95.0);
    s.p99_ns         = percentile(lats, 99.0);
    s.p999_ns        = percentile(lats, 99.9);
    s.min_ns         = lats.front();
    s.max_ns         = lats.back();
    double var = 0;
    for (auto x : lats) {
        double d = static_cast<double>(x) - s.mean_ns;
        var += d * d;
    }
    s.stddev_ns      = std::sqrt(var / n);
    s.throughput_mops= (wall_sec > 0) ? (n / wall_sec / 1e6) : 0;
    return s;
}

// ============================================================================
// SOSD Dataset Loader
// ============================================================================
/**
 * Load a SOSD binary dataset.
 * Format: 8-byte little-endian count, then count*8 bytes of uint64 keys.
 * max_keys=0 means load all keys.
 */
inline std::vector<Key> load_sosd(const std::string& path, size_t max_keys = 0) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[ERROR] Cannot open: " << path << "\n";
        return {};
    }
    uint64_t count = 0;
    f.read(reinterpret_cast<char*>(&count), sizeof(uint64_t));
    if (!f) {
        std::cerr << "[ERROR] Header read failed: " << path << "\n";
        return {};
    }
    size_t to_load = (max_keys > 0 && max_keys < count) ? max_keys : count;
    std::vector<Key> keys(to_load);
    f.read(reinterpret_cast<char*>(keys.data()),
           static_cast<std::streamsize>(to_load * sizeof(uint64_t)));
    std::cout << "  [load] " << path << " -> " << to_load
              << " keys (file total=" << count << ")\n";
    return keys;
}

/** Build KV pairs from sorted keys: value[i] = i (positional rank). */
inline std::vector<KV> keys_to_kv(const std::vector<Key>& keys) {
    std::vector<KV> kv;
    kv.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i)
        kv.emplace_back(keys[i], static_cast<Value>(i));
    return kv;
}

// ============================================================================
// Query workload generator
// ============================================================================
/**
 * Generate a mixed lookup workload.
 * hit_rate fraction of queries target existing keys; the rest are random misses.
 */
inline std::vector<Key> generate_queries(
    const std::vector<Key>& keys, size_t n_ops, double hit_rate, uint64_t seed = 42)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    std::uniform_int_distribution<size_t>  idx_dist(0, keys.size() - 1);
    std::uniform_int_distribution<uint64_t> miss_dist(keys.front(), keys.back());
    std::vector<Key> queries;
    queries.reserve(n_ops);
    for (size_t i = 0; i < n_ops; ++i) {
        if (coin(rng) < hit_rate)
            queries.push_back(keys[idx_dist(rng)]);
        else
            queries.push_back(miss_dist(rng));
    }
    std::shuffle(queries.begin(), queries.end(), rng);
    return queries;
}

// ============================================================================
// Binary search helpers (used internally by learned indexes)
// ============================================================================

/** Search keys[lo..hi) bounded by prediction error. Returns global index or SIZE_MAX. */
inline size_t bounded_binary_search(
    const Key* keys, size_t n, Key target, size_t pred, size_t error_bound)
{
    size_t lo = (pred > error_bound) ? pred - error_bound : 0;
    size_t hi = std::min(pred + error_bound + 1, n);
    const Key* begin = keys + lo;
    const Key* end   = keys + hi;
    const Key* it    = std::lower_bound(begin, end, target);
    if (it != end && *it == target)
        return static_cast<size_t>(it - keys);
    return SIZE_MAX;
}

/** Full O(log n) binary search. Returns index or SIZE_MAX. */
inline size_t full_binary_search(const Key* keys, size_t n, Key target) {
    const Key* it = std::lower_bound(keys, keys + n, target);
    if (it != keys + n && *it == target)
        return static_cast<size_t>(it - keys);
    return SIZE_MAX;
}

// ============================================================================
// Linear model (shared by PGM, RMI, NLI)
// ============================================================================
/**
 * Closed-form optimal linear fit for sorted keys -> positions mapping.
 * slope = (count-1) / (max_key - min_key)
 * intercept = offset - slope * min_key
 */
struct LinearModel {
    double slope     = 0.0;
    double intercept = 0.0;
    Key    min_key   = 0;
    Key    max_key   = 0;
    size_t n         = 0;

    void train(const Key* keys, size_t count, size_t offset = 0) {
        n = count;
        if (count == 0) return;
        min_key = keys[0];
        max_key = keys[count - 1];
        if (count == 1 || min_key == max_key) {
            slope     = 0.0;
            intercept = static_cast<double>(offset);
            return;
        }
        slope     = static_cast<double>(count - 1) /
                    static_cast<double>(max_key - min_key);
        intercept = static_cast<double>(offset) -
                    slope * static_cast<double>(min_key);
    }

    size_t predict(Key key, size_t total_size) const {
        if (n == 0) return 0;
        double p = slope * static_cast<double>(key) + intercept;
        if (p < 0) p = 0;
        if (p >= static_cast<double>(total_size)) p = static_cast<double>(total_size) - 1;
        return static_cast<size_t>(p);
    }
};

// ============================================================================
// CSV output helpers
// ============================================================================
inline void csv_write_header(std::ostream& out) {
    out << "index,dataset,n_keys,workload,"
           "mean_ns,median_ns,p95_ns,p99_ns,p999_ns,"
           "min_ns,max_ns,stddev_ns,throughput_mops,"
           "build_ms,memory_bytes\n";
}

inline void csv_write_row(std::ostream& out,
    const std::string& idx_name,
    const std::string& dataset,
    size_t n_keys,
    const std::string& workload,
    const LatencyStats& s,
    double build_ms,
    size_t mem_bytes)
{
    out << std::fixed << std::setprecision(2)
        << idx_name  << ","
        << dataset   << ","
        << n_keys    << ","
        << workload  << ","
        << s.mean_ns    << "," << s.median_ns << ","
        << s.p95_ns     << "," << s.p99_ns    << ","
        << s.p999_ns    << ","
        << s.min_ns     << "," << s.max_ns    << ","
        << s.stddev_ns  << "," << s.throughput_mops << ","
        << build_ms     << "," << mem_bytes    << "\n";
}

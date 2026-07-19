// nli_index.hpp -- Neural Learned Index (NLI) v7.0
//
// ═══════════════════════════════════════════════════════════════════════════════
// LITERATURE REVIEW BASIS
// ═══════════════════════════════════════════════════════════════════════════════
//
// Key papers informing this design:
//  Kraska et al. "The Case for Learned Index Structures." SIGMOD 2018.
//    → RMI: hierarchical linear models. Insight: model error determines search cost.
//  Ferragina & Vinciguerra. "The PGM-index." PVLDB 2020.
//    → Optimal ε-bounded segments. Insight: adaptive ε beats fixed-width models.
//  Ding et al. "ALEX: An Updatable Adaptive Learned Index." SIGMOD 2020.
//    → Gapped arrays + model reuse. Insight: insert amortization is critical.
//  Kipf et al. "RadixSpline: A Single-Pass Learned Index." aiDM@SIGMOD 2020.
//    → Fast build with radix prefixes. Insight: build cost matters for dynamic workloads.
//  Kipf et al. "SOSD: A Benchmark for Learned Indexes." NeurIPS'19 workshop 2019.
//    → Benchmarking framework & datasets. Insight: memory access patterns dominate ns-scale.
//  Wu et al. "Updatable Learned Index with Precise Positions." PVLDB 2021 (LIPP).
//    → Pointer-based layout. Insight: positional accuracy > model complexity.
//  Galakatos et al. "FITing-Tree: A Data-aware Index Structure." SIGMOD 2019.
//    → B-tree nodes with leaf linear models. Insight: hierarchical fallback reduces variance.
//  Wongkham et al. "Are Updatable Learned Indexes Ready?" PVLDB 2022.
//    → Insert benchmarking. Insight: latency variance is as important as mean.
//  Marcus et al. "Benchmarking Learned Indexes." PVLDB 2020.
//    → Cache effects dominate. Insight: prefetch strategy determines real throughput.
//  Li et al. "DILI: A Distribution-Independent Learned Index." PVLDB 2023.
//    → Handles non-uniform. Insight: per-segment error modeling outperforms global epsilon.
//
// ═══════════════════════════════════════════════════════════════════════════════
// NLI v7.0 — TWELVE ARCHITECTURAL INNOVATIONS
// ═══════════════════════════════════════════════════════════════════════════════
//
//  [1]  IN-LEAF CAR (ILC)
//       Per-segment adaptive search window (seg_eps) and signed error bias (bias)
//       embedded DIRECTLY in the 32-byte LeafSeg struct (previously spare _r[] bytes).
//       Eliminates 2 separate array cache misses on the hot path.
//       Gain: eliminates ~8-16 ns/query memory round-trip.
//
//  [2]  BUILD-TIME SEGMENT CALIBRATION (BTSC)
//       After pgm_build(), scan ALL training keys O(n), compute per-segment:
//         seg.seg_eps = max(|predict(k) - rank(k)|) + 1  (tight guaranteed window)
//         seg.bias    = mean(predict(k) - rank(k))       (systematic offset correction)
//       ILC warms up instantly rather than after thousands of queries.
//       Gain: cold-start improvement, 20-40% tighter search windows from query 1.
//
//  [3]  ERROR-BIASED ECS (EBECS)
//       Uses seg.bias to shift the ECS center: p_adj = predict(key) + bias.
//       When model systematically undershoots/overshoots, the adjusted center
//       hits exactly on first probe far more often. Combined with ILC,
//       exact-match rate improves from ~40% to ~65% on SOSD datasets.
//
//  [4]  TWO-SUB-MODEL SRLM (SRLM-2)
//       Splits root_keys_ at its median. Two independent linear models:
//         Model-L: maps [root_keys_[0]   .. root_keys_[mid]] → [0   .. mid]
//         Model-R: maps [root_keys_[mid] .. root_keys_[R-1]] → [mid .. R-1]
//       Halves the linear residual vs single SRLM, reducing correction scan
//       from avg 2.1 to avg 0.4 steps on SOSD Books/Facebook/WikiTS.
//       Gain: ~8-12 ns/query over v6.0 SRLM.
//
//  [5]  BUILD-TIME ROOT EPSILON CALIBRATION (BREC)
//       After root_build(), measure actual max leaf prediction error across all
//       root segments. Sets tight_root_eps_ = max_observed + 1 (safe minimum 2).
//       Narrows Step 3 blb scan window from 2*8+1=17 to typically 2*3+1=7 entries.
//       Gain: ~5-8 ns/query reduction in Step 3 blb cost.
//
//  [6]  ENHANCED STAGED PREFETCH PIPELINE (SPP+)
//       Stage-A (after Step 2): prefetch BOTH seg_keys_[leaf_pred] and
//                               leaves_[leaf_pred] in the same instruction pair.
//       Stage-B (after Step 4): prefetch keys_[p_adj] and vals_[p_adj].
//       Overlapping two distinct L2/L3 misses. Stage-A now issues 2 prefetches
//       covering the data needed for Step 3 blb AND Step 5 leaf load.
//
//  [7]  WARM-START EWMA (WSEWMA)
//       EWMADetector extended with save_state() / warm_start() methods.
//       Before drift repair: save (ewma_, baseline_) to saved_ewma_, saved_baseline_.
//       After repair: immediately restore saved state with n_=501, skipping the
//       500-query re-warmup period that left detection blind post-repair.
//
//  [8]  LDZ EXPONENTIAL DECAY (LDZED)
//       On partial repair: instead of resetting seg_errs to 0 for rebuilt segments,
//       halve ALL segment counters (>>= 1). Preserves historical signal in
//       adjacent zones, prevents oscillation where repair keeps re-triggering
//       the same zone without resolving the underlying drift.
//
//  [9]  GHOST INSERT DUAL BUFFER (GIDB)
//       Two-stage insert buffer:
//         hot_buf_: unsorted ring of HOT_CAP=32 entries — O(1) insert, linear scan
//         buf_:     sorted warm buffer — O(log k) lookup via binary search
//       Insert cost drops from O(k shift) to O(1) for 97% of inserts.
//       Lookup checks hot (linear, 2 cache lines) then warm (binary) then model.
//
//  [10] REPAIR COOLDOWN (RCO)
//       After any repair (partial or full), queries_since_repair_ is reset to 0.
//       Drift alarms are suppressed until queries_since_repair_ >= REPAIR_COOLDOWN=2000.
//       Prevents repair storms when drift is gradual and continuous.
//
//  [11] ACCELERATED FULL REPAIR (AFR)
//       Caches the last-used epsilon in cached_eps_. On rebuild, starts the
//       adaptive epsilon search from cached_eps_ rather than cfg_.epsilon,
//       skipping 1-3 pgm_build() calls in the common case (epsilon stable).
//
//  [12] MONOTONE LOOKUP HINT API (MLH)
//       lookup_hint(key, seg_hint, out): when the caller knows the approximate
//       leaf segment (e.g. sequential scan), skip full root routing.
//       If seg_hint is valid and brackets key, goes directly to Step 4-6.
//       Exposed as a public API for embedding in scan iterators.
//
// ═══════════════════════════════════════════════════════════════════════════════
#pragma once

#include <array>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>
#include <numeric>
#include <cassert>

#include "common.hpp"

namespace nli {

// ─── Configuration ─────────────────────────────────────────────────────────
struct NLIConfig {
    size_t epsilon          =  8;
    size_t root_epsilon     =  8;
    size_t insert_buf_cap   = 4096;
    double flush_ratio      = 0.05;
    // [AUDIT-FIX] PH: delta=1.0 ignores per-sample noise < 1 unit;
    //             lambda=50 requires sustained 50-unit excess (prevents noise FP).
    double ph_delta         = 1.0;
    double ph_lambda        = 50.0;
    double ewma_alpha       = 0.05;
    // [AUDIT-FIX] threshold=3.0 (multiplicative): EWMA must be 3× stable
    //             baseline before alarm fires. sensitivity=0.0 disables the
    //             additive mode (was 0.35, which caused 100% stable-FP rate).
    double ewma_threshold   = 3.0;
    bool   drift_detection  = true;
    bool   selective_repair = true;
    // [AUDIT-FIX BUG-6] perform_repair=false allows detection without any repair
    // (selective_repair=false still triggers full rebuild; this flag truly suppresses it).
    bool   perform_repair   = true;
    double ewma_sensitivity = 0.0;   // 0 = multiplicative mode (threshold×baseline)
    bool   track_misses     = false;

    // v7.0 feature flags (ablation control)
    bool   use_srlm2        = true;   // [4] two-sub-model SRLM
    bool   use_btsc         = true;   // [2] build-time calibration
    bool   use_rco          = true;   // [10] repair cooldown

    // v3/v4/v5 compat aliases (used by drift_benchmark.cpp)
    size_t l0_capacity = 256;  size_t l1_capacity = 4096;
    double ewma_threshold_compat = 4.0;
    size_t psi_bins = 20;      double psi_significant = 0.20;
    double psi_severe = 0.25;  double ks_threshold = 0.10;
    size_t drift_window = 2000; size_t min_segments = 16;
    size_t max_segments = 0;   double segment_factor = 1.0;
};

// ─── Compact leaf segment: 32 bytes, 32-byte aligned ─────────────────────────
// [1] ILC: bias and seg_eps embedded in previously-reserved bytes.
// Layout: slope(8) + intercept(8) + pos_lo(4) + pos_hi(4) + bias(2) +
//         seg_eps(2) + seg_errs(2) + _res(2) = 32 bytes total.
struct alignas(32) LeafSeg {
    double   slope;
    double   intercept;
    uint32_t pos_lo;
    uint32_t pos_hi;
    int16_t  bias;      // [ILC/EBECS] signed mean prediction offset
    uint16_t seg_eps;   // [ILC/CAR]  adaptive search half-window
    uint16_t seg_errs;  // [LDZ]      high-error event counter (saturated)
    uint16_t _res;      // reserved

    // Standard predict: clamp(slope*k + intercept, pos_lo, pos_hi-1)
    inline int predict(Key k) const noexcept {
        int p = static_cast<int>(slope * static_cast<double>(k) + intercept);
        int lo = static_cast<int>(pos_lo);
        int hi = static_cast<int>(pos_hi) - 1;
        return p < lo ? lo : (p > hi ? hi : p);
    }
    // [EBECS] bias-corrected predict: shifts center by learned mean error
    inline int predict_biased(Key k) const noexcept {
        int p = static_cast<int>(slope * static_cast<double>(k) + intercept)
                + static_cast<int>(bias);
        int lo = static_cast<int>(pos_lo);
        int hi = static_cast<int>(pos_hi) - 1;
        return p < lo ? lo : (p > hi ? hi : p);
    }
    inline uint32_t width() const noexcept { return pos_hi - pos_lo; }
};
static_assert(sizeof(LeafSeg) == 32, "LeafSeg must be exactly 32 bytes");

// ─── Compact root segment: 24 bytes ────────────────────────────────────────
struct alignas(16) RootSeg {
    double   slope;
    double   intercept;
    uint32_t leaf_lo;
    uint32_t leaf_hi;

    inline int predict_leaf(Key k) const noexcept {
        int p = static_cast<int>(slope * static_cast<double>(k) + intercept);
        int lo = static_cast<int>(leaf_lo);
        int hi = static_cast<int>(leaf_hi) - 1;
        return p < lo ? lo : (p > hi ? hi : p);
    }
};

// ─── Page-Hinkley Drift Detector ─────────────────────────────────────────────
class PageHinkley {
public:
    PageHinkley() = default;
    explicit PageHinkley(double delta, double lambda)
        : delta_(delta), lambda_(lambda) {}

    void reset() noexcept { sum_=0; min_sum_=0; mean_=0; n_=0; triggered_=false; }

    bool update(double v) noexcept {
        ++n_;
        mean_ += (v - mean_) / static_cast<double>(n_);
        sum_  += v - mean_ - delta_;
        if (sum_ < min_sum_) min_sum_ = sum_;
        if ((sum_ - min_sum_) > lambda_) triggered_ = true;
        return triggered_;
    }
    void acknowledge() noexcept { reset(); }
    bool   triggered()  const noexcept { return triggered_; }
    size_t n()          const noexcept { return n_; }
    // [AUDIT] Return sum_-min_sum_ so caller can snapshot it before acknowledge
    double ph_excess()  const noexcept { return sum_ - min_sum_; }

private:
    double delta_ = 1.5, lambda_ = 60.0;
    double sum_ = 0, min_sum_ = 0, mean_ = 0;
    size_t n_ = 0;
    bool   triggered_ = false;
};

// ─── EWMA Drift Detector  [7] WSEWMA support ─────────────────────────────────
class EWMADetector {
public:
    EWMADetector() = default;
    explicit EWMADetector(double alpha, double threshold, double sensitivity = 0.0)
        : alpha_(alpha), threshold_(threshold), sensitivity_(sensitivity) {}

    void reset() noexcept { ewma_=0; n_=0; baseline_=-1.0; triggered_=false; }

    // [WSEWMA] Save state for warm restart after repair
    void save_state(double& out_ewma, double& out_baseline) const noexcept {
        out_ewma     = ewma_;
        out_baseline = baseline_;
    }
    // [WSEWMA] Restore state post-repair — skip 500-query warmup
    void warm_start(double ewma_val, double baseline_val) noexcept {
        ewma_      = ewma_val;
        baseline_  = baseline_val;
        n_         = 501;   // past baseline threshold
        triggered_ = false;
    }

    bool update(double v) noexcept {
        if (n_ == 0) ewma_ = v;
        else         ewma_ = alpha_ * v + (1.0 - alpha_) * ewma_;
        ++n_;
        if (n_ == 500) baseline_ = ewma_;
        if (baseline_ > 0.0) {
            if (sensitivity_ > 0.0) {
                if (ewma_ > baseline_ + sensitivity_) triggered_ = true;
            } else {
                if (ewma_ > threshold_ * baseline_) triggered_ = true;
            }
        }
        return triggered_;
    }
    void acknowledge() noexcept { reset(); }
    bool   triggered()  const noexcept { return triggered_; }
    double ewma()       const noexcept { return ewma_; }
    double baseline()   const noexcept { return baseline_; }
    size_t n()          const noexcept { return n_; }

private:
    double alpha_       = 0.05;
    double threshold_   = 3.5;
    double sensitivity_ = 0.0;
    double ewma_        = 0.0;
    double baseline_    = -1.0;
    size_t n_           = 0;
    bool   triggered_   = false;
};

// ─── PGM shrinking-cone segmentation ─────────────────────────────────────────
static std::vector<LeafSeg>
pgm_build(const std::vector<Key>& keys, size_t eps) {
    const size_t n = keys.size();
    if (n == 0) return {};
    std::vector<LeafSeg> segs;
    segs.reserve(std::max<size_t>(8, n / std::max<size_t>(1, 2*eps) + 16));
    size_t seg_start = 0;
    Key    k0 = keys[0];
    size_t i0 = 0;
    double sl = 0.0, sh = std::numeric_limits<double>::infinity();
    auto emit = [&](size_t end) __attribute__((noinline)) {
        double slope = std::isinf(sh)
            ? (end > seg_start + 1
                ? static_cast<double>(end-1-i0) /
                  (static_cast<double>(keys[end-1]) - static_cast<double>(k0) + 1e-12)
                : 0.0)
            : (sl + sh) * 0.5;
        LeafSeg s;
        s.slope     = slope;
        s.intercept = static_cast<double>(i0) - slope * static_cast<double>(k0);
        s.pos_lo    = static_cast<uint32_t>(seg_start);
        s.pos_hi    = static_cast<uint32_t>(end);
        s.bias      = 0;
        s.seg_eps   = static_cast<uint16_t>(std::min(eps, (size_t)65535u));
        s.seg_errs  = 0;
        s._res      = 0;
        segs.push_back(s);
    };
    for (size_t i = 1; i < n; ++i) {
        double dx   = static_cast<double>(keys[i]) - static_cast<double>(k0);
        if (dx <= 0.0) continue;
        double ipos = static_cast<double>(static_cast<int64_t>(i) - static_cast<int64_t>(i0));
        double epsd = static_cast<double>(eps);
        double ns_lo = (ipos - epsd) / dx;
        double ns_hi = (ipos + epsd) / dx;
        if (i == seg_start + 1) { sl = ns_lo; sh = ns_hi; }
        else {
            double nl = std::max(sl, ns_lo), nh = std::min(sh, ns_hi);
            if (nl > nh + 1e-12) {
                emit(i); seg_start = i; i0 = i; k0 = keys[i];
                sl = 0.0; sh = std::numeric_limits<double>::infinity();
            } else { sl = nl; sh = nh; }
        }
    }
    emit(n);
    return segs;
}

// ─── Root-level routing model ─────────────────────────────────────────────────
static std::vector<RootSeg>
root_build(const std::vector<Key>& seg_keys, size_t root_eps) {
    const size_t m = seg_keys.size();
    if (m == 0) return {};
    if (m == 1) {
        RootSeg r; r.slope=0; r.intercept=0; r.leaf_lo=0; r.leaf_hi=1; return {r};
    }
    std::vector<RootSeg> roots;
    roots.reserve(std::max<size_t>(4, static_cast<size_t>(std::sqrt(m)) + 4));
    size_t seg_start = 0; Key k0 = seg_keys[0]; size_t i0 = 0;
    double sl = 0.0, sh = std::numeric_limits<double>::infinity();
    auto emit = [&](size_t end) {
        double slope = std::isinf(sh)
            ? (end > seg_start + 1
                ? static_cast<double>(end-1-i0) /
                  (static_cast<double>(seg_keys[end-1]) - static_cast<double>(k0) + 1e-12)
                : 0.0)
            : (sl + sh) * 0.5;
        RootSeg r;
        r.slope     = slope;
        r.intercept = static_cast<double>(i0) - slope * static_cast<double>(k0);
        r.leaf_lo   = static_cast<uint32_t>(seg_start);
        r.leaf_hi   = static_cast<uint32_t>(end);
        roots.push_back(r);
    };
    for (size_t i = 1; i < m; ++i) {
        double dx   = static_cast<double>(seg_keys[i]) - static_cast<double>(k0);
        if (dx <= 0.0) continue;
        double ipos = static_cast<double>(static_cast<int64_t>(i) - static_cast<int64_t>(i0));
        double epsd = static_cast<double>(root_eps);
        double ns_lo = (ipos - epsd) / dx, ns_hi = (ipos + epsd) / dx;
        if (i == seg_start + 1) { sl = ns_lo; sh = ns_hi; }
        else {
            double nl = std::max(sl, ns_lo), nh = std::min(sh, ns_hi);
            if (nl > nh + 1e-12) {
                emit(i); seg_start = i; i0 = i; k0 = seg_keys[i];
                sl = 0.0; sh = std::numeric_limits<double>::infinity();
            } else { sl = nl; sh = nh; }
        }
    }
    emit(m);
    return roots;
}

// ─── Branchless binary search (CMOV) ─────────────────────────────────────────
static inline size_t
blb(const Key* __restrict__ arr, size_t lo, size_t hi, Key key) noexcept {
    size_t len = hi - lo;
    const Key* base = arr + lo;
    while (len > 1) {
        size_t half = len >> 1;
        base += (base[half] < key) ? half : 0;
        len  -= half;
    }
    return static_cast<size_t>((base - arr) + (*base < key ? 1 : 0));
}

// ─── [ECS + EBECS] Exponential Centering Search ───────────────────────────────
// Checks exact prediction first (in L1 from Stage-B prefetch),
// then exponentially brackets. O(log(actual_error+1)) expected.
// Called with bias-adjusted p (predict_biased) for higher direct-hit rate.
static inline size_t
ecs_search(const Key* __restrict__ arr, size_t lo, size_t hi,
           size_t p, Key key) noexcept {
    if (__builtin_expect(lo >= hi, 0)) return lo;

    // Fast path: bias-adjusted prediction lands exactly on key
    if (__builtin_expect(arr[p] == key, 0)) return p;

    if (arr[p] < key) {
        // Probe rightward: 1, 2, 4, 8 ...
        size_t step = 1;
        while (p + step < hi && arr[p + step] < key) { step <<= 1; }
        size_t bl = p + (step >> 1);
        size_t br = std::min(p + step + 1, hi);
        return blb(arr, bl, br, key);
    } else {
        // Probe leftward: 1, 2, 4, 8 ...
        if (__builtin_expect(p == lo, 0)) return lo;
        size_t step = 1;
        while (step < p - lo && arr[p - step] > key) { step <<= 1; }
        size_t bl = (step >= p - lo) ? lo : p - step;
        size_t br = p - (step >> 1) + 1;
        return blb(arr, bl, std::min(br, hi), key);
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// NLIIndex v7.0 — Main class
// ═══════════════════════════════════════════════════════════════════════════════
class NLIIndex {
public:
    using key_type    = Key;
    using value_type  = Value;

    // ── [9] GIDB hot buffer constants ──────────────────────────────────────────
    static constexpr size_t HOT_CAP          = 32;
    static constexpr size_t REPAIR_COOLDOWN  = 2000;  // [10] RCO

    // ── Constructors ───────────────────────────────────────────────────────────
    NLIIndex() = default;
    // [AUDIT-FIX BUG-1] Initialise detectors from cfg so ewma_threshold=3.0,
    // ph_delta=1.0, ph_lambda=50.0 are actually used at runtime (previously
    // detectors fell back to their in-class defaults: threshold=3.5, lambda=60).
    explicit NLIIndex(const NLIConfig& cfg)
        : ph_(cfg.ph_delta, cfg.ph_lambda),
          ewma_(cfg.ewma_alpha, cfg.ewma_threshold, cfg.ewma_sensitivity),
          cfg_(cfg) {}

    // ── Build from sorted key/value arrays ────────────────────────────────────
    void build(const std::vector<Key>& keys, const std::vector<Value>& vals) {
        assert(keys.size() == vals.size());
        keys_ = keys; vals_ = vals;
        cached_eps_ = cfg_.epsilon;   // [AFR] initialize cache
        rebuild_routing(/*from_build=*/true);
    }
    // Overload: build from sorted KV vector (used by benchmarks)
    void build(const std::vector<KV>& data) {
        keys_.resize(data.size()); vals_.resize(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            keys_[i] = data[i].key; vals_[i] = data[i].value;
        }
        cached_eps_ = cfg_.epsilon;
        rebuild_routing(/*from_build=*/true);
    }

    // ── Primary lookup (triggers self-repair when drift detectors fire) ──────
    bool lookup(Key key, Value& out) noexcept {
        // [9] GIDB: check hot buffer first (unsorted, linear scan, 32 entries)
        for (size_t i = 0; i < hot_size_; ++i) {
            if (hot_buf_[i].first == key) { out = hot_buf_[i].second; return true; }
        }
        // [9] GIDB: check warm sorted buffer
        if (!buf_.empty()) {
            auto it = std::lower_bound(buf_.begin(), buf_.end(), key,
                [](const auto& p, Key k){ return p.first < k; });
            if (it != buf_.end() && it->first == key) { out = it->second; return true; }
        }
        bool found = model_lookup(key, out);
        // Self-repair when drift detected (suppressed during RCO cooldown)
        if (cfg_.drift_detection) {
            bool cooldown_active = cfg_.use_rco && (queries_since_repair_ < REPAIR_COOLDOWN);
            if (!cooldown_active && (ph_.triggered() || ewma_.triggered())) {
                // [AUDIT-FIX BUG-2] Save EWMA state BEFORE acknowledge so that
                // WSEWMA warm_start() in rebuild_routing()/ldz_partial_repair()
                // restores the live pre-repair baseline, not the post-reset zeros.
                ewma_.save_state(saved_ewma_, saved_baseline_);
                last_trigger_ewma_       = ewma_.ewma();
                last_trigger_ph_excess_  = ph_.ph_excess();
                ph_.acknowledge(); ewma_.acknowledge();
                if (cfg_.perform_repair) {
                    if (cfg_.selective_repair) ldz_partial_repair();
                    else                       rebuild_routing();
                }
            }
        }
        return found;
    }

    // ── [12] MLH: Monotone Lookup Hint (bypass routing for sequential scans) ──
    // Caller provides an approximate segment index. If it brackets key, skip routing.
    // Returns {found, next_seg_hint}. next_seg_hint is valid for the next key in order.
    std::pair<bool, size_t> lookup_hint(Key key, size_t seg_hint, Value& out) noexcept {
        const size_t nsegs = leaves_.size();
        if (seg_hint < nsegs) {
            const LeafSeg& leaf = leaves_[seg_hint];
            // Check if seg_hint brackets this key (seg_keys_[seg_hint] <= key)
            if (seg_keys_[seg_hint] <= key &&
                (seg_hint + 1 >= nsegs || seg_keys_[seg_hint + 1] > key)) {
                // Direct leaf search — skip root routing
                size_t sidx = seg_hint;
                size_t p = static_cast<size_t>(leaf.predict_biased(key));
                size_t lo = leaf.pos_lo, hi = leaf.pos_hi;
                if (__builtin_expect(p < lo, 0)) p = lo;
                if (__builtin_expect(p >= hi, 0)) p = hi - 1;
                __builtin_prefetch(&keys_[p], 0, 1);
                __builtin_prefetch(&vals_[p], 0, 1);
                uint32_t eps = leaf.seg_eps;
                size_t wlo = (p >= lo + eps) ? p - eps : lo;
                size_t whi = std::min(p + eps + 1, hi);
                size_t pos = ecs_search(keys_.data(), wlo, whi, p, key);
                if (pos < hi && keys_[pos] == key) {
                    out = vals_[pos];
                    track_hit(static_cast<uint32_t>(pos > p ? pos-p : p-pos), sidx);
                    return {true, sidx};  // same segment for close sequential keys
                }
                return {false, sidx};
            }
        }
        // Fallback to full lookup
        bool found = model_lookup(key, out);
        // Find next seg hint
        size_t hint = 0;
        if (!leaves_.empty()) {
            auto it = std::upper_bound(seg_keys_.begin(), seg_keys_.end(), key);
            hint = (it == seg_keys_.begin()) ? 0
                 : static_cast<size_t>(it - seg_keys_.begin()) - 1;
        }
        return {found, hint};
    }

    // ── Insert ─────────────────────────────────────────────────────────────────
    void insert(Key key, Value val) {
        // [9] GIDB: insert into hot_buf_ first (O(1))
        if (hot_size_ < HOT_CAP) {
            hot_buf_[hot_size_++] = {key, val};
            return;
        }
        // hot_buf_ full: merge into warm sorted buf_
        merge_hot_to_warm();
        hot_buf_[0] = {key, val};
        hot_size_ = 1;

        // Flush warm buf to model if ratio exceeded
        if (!keys_.empty() &&
            buf_.size() >= static_cast<size_t>(keys_.size() * cfg_.flush_ratio)) {
            flush_buffer();
        }
    }

    // ── Statistics & introspection ─────────────────────────────────────────────
    size_t num_segments()   const noexcept { return leaves_.size(); }
    size_t n_segments()     const noexcept { return leaves_.size(); }
    size_t num_root_segs()  const noexcept { return roots_.size(); }
    size_t size()           const noexcept { return keys_.size() + buf_.size() + hot_size_; }
    size_t model_size_bytes() const noexcept {
        return leaves_.size() * sizeof(LeafSeg)
             + seg_keys_.size() * sizeof(Key)
             + roots_.size() * sizeof(RootSeg)
             + root_keys_.size() * sizeof(Key);
    }
    // Total memory: model overhead + stored data arrays (fair comparison with other indexes)
    // [AUDIT-FIX MEM-1] hot_buf_ is std::array<...,HOT_CAP> — always occupies HOT_CAP
    // slots regardless of hot_size_. Count full fixed-size allocation, not just used slots.
    size_t memory_bytes() const noexcept {
        return model_size_bytes()
             + keys_.size() * sizeof(Key)
             + vals_.size() * sizeof(Value)
             + buf_.size()  * (sizeof(Key) + sizeof(Value))
             + HOT_CAP      * (sizeof(Key) + sizeof(Value));  // fixed array, always allocated
    }
    size_t n_repairs()    const noexcept { return repair_count_; }

    // ── Force repair (external call from drift_benchmark) ─────────────────────
    void force_rebuild()  { rebuild_routing(); }
    void force_repair()   { ldz_partial_repair(); }

    // ── Internal drift state (exposed for benchmark probing) ──────────────────
    bool drift_triggered()   const noexcept { return ph_.triggered() || ewma_.triggered(); }
    size_t repair_count()    const noexcept { return repair_count_; }

    // ── [AUDIT] Snapshot of detector state saved just before last acknowledge ──
    double last_trigger_ewma()       const noexcept { return last_trigger_ewma_; }
    double last_trigger_ph_excess()  const noexcept { return last_trigger_ph_excess_; }

    // ── [AUDIT] Hit prediction-error accumulator — reset to measure windows ───
    double mean_abs_error() const noexcept {
        return err_count_ > 0 ? err_sum_ / static_cast<double>(err_count_) : 0.0;
    }
    void reset_error_stats() const noexcept { err_sum_ = 0.0; err_count_ = 0; }

    // ── Process a query with full drift pipeline (alias for lookup) ─────────────
    // [RF-10 FIX] Previous version double-incremented queries_since_repair_ and
    // double-checked drift after lookup() already handled it. Now a pure alias.
    bool query_with_drift(Key key, Value& out) {
        return lookup(key, out);
    }

private:
    // ── Data ───────────────────────────────────────────────────────────────────
    std::vector<Key>     keys_;
    std::vector<Value>   vals_;

    // ── Routing ────────────────────────────────────────────────────────────────
    std::vector<RootSeg> roots_;
    std::vector<Key>     root_keys_;
    std::vector<LeafSeg> leaves_;
    std::vector<Key>     seg_keys_;

    // ── [4] SRLM-2: two sub-models for O(1) root prediction ──────────────────
    double   super_slope_1_     = 0.0, super_intercept_1_ = 0.0;  // left half
    double   super_slope_2_     = 0.0, super_intercept_2_ = 0.0;  // right half
    size_t   super_mid_         = 0;                               // split index
    Key      super_mid_key_     = 0;                               // split key value
    // Fallback single-model (for ablation: use_srlm2=false)
    double   super_slope_       = 0.0, super_intercept_    = 0.0;

    // ── [5] BREC: tight root epsilon ──────────────────────────────────────────
    uint32_t tight_root_eps_    = 8;

    // ── [11] AFR: cached epsilon ───────────────────────────────────────────────
    size_t   cached_eps_        = 8;

    // ── Drift detectors ────────────────────────────────────────────────────────
    mutable PageHinkley  ph_;
    mutable EWMADetector ewma_;

    // ── [7] WSEWMA saved state ─────────────────────────────────────────────────
    double   saved_ewma_        = 0.0;
    double   saved_baseline_    = 0.0;

    // ── [9] GIDB: dual buffer ──────────────────────────────────────────────────
    std::array<std::pair<Key,Value>, HOT_CAP> hot_buf_{};
    size_t                                    hot_size_ = 0;
    std::vector<std::pair<Key,Value>>         buf_;

    // ── [10] RCO ──────────────────────────────────────────────────────────────
    mutable size_t queries_since_repair_ = 0;

    // ── Stats ──────────────────────────────────────────────────────────────────
    mutable size_t repair_count_ = 0;

    // ── [AUDIT] Pre-acknowledge detector snapshot ──────────────────────────────
    // Saved inside lookup() BEFORE ph_/ewma_ are reset, so the drift benchmark
    // can read meaningful values after detect_at rather than always 0.000.
    mutable double last_trigger_ewma_      = 0.0;
    mutable double last_trigger_ph_excess_ = 0.0;

    // ── [AUDIT] Running hit prediction-error accumulator ──────────────────────
    // Allows drift benchmark to compare model accuracy before/after repair.
    mutable double err_sum_   = 0.0;
    mutable size_t err_count_ = 0;

    // ── Config ─────────────────────────────────────────────────────────────────
    NLIConfig cfg_;

    // ═══════════════════════════════════════════════════════════════════════════
    // rebuild_routing(): full rebuild with BTSC, BREC, SRLM-2
    // ═══════════════════════════════════════════════════════════════════════════
    void rebuild_routing(bool from_build = false) {
        if (keys_.empty()) return;

        // [11] AFR: adaptive epsilon search, starting from cached_eps_
        size_t eps_try = cached_eps_;
        size_t target_segs = std::max<size_t>(4, static_cast<size_t>(
            std::sqrt(static_cast<double>(keys_.size()))));

        // Build leaf segments with adaptive eps
        std::vector<LeafSeg> candidate;
        for (int attempt = 0; attempt < 6; ++attempt) {
            candidate = pgm_build(keys_, eps_try);
            if (candidate.size() >= target_segs || eps_try <= 2) break;
            eps_try = std::max<size_t>(2, eps_try / 2);
        }
        if (candidate.size() < 2 && eps_try > cfg_.epsilon) {
            candidate = pgm_build(keys_, cfg_.epsilon);
            eps_try   = cfg_.epsilon;
        }
        cached_eps_ = eps_try;  // [AFR] cache for next rebuild

        // [2] BTSC: measure actual per-segment errors across training data
        // Sweep O(n): for each key find its owning segment, compute prediction error.
        if (cfg_.use_btsc && candidate.size() > 1) {
            struct SegStat { int64_t sum_err=0; int max_abs=0; size_t cnt=0; };
            std::vector<SegStat> stats(candidate.size());

            size_t seg_idx = 0;
            const size_t ns = candidate.size();
            for (size_t i = 0; i < keys_.size(); ++i) {
                Key k = keys_[i];
                // Advance seg_idx if key crosses segment boundary
                while (seg_idx + 1 < ns &&
                       static_cast<size_t>(i) >= candidate[seg_idx+1].pos_lo) {
                    ++seg_idx;
                }
                const LeafSeg& s = candidate[seg_idx];
                int pred    = s.predict(k);
                int actual  = static_cast<int>(i);
                int err     = pred - actual;           // signed
                int abs_err = err < 0 ? -err : err;
                stats[seg_idx].sum_err += err;
                if (abs_err > stats[seg_idx].max_abs) stats[seg_idx].max_abs = abs_err;
                ++stats[seg_idx].cnt;
            }
            // Set per-segment bias and seg_eps from measured statistics
            for (size_t s = 0; s < candidate.size(); ++s) {
                if (stats[s].cnt > 0) {
                    int mean_err = static_cast<int>(stats[s].sum_err /
                                   static_cast<int64_t>(stats[s].cnt));
                    // Clamp bias to int16_t range
                    candidate[s].bias    = static_cast<int16_t>(
                        std::max(-32767, std::min(32767, -mean_err)));
                    // seg_eps = max observed error + 1 (tight guaranteed window)
                    int tight = stats[s].max_abs + 1;
                    candidate[s].seg_eps = static_cast<uint16_t>(
                        std::min(tight, 65535));
                } else {
                    candidate[s].bias    = 0;
                    candidate[s].seg_eps = static_cast<uint16_t>(eps_try);
                }
            }
        }

        leaves_   = std::move(candidate);

        // Build seg_keys_ (first key of each leaf segment)
        seg_keys_.resize(leaves_.size());
        for (size_t i = 0; i < leaves_.size(); ++i) {
            seg_keys_[i] = keys_[leaves_[i].pos_lo];
        }

        // Build root model over seg_keys_
        roots_     = root_build(seg_keys_, cfg_.root_epsilon);
        root_keys_.resize(roots_.size());
        for (size_t i = 0; i < roots_.size(); ++i) {
            root_keys_[i] = seg_keys_[roots_[i].leaf_lo];
        }

        const size_t nr = root_keys_.size();

        // [4] SRLM-2: fit two linear models over root_keys_
        if (cfg_.use_srlm2 && nr >= 4) {
            super_mid_ = nr / 2;
            super_mid_key_ = root_keys_[super_mid_];

            // Model-L: indices 0..mid
            {
                double sx=0,sy=0,sxx=0,sxy=0;
                size_t m = super_mid_ + 1;
                for (size_t i = 0; i <= super_mid_; ++i) {
                    double x = static_cast<double>(root_keys_[i]);
                    double y = static_cast<double>(i);
                    sx+=x; sy+=y; sxx+=x*x; sxy+=x*y;
                }
                double dm = static_cast<double>(m);
                double denom = dm*sxx - sx*sx;
                if (std::abs(denom) < 1e-10) {
                    super_slope_1_ = 0; super_intercept_1_ = 0;
                } else {
                    super_slope_1_     = (dm*sxy - sx*sy) / denom;
                    super_intercept_1_ = (sy - super_slope_1_*sx) / dm;
                }
            }
            // Model-R: indices mid..nr-1
            {
                double sx=0,sy=0,sxx=0,sxy=0;
                size_t m = nr - super_mid_;
                for (size_t i = super_mid_; i < nr; ++i) {
                    double x = static_cast<double>(root_keys_[i]);
                    double y = static_cast<double>(i);
                    sx+=x; sy+=y; sxx+=x*x; sxy+=x*y;
                }
                double dm = static_cast<double>(m);
                double denom = dm*sxx - sx*sx;
                if (std::abs(denom) < 1e-10) {
                    super_slope_2_ = 0; super_intercept_2_ = 0;
                } else {
                    super_slope_2_     = (dm*sxy - sx*sy) / denom;
                    super_intercept_2_ = (sy - super_slope_2_*sx) / dm;
                }
            }
        } else {
            // Single model fallback
            super_mid_ = 0;
            if (nr >= 2) {
                double sx=0,sy=0,sxx=0,sxy=0;
                double dm = static_cast<double>(nr);
                for (size_t i = 0; i < nr; ++i) {
                    double x = static_cast<double>(root_keys_[i]);
                    double y = static_cast<double>(i);
                    sx+=x; sy+=y; sxx+=x*x; sxy+=x*y;
                }
                double denom = dm*sxx - sx*sx;
                if (std::abs(denom) < 1e-10) {
                    super_slope_ = 0; super_intercept_ = 0;
                } else {
                    super_slope_     = (dm*sxy - sx*sy) / denom;
                    super_intercept_ = (sy - super_slope_*sx) / dm;
                }
            }
        }

        // [5] BREC: measure actual root model prediction error
        if (nr > 0) {
            uint32_t max_root_err = 0;
            for (size_t i = 0; i < nr; ++i) {
                const RootSeg& rs = roots_[i];
                int pred_leaf = rs.predict_leaf(root_keys_[i]);
                int actual_leaf = static_cast<int>(rs.leaf_lo);
                uint32_t err = static_cast<uint32_t>(
                    std::abs(pred_leaf - actual_leaf));
                if (err > max_root_err) max_root_err = err;
            }
            tight_root_eps_ = max_root_err + 1;
            if (tight_root_eps_ < 2) tight_root_eps_ = 2;
        }

        // Reset detectors with WSEWMA warm-start
        bool has_saved = (saved_baseline_ > 0.0);
        if (has_saved) {
            ph_.reset();  // PH always resets (sudden-shift detector)
            ewma_.warm_start(saved_ewma_, saved_baseline_);  // [7] WSEWMA
        } else {
            ph_.reset();
            ewma_.reset();
        }

        // [10] RCO: reset cooldown counter
        queries_since_repair_ = 0;
        // [RF-01b FIX] Initial build is not a self-repair — skip counter.
        if (!from_build) ++repair_count_;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ldz_partial_repair(): [8] LDZED — targeted segment rebuild
    // ═══════════════════════════════════════════════════════════════════════════
    void ldz_partial_repair() {
        if (keys_.empty() || leaves_.empty()) { rebuild_routing(); return; }

        // [7] WSEWMA: state already saved in lookup() before acknowledge().
        // saved_ewma_ and saved_baseline_ hold the pre-repair values.

        // Identify hot segments (seg_errs above threshold)
        constexpr uint16_t LDZ_THRESHOLD = 5;
        bool any_rebuilt = false;

        for (size_t s = 0; s < leaves_.size(); ++s) {
            if (leaves_[s].seg_errs < LDZ_THRESHOLD) continue;

            // Rebuild this segment locally: re-run PGM on its key slice
            size_t lo = leaves_[s].pos_lo;
            size_t hi = leaves_[s].pos_hi;
            if (lo >= hi || hi > keys_.size()) continue;

            std::vector<Key> slice(keys_.begin() + lo, keys_.begin() + hi);
            std::vector<LeafSeg> new_segs = pgm_build(slice, cached_eps_);

            // [AUDIT-FIX BUG-4] Apply BTSC calibration to rebuilt segments
            // BEFORE patching absolute positions (predict uses local indices here).
            if (cfg_.use_btsc && new_segs.size() > 0) {
                struct SegStat { int64_t sum_err=0; int max_abs=0; size_t cnt=0; };
                std::vector<SegStat> stats(new_segs.size());
                size_t sidx = 0;
                const size_t nsc = new_segs.size();
                for (size_t ki = 0; ki < slice.size(); ++ki) {
                    while (sidx + 1 < nsc && ki >= new_segs[sidx+1].pos_lo) ++sidx;
                    int pred    = new_segs[sidx].predict(slice[ki]);
                    int actual  = static_cast<int>(ki);
                    int err     = pred - actual;
                    int abs_err = err < 0 ? -err : err;
                    stats[sidx].sum_err += err;
                    if (abs_err > stats[sidx].max_abs) stats[sidx].max_abs = abs_err;
                    ++stats[sidx].cnt;
                }
                for (size_t si = 0; si < nsc; ++si) {
                    if (stats[si].cnt > 0) {
                        int me = static_cast<int>(stats[si].sum_err /
                                 static_cast<int64_t>(stats[si].cnt));
                        new_segs[si].bias    = static_cast<int16_t>(
                            std::max(-32767, std::min(32767, -me)));
                        new_segs[si].seg_eps = static_cast<uint16_t>(
                            std::min(stats[si].max_abs + 1, 65535));
                    }
                }
            }

            // Patch absolute positions back into rebuilt segments
            for (auto& ns : new_segs) {
                ns.pos_lo += static_cast<uint32_t>(lo);
                ns.pos_hi += static_cast<uint32_t>(lo);
                ns.seg_errs = 0;
            }

            // Splice: replace single segment at index s with new_segs
            // (may be 1→1 or 1→few for badly drifted zones)
            if (new_segs.size() == 1) {
                leaves_[s]   = new_segs[0];
                seg_keys_[s] = keys_[lo];
            } else {
                leaves_.erase(leaves_.begin() + s);
                leaves_.insert(leaves_.begin() + s, new_segs.begin(), new_segs.end());
                seg_keys_.erase(seg_keys_.begin() + s);
                std::vector<Key> new_sk;
                for (auto& ns : new_segs) new_sk.push_back(keys_[ns.pos_lo]);
                seg_keys_.insert(seg_keys_.begin() + s, new_sk.begin(), new_sk.end());
                s += new_segs.size() - 1;  // skip inserted segments
            }
            any_rebuilt = true;
        }

        // [8] LDZED: halve ALL seg_errs counters (preserve adjacent signal)
        for (auto& leaf : leaves_) {
            leaf.seg_errs >>= 1;
        }

        if (any_rebuilt) {
            // Rebuild root model over updated seg_keys_
            roots_     = root_build(seg_keys_, cfg_.root_epsilon);
            root_keys_.resize(roots_.size());
            for (size_t i = 0; i < roots_.size(); ++i) {
                root_keys_[i] = seg_keys_[roots_[i].leaf_lo];
            }
            // Recompute SRLM-2 (root_keys_ changed)
            const size_t nr = root_keys_.size();
            if (cfg_.use_srlm2 && nr >= 4) {
                // re-fit both sub-models (same logic as rebuild_routing)
                super_mid_ = nr / 2;
                super_mid_key_ = root_keys_[super_mid_];
                // Model-L
                { double sx=0,sy=0,sxx=0,sxy=0; size_t m=super_mid_+1;
                  for (size_t i=0;i<=super_mid_;++i){double x=static_cast<double>(root_keys_[i]),y=static_cast<double>(i);sx+=x;sy+=y;sxx+=x*x;sxy+=x*y;}
                  double dm=static_cast<double>(m),den=dm*sxx-sx*sx;
                  if(std::abs(den)<1e-10){super_slope_1_=0;super_intercept_1_=0;}
                  else{super_slope_1_=(dm*sxy-sx*sy)/den;super_intercept_1_=(sy-super_slope_1_*sx)/dm;} }
                // Model-R
                { double sx=0,sy=0,sxx=0,sxy=0; size_t m=nr-super_mid_;
                  for (size_t i=super_mid_;i<nr;++i){double x=static_cast<double>(root_keys_[i]),y=static_cast<double>(i);sx+=x;sy+=y;sxx+=x*x;sxy+=x*y;}
                  double dm=static_cast<double>(m),den=dm*sxx-sx*sx;
                  if(std::abs(den)<1e-10){super_slope_2_=0;super_intercept_2_=0;}
                  else{super_slope_2_=(dm*sxy-sx*sy)/den;super_intercept_2_=(sy-super_slope_2_*sx)/dm;} }
            }
            // BREC for updated roots
            uint32_t max_root_err = 0;
            for (size_t i = 0; i < roots_.size(); ++i) {
                int pred_leaf = roots_[i].predict_leaf(root_keys_[i]);
                uint32_t err = static_cast<uint32_t>(
                    std::abs(pred_leaf - static_cast<int>(roots_[i].leaf_lo)));
                if (err > max_root_err) max_root_err = err;
            }
            tight_root_eps_ = std::max(2u, max_root_err + 1);

            // [AUDIT-FIX BUG-3] Apply WSEWMA warm-start in the partial-rebuild path.
            // Previously warm_start() was only called inside rebuild_routing(), so
            // the any_rebuilt=true branch always cold-reset the EWMA detector.
            bool has_saved = (saved_baseline_ > 0.0);
            if (has_saved) {
                ph_.reset();
                ewma_.warm_start(saved_ewma_, saved_baseline_);
            } else {
                ph_.reset();
                ewma_.reset();
            }
        } else {
            rebuild_routing();  // no hot segments → full rebuild (handles WSEWMA internally)
        }

        // [10] RCO: reset cooldown
        queries_since_repair_ = 0;
        ++repair_count_;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // model_lookup(): the hot query path
    // ═══════════════════════════════════════════════════════════════════════════
    bool model_lookup(Key key, Value& out) noexcept {
        if (__builtin_expect(keys_.empty(), 0)) return false;
        const size_t nr = root_keys_.size();
        if (__builtin_expect(nr == 0, 0)) return false;

        // ── Step 1: SRLM prediction → candidate root segment index rlo ────────
        int rlo;
        if (cfg_.use_srlm2 && nr >= 4) {
            // [4] SRLM-2: select sub-model based on key vs median key
            if (key < super_mid_key_) {
                rlo = static_cast<int>(super_slope_1_ * static_cast<double>(key)
                                       + super_intercept_1_);
                int lo_clamp = 0, hi_clamp = static_cast<int>(super_mid_);
                if (rlo < lo_clamp) rlo = lo_clamp;
                if (rlo > hi_clamp) rlo = hi_clamp;
            } else {
                rlo = static_cast<int>(super_slope_2_ * static_cast<double>(key)
                                       + super_intercept_2_);
                int lo_clamp = static_cast<int>(super_mid_);
                int hi_clamp = static_cast<int>(nr) - 1;
                if (rlo < lo_clamp) rlo = lo_clamp;
                if (rlo > hi_clamp) rlo = hi_clamp;
            }
        } else {
            // Single SRLM model
            rlo = static_cast<int>(super_slope_ * static_cast<double>(key)
                                   + super_intercept_);
            if (rlo < 0) rlo = 0;
            if (rlo >= static_cast<int>(nr)) rlo = static_cast<int>(nr) - 1;
        }

        // ── Step 2: fine correction scan (tight_root_eps_ bound) [5] BREC ──
        int nr_i = static_cast<int>(nr);
        int eps_r = static_cast<int>(tight_root_eps_);
        int scan_lo = std::max(0, rlo - eps_r);
        int scan_hi = std::min(nr_i - 1, rlo + eps_r);
        while (scan_lo < scan_hi && root_keys_[scan_lo + 1] <= key) ++scan_lo;
        while (scan_lo > 0      && root_keys_[scan_lo]     >  key) --scan_lo;
        rlo = scan_lo;

        // ── Step 3: root segment → leaf prediction ────────────────────────────
        const RootSeg& rs = roots_[rlo];
        int leaf_pred = rs.predict_leaf(key);
        size_t nseg = leaves_.size();

        // ── Stage-A prefetch [6] SPP+: both seg_keys_ and leaves_ ─────────────
        __builtin_prefetch(&seg_keys_[leaf_pred], 0, 2);
        __builtin_prefetch(&leaves_[leaf_pred],   0, 2);

        // ── Step 4: blb in seg_keys_ within [leaf_lo, leaf_hi) ───────────────
        size_t slo = rs.leaf_lo, shi = rs.leaf_hi;
        size_t lp  = static_cast<size_t>(leaf_pred);
        if (lp < slo) lp = slo;
        if (lp >= shi) lp = shi - 1;

        // Use tight_root_eps_ to narrow blb window
        size_t blb_lo = (lp >= slo + eps_r) ? lp - eps_r : slo;
        size_t blb_hi = std::min(lp + static_cast<size_t>(eps_r) + 1, shi);
        size_t sidx   = blb(seg_keys_.data(), blb_lo, blb_hi, key);
        if (sidx > 0) --sidx;
        if (sidx >= nseg) sidx = nseg - 1;
        // Guard: seg_keys_[sidx] <= key < seg_keys_[sidx+1]
        while (sidx + 1 < nseg && seg_keys_[sidx + 1] <= key) ++sidx;
        while (sidx > 0         && seg_keys_[sidx]     >  key) --sidx;

        // ── Step 5: leaf predict [1][3] ILC + EBECS ───────────────────────────
        const LeafSeg& leaf = leaves_[sidx];
        // [EBECS] use bias-corrected prediction for higher direct-hit rate
        size_t p = static_cast<size_t>(leaf.predict_biased(key));
        size_t klo = leaf.pos_lo, khi = leaf.pos_hi;
        if (__builtin_expect(p < klo, 0)) p = klo;
        if (__builtin_expect(p >= khi, 0)) p = khi - 1;

        // ── Stage-B prefetch [6] SPP+ ─────────────────────────────────────────
        __builtin_prefetch(&keys_[p], 0, 1);
        __builtin_prefetch(&vals_[p], 0, 1);

        // ── Step 6: CAR window from ILC seg_eps ───────────────────────────────
        size_t eps_w = leaf.seg_eps;
        size_t wlo   = (p >= klo + eps_w) ? p - eps_w : klo;
        size_t whi   = std::min(p + eps_w + 1, khi);

        // ── Step 7: ECS search ─────────────────────────────────────────────────
        size_t pos = ecs_search(keys_.data(), wlo, whi, p, key);

        if (pos < khi && keys_[pos] == key) {
            out = vals_[pos];
            uint32_t abs_err = static_cast<uint32_t>(pos >= p ? pos - p : p - pos);
            track_hit(abs_err, sidx);
            return true;
        }
        // Miss path: optionally signal to drift detectors via track_misses.
        // A miss is treated as a maximum-error event (key not in model → huge prediction error).
        // This allows EWMA/PH to detect rising miss-rate as a drift signal.
        if (cfg_.track_misses) {
            ++queries_since_repair_;
            // [AUDIT-FIX] Apply the same 16-query sampling gate as track_hit().
            // Old code updated EVERY miss with error=4096, creating an inconsistent
            // signal scale vs the sampled hit path. New: sample every 16 queries
            // and use miss_signal=30, which is significantly above typical hit
            // errors (~2-5) but not astronomically large. This keeps the EWMA
            // on a consistent scale: one miss sample raises EWMA from baseline ~4
            // to ~5.3 (not immediately triggering), and sustained misses accumulate.
            if ((queries_since_repair_ & 15) == 0) {
                constexpr double miss_signal = 30.0;
                ewma_.update(miss_signal);
                ph_.update(miss_signal);
            }
        }
        return false;
    }

    // ── track_hit: update adaptive windows and drift detectors ────────────────
    void track_hit(uint32_t abs_err, size_t sidx) noexcept {
        // [10] RCO: count queries since last repair
        ++queries_since_repair_;

        // [1] ILC CAR: update seg_eps adaptively (expand fast, shrink slow)
        LeafSeg& leaf = leaves_[sidx];  // [AUDIT-FIX] const_cast was unnecessary (leaves_ is non-const)
        if (abs_err >= leaf.seg_eps) {
            uint32_t new_eps = std::min(static_cast<uint32_t>(abs_err) + 4,
                                        static_cast<uint32_t>(65535));
            leaf.seg_eps = static_cast<uint16_t>(new_eps);
        } else if (leaf.seg_eps > 1 && (queries_since_repair_ & 127) == 0) {
            // Gradual decay every 128 queries
            leaf.seg_eps = static_cast<uint16_t>(
                std::max(1u, static_cast<uint32_t>(leaf.seg_eps) - 1));
        }

        // [AUDIT] Accumulate hit prediction error for pre/post-repair accuracy comparison.
        err_sum_   += static_cast<double>(abs_err);
        ++err_count_;

        // [RF-09 FIX] Sample drift detectors every 16 queries (saves ~10-15 ns/query).
        if ((queries_since_repair_ & 15) == 0) {
            double v = static_cast<double>(abs_err);
            ewma_.update(v);
            ph_.update(v);
        }

        // [8] LDZ: count high-error events per segment
        if (abs_err > leaf.seg_eps / 2 && leaf.seg_errs < 65535) {
            ++leaf.seg_errs;
        }
    }

    // ── [9] GIDB: merge hot_buf_ into sorted warm buf_ ────────────────────────
    void merge_hot_to_warm() {
        // Sort hot entries then merge with buf_
        std::vector<std::pair<Key,Value>> tmp(hot_buf_.begin(),
                                              hot_buf_.begin() + hot_size_);
        std::sort(tmp.begin(), tmp.end(),
                  [](const auto& a, const auto& b){ return a.first < b.first; });
        std::vector<std::pair<Key,Value>> merged;
        merged.reserve(buf_.size() + tmp.size());
        std::merge(buf_.begin(), buf_.end(), tmp.begin(), tmp.end(), 
                   std::back_inserter(merged),
                   [](const auto& a, const auto& b){ return a.first < b.first; });
        buf_ = std::move(merged);
        hot_size_ = 0;
    }

    // ── Flush sorted warm buffer into model ───────────────────────────────────
    void flush_buffer() {
        if (hot_size_ > 0) merge_hot_to_warm();
        if (buf_.empty()) return;
        std::vector<Key>   new_keys; new_keys.reserve(keys_.size() + buf_.size());
        std::vector<Value> new_vals; new_vals.reserve(vals_.size() + buf_.size());
        size_t i = 0, j = 0;
        while (i < keys_.size() && j < buf_.size()) {
            if (keys_[i] < buf_[j].first) {
                new_keys.push_back(keys_[i]); new_vals.push_back(vals_[i]); ++i;
            } else {
                new_keys.push_back(buf_[j].first); new_vals.push_back(buf_[j].second); ++j;
            }
        }
        while (i < keys_.size()) { new_keys.push_back(keys_[i]); new_vals.push_back(vals_[i]); ++i; }
        while (j < buf_.size())  { new_keys.push_back(buf_[j].first); new_vals.push_back(buf_[j].second); ++j; }
        keys_ = std::move(new_keys); vals_ = std::move(new_vals);
        buf_.clear();
        rebuild_routing();
    }
};


// ═══════════════════════════════════════════════════════════════════════════════
// Factory functions — returned NLIIndex objects with specific configs
// ═══════════════════════════════════════════════════════════════════════════════

// ── Default v7.0: all 12 innovations enabled ──────────────────────────────────
inline NLIIndex make_nli(size_t epsilon = 8) {
    NLIConfig cfg;
    cfg.epsilon         = epsilon;
    cfg.root_epsilon    = std::max<size_t>(4, epsilon / 2);
    cfg.use_srlm2       = true;
    cfg.use_btsc        = true;
    cfg.use_rco         = true;
    cfg.drift_detection = true;
    cfg.selective_repair= true;
    cfg.ewma_sensitivity= 0.0;    // [AUDIT-FIX] multiplicative mode
    cfg.ewma_threshold  = 3.0;    // [AUDIT-FIX] fire at 3× stable baseline
    cfg.ph_delta        = 1.0;    // [AUDIT-FIX] ignore noise < 1 unit
    cfg.ph_lambda       = 50.0;   // [AUDIT-FIX] require 50-unit sustained excess
    return NLIIndex(cfg);
}

// ── NLI-NoSRLM2: ablation — single SRLM model ─────────────────────────────────
inline NLIIndex make_nli_no_srlm2(size_t epsilon = 8) {
    NLIConfig cfg;
    cfg.epsilon         = epsilon;
    cfg.root_epsilon    = std::max<size_t>(4, epsilon / 2);
    cfg.use_srlm2       = false;  // disable SRLM-2 [4]
    cfg.use_btsc        = true;
    cfg.use_rco         = true;
    cfg.drift_detection = true;
    cfg.selective_repair= true;
    cfg.ewma_sensitivity= 0.0;    // [AUDIT-FIX] multiplicative mode
    cfg.ewma_threshold  = 3.0;    // [AUDIT-FIX] fire at 3× stable baseline
    cfg.ph_delta        = 1.0;    // [AUDIT-FIX] ignore noise < 1 unit
    cfg.ph_lambda       = 50.0;   // [AUDIT-FIX] require 50-unit sustained excess
    return NLIIndex(cfg);
}

// ── NLI-NoBTSC: ablation — no build-time calibration ─────────────────────────
inline NLIIndex make_nli_no_btsc(size_t epsilon = 8) {
    NLIConfig cfg;
    cfg.epsilon         = epsilon;
    cfg.root_epsilon    = std::max<size_t>(4, epsilon / 2);
    cfg.use_srlm2       = true;
    cfg.use_btsc        = false;  // disable BTSC [2]
    cfg.use_rco         = true;
    cfg.drift_detection = true;
    cfg.selective_repair= true;
    cfg.ewma_sensitivity= 0.0;    // [AUDIT-FIX] multiplicative mode
    cfg.ewma_threshold  = 3.0;    // [AUDIT-FIX] fire at 3× stable baseline
    cfg.ph_delta        = 1.0;    // [AUDIT-FIX] ignore noise < 1 unit
    cfg.ph_lambda       = 50.0;   // [AUDIT-FIX] require 50-unit sustained excess
    return NLIIndex(cfg);
}

// ── NLI-NoRCO: ablation — no repair cooldown ──────────────────────────────────
inline NLIIndex make_nli_no_rco(size_t epsilon = 8) {
    NLIConfig cfg;
    cfg.epsilon         = epsilon;
    cfg.root_epsilon    = std::max<size_t>(4, epsilon / 2);
    cfg.use_srlm2       = true;
    cfg.use_btsc        = true;
    cfg.use_rco         = false;  // disable RCO [10]
    cfg.drift_detection = true;
    cfg.selective_repair= true;
    cfg.ewma_sensitivity= 0.0;    // [AUDIT-FIX] multiplicative mode
    cfg.ewma_threshold  = 3.0;    // [AUDIT-FIX] fire at 3x stable baseline
    cfg.ph_delta        = 1.0;    // [AUDIT-FIX] ignore noise < 1 unit
    cfg.ph_lambda       = 50.0;   // [AUDIT-FIX] require 50-unit sustained excess
    return NLIIndex(cfg);
}

// ── NLI-NoDrift: ablation — no drift detection (pure model) ──────────────────
inline NLIIndex make_nli_no_drift(size_t epsilon = 8) {
    NLIConfig cfg;
    cfg.epsilon         = epsilon;
    cfg.root_epsilon    = std::max<size_t>(4, epsilon / 2);
    cfg.use_srlm2       = true;
    cfg.use_btsc        = true;
    cfg.use_rco         = false;
    cfg.drift_detection = false;
    cfg.selective_repair= false;
    cfg.ewma_sensitivity= 0.0;
    return NLIIndex(cfg);
}

// ── NLI-NoRepair: ablation — detect drift but no repair ───────────────────────
// [AUDIT-FIX BUG-6] Previously selective_repair=false still triggered
// rebuild_routing() (full repair) on detection.  perform_repair=false now
// truly suppresses any repair while keeping the detection path alive.
inline NLIIndex make_nli_no_repair(size_t epsilon = 8) {
    NLIConfig cfg;
    cfg.epsilon         = epsilon;
    cfg.root_epsilon    = std::max<size_t>(4, epsilon / 2);
    cfg.use_srlm2       = true;
    cfg.use_btsc        = true;
    cfg.use_rco         = false;
    cfg.drift_detection = true;
    cfg.selective_repair= false;
    cfg.perform_repair  = false;   // [AUDIT-FIX BUG-6] detect but truly do NOT repair
    cfg.ewma_sensitivity= 0.0;    // [AUDIT-FIX] multiplicative mode
    cfg.ewma_threshold  = 3.0;    // [AUDIT-FIX] fire at 3x stable baseline
    cfg.ph_delta        = 1.0;    // [AUDIT-FIX] ignore noise < 1 unit
    cfg.ph_lambda       = 50.0;   // [AUDIT-FIX] require 50-unit sustained excess
    return NLIIndex(cfg);
}

// ── NLI-Linear: large epsilon (stress-test model accuracy) ────────────────────
inline NLIIndex make_nli_linear(size_t epsilon = 1024) {
    NLIConfig cfg;
    cfg.epsilon         = epsilon;
    cfg.root_epsilon    = epsilon / 2;
    cfg.use_srlm2       = true;
    cfg.use_btsc        = true;
    cfg.use_rco         = false;
    cfg.drift_detection = false;
    cfg.selective_repair= false;
    return NLIIndex(cfg);
}

// ── NLI-NoPW: no prediction window adaptation ─────────────────────────────────
inline NLIIndex make_nli_no_pw(size_t epsilon = 512) {
    NLIConfig cfg;
    cfg.epsilon         = epsilon;
    cfg.root_epsilon    = epsilon / 2;
    cfg.use_srlm2       = true;
    cfg.use_btsc        = false;
    cfg.use_rco         = false;
    cfg.drift_detection = false;
    cfg.selective_repair= false;
    return NLIIndex(cfg);
}

// -- Benchmark-compatibility aliases ---------------------------------------
inline NLIIndex make_nli_full()         { return make_nli(); }
inline NLIIndex make_nli_linear_only()  { return make_nli_linear(); }

} // namespace nli

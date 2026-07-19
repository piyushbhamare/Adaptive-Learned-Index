/**
 * rmi_index.hpp — Recursive Model Index (RMI)
 *
 * Based on:
 *   Kraska et al., "The Case for Learned Index Structures", SIGMOD 2018.
 *
 * Architecture:
 *   Two-level hierarchy:
 *     Level 0 (Root): single linear model predicting which leaf model to use.
 *     Level 1 (Leaves): L1_SIZE independent linear models, each covering a
 *       partition of the key space.
 *
 *   Build: O(n) — train root, partition keys into leaf models, train each leaf.
 *
 *   Lookup:
 *     1. Root model predicts leaf index i = clamp(slope*key+b, 0, L1-1)
 *     2. Leaf model i predicts position p
 *     3. Bounded search within [p - max_err_i, p + max_err_i]
 *
 *   Error bounds: per-leaf maximum error is stored at build time to bound search.
 *
 * Parameter choices (empirically tuned for SOSD datasets):
 *   L1_SIZE = 1024 leaf models — provides good granularity across 200M keys.
 *   Bounded search window = max_error + 1 (tight bound, no wasted comparisons).
 *
 * Author: NLI Group 19, 2025-26
 */

#pragma once

#include "common.hpp"

namespace nli {

// ============================================================================
// RMI Leaf
// ============================================================================

/**
 * @brief One leaf model in the RMI Level-1.
 *
 * Covers keys_[start .. end) in the global sorted array.
 * Trained via the simple closed-form linear map (same as LinearModel in common.hpp).
 */
struct RMILeaf {
    size_t start    = 0;   ///< First global index covered
    size_t count    = 0;   ///< Number of keys covered
    double slope    = 0.0;
    double intercept= 0.0;
    size_t max_err  = 0;   ///< Maximum absolute prediction error on training data

    /// Predict global position for key
    [[nodiscard]] size_t predict(Key key, size_t n_total) const {
        double p = slope * static_cast<double>(key) + intercept;
        if (p < static_cast<double>(start))       p = static_cast<double>(start);
        if (p >= static_cast<double>(start + count)) p = static_cast<double>(start + count - 1);
        if (p >= static_cast<double>(n_total))    p = static_cast<double>(n_total - 1);
        return static_cast<size_t>(p);
    }

    /// Train leaf on keys[start..start+count) with target positions [start..start+count)
    void train(const Key* keys) {
        if (count == 0) return;
        const Key lo = keys[start];
        const Key hi = keys[start + count - 1];
        if (count == 1 || lo == hi) {
            slope     = 0.0;
            intercept = static_cast<double>(start);
            max_err   = 0;
            return;
        }
        slope     = static_cast<double>(count - 1) / static_cast<double>(hi - lo);
        intercept = static_cast<double>(start) - slope * static_cast<double>(lo);

        // Compute maximum error on training set
        max_err = 0;
        for (size_t i = 0; i < count; ++i) {
            double p   = slope * static_cast<double>(keys[start + i]) + intercept;
            size_t pos = static_cast<size_t>(std::max(0.0, p));
            size_t actual = start + i;
            size_t err = (pos > actual) ? pos - actual : actual - pos;
            if (err > max_err) max_err = err;
        }
    }
};

// ============================================================================
// RMI Index
// ============================================================================

/**
 * @brief Two-level Recursive Model Index.
 *
 * L1_SIZE is the number of leaf models. 1024 is a good default for 100M–200M key datasets.
 */
class RMIIndex {
public:
    static constexpr size_t      DEFAULT_L1 = 1024;
    static constexpr const char* NAME       = "RMI";

    explicit RMIIndex(size_t l1_size = DEFAULT_L1)
        : l1_size_(l1_size) {}

    const char* name() const { return NAME; }

    // -------------------------------------------------------------------------
    // Build
    // -------------------------------------------------------------------------

    /**
     * @brief Bulk build from sorted KV pairs.
     *
     * Algorithm:
     *   1. Train root model: slope = (l1_size-1) / (max_key - min_key)
     *      Maps any key -> leaf index in [0, l1_size-1]
     *   2. Partition keys into l1_size buckets by root prediction
     *   3. For each bucket: train a leaf linear model
     */
    void build(const std::vector<KV>& data) {
        n_ = data.size();
        if (n_ == 0) return;

        keys_.resize(n_);
        vals_.resize(n_);
        for (size_t i = 0; i < n_; ++i) {
            keys_[i] = data[i].key;
            vals_[i] = data[i].value;
        }

        const Key min_k = keys_.front();
        const Key max_k = keys_.back();

        // --- Train root model ---
        if (min_k == max_k) {
            root_slope_     = 0.0;
            root_intercept_ = 0.0;
        } else {
            root_slope_     = static_cast<double>(l1_size_ - 1)
                              / static_cast<double>(max_k - min_k);
            root_intercept_ = -root_slope_ * static_cast<double>(min_k);
        }

        // --- Determine bucket boundaries ---
        // Assign each key to a leaf via root prediction
        std::vector<size_t> leaf_start(l1_size_ + 1, 0);
        for (size_t i = 0; i < n_; ++i) {
            size_t li = root_predict(keys_[i]);
            leaf_start[li + 1] = i + 1;  // track last assignment
        }
        // Convert to first-occurrence boundaries
        // leaf_start[li] = first index with predicted leaf >= li
        std::fill(leaf_start.begin(), leaf_start.end(), 0);
        {
            size_t prev_li = SIZE_MAX;
            for (size_t i = 0; i < n_; ++i) {
                size_t li = root_predict(keys_[i]);
                if (li != prev_li) {
                    // Scan forward to fill any skipped leaves
                    for (size_t l = (prev_li == SIZE_MAX ? 0 : prev_li + 1); l <= li; ++l)
                        leaf_start[l] = i;
                    prev_li = li;
                }
            }
            // Fill remaining leaves
            for (size_t l = (prev_li == SIZE_MAX ? 0 : prev_li + 1); l <= l1_size_; ++l)
                leaf_start[l] = n_;
        }

        // --- Train leaf models ---
        leaves_.resize(l1_size_);
        for (size_t li = 0; li < l1_size_; ++li) {
            leaves_[li].start = leaf_start[li];
            leaves_[li].count = leaf_start[li + 1] - leaf_start[li];
            if (leaves_[li].count > 0)
                leaves_[li].train(keys_.data());
        }
    }

    // -------------------------------------------------------------------------
    // Lookup
    // -------------------------------------------------------------------------

    /**
     * @brief Look up a key via 2-level RMI + bounded search.
     *
     * @return true if found, sets out to associated value
     */
    bool lookup(Key key, Value& out) const {
        if (n_ == 0) return false;

        // Level 0: predict leaf index
        size_t li = root_predict(key);
        const RMILeaf& leaf = leaves_[li];

        if (leaf.count == 0) {
            // Empty leaf — key not present
            return false;
        }

        // Level 1: predict position within global array
        size_t pred   = leaf.predict(key, n_);
        size_t err    = leaf.max_err + 1;  // +1 for safety

        // Bounded binary search
        size_t lo = (pred > err) ? pred - err : 0;
        size_t hi = std::min(pred + err + 1, n_);

        const Key* begin = keys_.data() + lo;
        const Key* end   = keys_.data() + hi;
        const Key* it    = std::lower_bound(begin, end, key);
        if (it != end && *it == key) {
            out = vals_[static_cast<size_t>(it - keys_.data())];
            return true;
        }
        return false;
    }

    // -------------------------------------------------------------------------
    // Insert (buffered rebuild)
    // -------------------------------------------------------------------------

    bool insert(Key key, Value val) {
        pending_.emplace_back(key, val);
        ++n_;
        if (pending_.size() >= l1_size_) {
            flush_pending();
        }
        return true;
    }

    // -------------------------------------------------------------------------
    // Memory
    // -------------------------------------------------------------------------

    size_t memory_bytes() const {
        return keys_.size()   * sizeof(Key)
             + vals_.size()   * sizeof(Value)
             + leaves_.size() * sizeof(RMILeaf);
    }

    size_t size()   const { return n_; }
    size_t l1_size() const { return l1_size_; }

    /// Mean max-error across non-empty leaves (quality metric)
    double mean_max_error() const {
        size_t cnt = 0;
        double sum = 0;
        for (auto& l : leaves_) {
            if (l.count > 0) { sum += l.max_err; ++cnt; }
        }
        return cnt > 0 ? sum / cnt : 0.0;
    }

private:
    size_t              l1_size_;
    size_t              n_  = 0;
    double              root_slope_     = 0.0;
    double              root_intercept_ = 0.0;
    std::vector<Key>    keys_;
    std::vector<Value>  vals_;
    std::vector<RMILeaf> leaves_;
    std::vector<KV>     pending_;

    /// Root model prediction clamped to [0, l1_size-1]
    [[nodiscard]] size_t root_predict(Key key) const {
        double p = root_slope_ * static_cast<double>(key) + root_intercept_;
        if (p < 0)                            return 0;
        if (p >= static_cast<double>(l1_size_)) return l1_size_ - 1;
        return static_cast<size_t>(p);
    }

    void flush_pending() {
        if (pending_.empty()) return;
        std::vector<KV> merged;
        merged.reserve(keys_.size() + pending_.size());
        for (size_t i = 0; i < keys_.size(); ++i)
            merged.emplace_back(keys_[i], vals_[i]);
        for (auto& kv : pending_) merged.push_back(kv);
        pending_.clear();
        std::sort(merged.begin(), merged.end());
        merged.erase(std::unique(merged.begin(), merged.end(),
            [](const KV& a, const KV& b){ return a.key == b.key; }), merged.end());
        n_ = merged.size();
        build(merged);
    }
};

} // namespace nli

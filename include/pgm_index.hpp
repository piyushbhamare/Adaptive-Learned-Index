// pgm_index.hpp
// Wrapper around the official PGM-Index (Apache 2.0 License)
// https://github.com/gvinciguerra/PGM-index
#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include <memory>
#include "pgm/pgm_index.hpp"
#include "common.hpp"

namespace nli {

static constexpr size_t PGM_EPSILON = 64;

class PGMIndex {
public:
    using PGMType = pgm::PGMIndex<Key, PGM_EPSILON>;
    PGMIndex() = default;

    void build(const std::vector<KV>& data) {
        if (data.empty()) return;
        data_.clear(); data_.reserve(data.size());
        vals_.clear(); vals_.reserve(data.size());
        for (const auto& kv : data) { data_.push_back(kv.key); vals_.push_back(kv.value); }
        rebuild();
        size_ = data.size();
    }

    bool lookup(Key key, Value& out) const {
        if (!inserts_.empty()) flush();
        if (!pgm_) return false;
        auto r  = pgm_->search(key);
        auto lo = data_.begin() + r.lo;
        auto hi = data_.begin() + std::min<size_t>(r.hi, data_.size());
        auto it = std::lower_bound(lo, hi, key);
        if (it == data_.end() || *it != key) return false;
        out = vals_[static_cast<size_t>(it - data_.begin())];
        return true;
    }

    void insert(Key key, Value val) {
        inserts_.push_back({key, val});
        ++size_;
        if (inserts_.size() > std::max<size_t>(1000UL, data_.size() / 20))
            flush();
    }

    size_t size()         const { return size_; }
    size_t memory_bytes() const {
        if (!pgm_) return (data_.size()+vals_.size())*8;
        // [AUDIT-FIX MEM-2] Use .size() not .capacity(): capacity() counts over-allocated
        // vector headroom and inflates PGM's reported memory, biasing comparisons.
        return pgm_->size_in_bytes() + data_.size()*sizeof(Key) + vals_.size()*sizeof(Value);
    }

private:
    mutable std::vector<Key>   data_;
    mutable std::vector<Value> vals_;
    struct PI { Key key; Value val; };
    mutable std::vector<PI> inserts_;
    mutable std::unique_ptr<PGMType> pgm_;
    size_t size_ = 0;

    void rebuild() const {
        pgm_ = std::make_unique<PGMType>(data_.begin(), data_.end());
    }

    void flush() const {
        std::sort(inserts_.begin(), inserts_.end(),
                  [](const PI& a, const PI& b){ return a.key < b.key; });
        std::vector<Key>   nk; nk.reserve(data_.size() + inserts_.size());
        std::vector<Value> nv; nv.reserve(data_.size() + inserts_.size());
        size_t i = 0, j = 0;
        while (i < data_.size() && j < inserts_.size()) {
            if (data_[i] <= inserts_[j].key) { nk.push_back(data_[i]); nv.push_back(vals_[i]); ++i; }
            else { nk.push_back(inserts_[j].key); nv.push_back(inserts_[j].val); ++j; }
        }
        while (i < data_.size())    { nk.push_back(data_[i]); nv.push_back(vals_[i]); ++i; }
        while (j < inserts_.size()) { nk.push_back(inserts_[j].key); nv.push_back(inserts_[j].val); ++j; }
        data_.swap(nk); vals_.swap(nv);
        inserts_.clear();
        rebuild();
    }
};

} // namespace nli

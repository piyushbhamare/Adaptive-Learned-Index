// alex_index.hpp
// Wrapper around the official Microsoft ALEX implementation (MIT License)
// https://github.com/microsoft/ALEX
// Avoids insert ambiguity by using pair overload.
#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include "alex/alex.h"
#include "common.hpp"

namespace nli {

class ALEXIndex {
public:
    using AlexT = alex::Alex<Key, Value>;
    ALEXIndex() = default;

    void build(const std::vector<KV>& data) {
        if (data.empty()) return;
        std::vector<std::pair<Key,Value>> pairs;
        pairs.reserve(data.size());
        for (const auto& kv : data) pairs.emplace_back(kv.key, kv.value);
        for (size_t i = 1; i < pairs.size(); ++i)
            if (pairs[i].first < pairs[i-1].first)
                throw std::runtime_error("ALEXIndex::build: data must be sorted");
        idx_.bulk_load(pairs.data(), static_cast<int>(pairs.size()));
        size_ = data.size();
    }

    bool lookup(Key key, Value& out) const {
        auto it = idx_.find(key);
        if (it == idx_.end()) return false;
        out = it.payload();
        return true;
    }

    void insert(Key key, Value val) {
        idx_.insert(std::pair<Key,Value>{key, val});
        ++size_;
    }

    size_t size()         const { return size_; }
    size_t memory_bytes() const { return idx_.model_size() + idx_.data_size(); }

private:
    mutable AlexT idx_;
    size_t size_ = 0;
};

} // namespace nli

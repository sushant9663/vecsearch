#pragma once
// Phase 2: attribute filters and the three strategies being compared.
//   pre-filter : find matching ids first (sorted array / inverted list), exact scan over them.
//   post-filter: ordinary HNSW search with a larger ef, then drop non-matching results.
//   in-graph   : HNSW search whose result set only admits matching nodes.
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <queue>
#include <random>
#include <utility>
#include <vector>

#include "vs/hnsw.hpp"

namespace vs {

struct Attributes {
  std::vector<int32_t> label;  // 0: 1% of points, 1: 10%, 2: 50%, 3: remaining 39%
  std::vector<float> value;    // in [0,1); range [0,s) selects ~s of points

  static int label_for(double u) { return u < 0.01 ? 0 : u < 0.11 ? 1 : u < 0.61 ? 2 : 3; }

  // correlated=false: attributes independent of vector position (easy case).
  // correlated=true : value follows the rank of coordinate 0 and label follows coordinate 1,
  //                   so matching points form a spatial region (hard case for filtering).
  static Attributes generate(const float* data, size_t n, size_t d, bool correlated, uint64_t seed = 7) {
    Attributes a;
    a.label.resize(n); a.value.resize(n);
    std::vector<double> u1(n), u2(n);
    if (correlated) {
      percentile(data, n, d, 0, u1);
      percentile(data, n, d, d > 1 ? 1 : 0, u2);
    } else {
      std::mt19937_64 rng(seed);
      std::uniform_real_distribution<double> U(0.0, 1.0);
      for (size_t i = 0; i < n; ++i) { u1[i] = U(rng); u2[i] = U(rng); }
    }
    for (size_t i = 0; i < n; ++i) { a.value[i] = static_cast<float>(u1[i]); a.label[i] = label_for(u2[i]); }
    return a;
  }

 private:
  static void percentile(const float* data, size_t n, size_t d, size_t col, std::vector<double>& u) {
    std::vector<uint32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0u);
    std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return data[a * d + col] < data[b * d + col]; });
    for (size_t r = 0; r < n; ++r) u[idx[r]] = (r + 0.5) / n;
  }
};

struct Filter {
  enum Kind { kLabel, kRange } kind = kLabel;
  int32_t label = 0;
  float lo = 0, hi = 1;
};

// Selectivity 0.01/0.1/0.5 -> label 0/1/2, or range [0, s).
inline Filter make_filter(bool range, double sel) {
  Filter f;
  if (range) { f.kind = Filter::kRange; f.lo = 0.f; f.hi = static_cast<float>(sel); }
  else { f.kind = Filter::kLabel; f.label = sel <= 0.015 ? 0 : sel <= 0.15 ? 1 : 2; }
  return f;
}

struct FilterPass {
  const Attributes* a; Filter f;
  bool operator()(uint32_t id) const {
    return f.kind == Filter::kLabel ? a->label[id] == f.label
                                    : (a->value[id] >= f.lo && a->value[id] < f.hi);
  }
};

// Inverted lists per label + ids sorted by value, so matching sets are found without a scan.
class FilterIndex {
 public:
  explicit FilterIndex(const Attributes& a) {
    size_t n = a.value.size();
    order_.resize(n);
    std::iota(order_.begin(), order_.end(), 0u);
    std::sort(order_.begin(), order_.end(), [&](uint32_t x, uint32_t y) { return a.value[x] < a.value[y]; });
    sorted_.resize(n);
    for (size_t i = 0; i < n; ++i) sorted_[i] = a.value[order_[i]];
    int32_t maxl = 0;
    for (int32_t l : a.label) maxl = std::max(maxl, l);
    lists_.resize(maxl + 1);
    for (size_t i = 0; i < n; ++i) lists_[a.label[i]].push_back(static_cast<uint32_t>(i));
  }
  std::pair<const uint32_t*, size_t> candidates(const Filter& f) const {
    if (f.kind == Filter::kLabel) {
      if (f.label < 0 || f.label >= static_cast<int32_t>(lists_.size())) return {nullptr, 0};
      return {lists_[f.label].data(), lists_[f.label].size()};
    }
    size_t lo = std::lower_bound(sorted_.begin(), sorted_.end(), f.lo) - sorted_.begin();
    size_t hi = std::lower_bound(sorted_.begin(), sorted_.end(), f.hi) - sorted_.begin();
    return {order_.data() + lo, hi - lo};
  }
 private:
  std::vector<uint32_t> order_;
  std::vector<float> sorted_;
  std::vector<std::vector<uint32_t>> lists_;
};

class FilteredSearch {
 public:
  FilteredSearch(const Hnsw& g, const float* base, size_t d, const Attributes& a, const FilterIndex& fi)
      : g_(g), base_(base), d_(d), a_(a), fi_(fi) {}

  // Exact top-k over the matching set. Also used to compute ground truth.
  int prefilter(const float* q, const Filter& f, int k, uint32_t* out) const {
    auto [ids, n] = fi_.candidates(f);
    using P = std::pair<float, uint32_t>;
    std::priority_queue<P> heap;
    for (size_t i = 0; i < n; ++i) {
      float dd = l2_sqr(q, base_ + static_cast<size_t>(ids[i]) * d_, d_);
      if (heap.size() < static_cast<size_t>(k)) heap.emplace(dd, ids[i]);
      else if (dd < heap.top().first) { heap.pop(); heap.emplace(dd, ids[i]); }
    }
    int c = static_cast<int>(heap.size());
    for (int i = c - 1; i >= 0; --i) { out[i] = heap.top().second; heap.pop(); }
    return c;
  }

  // Fetch `ef` unfiltered neighbors, keep the first k that match. Breaks when ef * selectivity < k.
  int postfilter(const float* q, const Filter& f, int k, uint32_t* out, int ef) const {
    static thread_local std::vector<uint32_t> tmp;
    tmp.resize(std::max(ef, k));
    int c = g_.search_topn(q, std::max(ef, k), tmp.data(), ef);
    FilterPass pass{&a_, f};
    int o = 0;
    for (int i = 0; i < c && o < k; ++i) if (pass(tmp[i])) out[o++] = tmp[i];
    return o;
  }

  int ingraph(const float* q, const Filter& f, int k, uint32_t* out, int ef, size_t max_visit = SIZE_MAX) const {
    return g_.search_filtered(q, k, out, ef, FilterPass{&a_, f}, max_visit);
  }

 private:
  const Hnsw& g_;
  const float* base_;
  size_t d_;
  const Attributes& a_;
  const FilterIndex& fi_;
};

}  // namespace vs

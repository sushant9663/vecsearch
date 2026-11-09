#pragma once
#include <queue>
#include <utility>

#include "vs/distance.hpp"
#include "vs/index.hpp"

namespace vs {

// Exact search. Serves as (1) the correctness oracle, (2) the QPS floor every
// approximate index must beat, (3) a check that the harness's recall
// computation is right (recall@10 must be 1.0 against SIFT1M ground truth).
class BruteForce : public Index {
 public:
  void build(const float* data, size_t n, size_t d) override { data_ = data; n_ = n; d_ = d; }
  void search(const float* q, int k, uint32_t* out) const override {
    using P = std::pair<float, uint32_t>;
    std::priority_queue<P> heap;  // max-heap of the current best k
    for (size_t i = 0; i < n_; ++i) {
      float dist = l2_sqr(q, data_ + i * d_, d_);
      if (heap.size() < static_cast<size_t>(k)) heap.emplace(dist, static_cast<uint32_t>(i));
      else if (dist < heap.top().first) { heap.pop(); heap.emplace(dist, static_cast<uint32_t>(i)); }
    }
    for (int i = static_cast<int>(heap.size()) - 1; i >= 0; --i) { out[i] = heap.top().second; heap.pop(); }
  }
  const char* name() const override { return "bruteforce"; }

 private:
  const float* data_ = nullptr;
  size_t n_ = 0, d_ = 0;
};

}  // namespace vs

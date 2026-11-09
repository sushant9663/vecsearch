#pragma once
#include <cstddef>
#include <cstdint>

namespace vs {

// Common interface so brute-force, HNSW, and FAISS wrappers all plug into the
// same benchmark harness. `search_param` is the single recall/speed knob
// (ef_search for HNSW), ignored by brute force.
class Index {
 public:
  virtual ~Index() = default;
  virtual void build(const float* data, size_t n, size_t d) = 0;
  virtual void set_search_param(int p) { (void)p; }
  // Writes the k nearest ids (ascending by distance) into out_ids.
  virtual void search(const float* query, int k, uint32_t* out_ids) const = 0;
  virtual const char* name() const = 0;
};

}  // namespace vs

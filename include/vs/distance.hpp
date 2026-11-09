#pragma once
#include <cstddef>

namespace vs {

// Squared L2. Written so GCC/Clang auto-vectorize it under -O3 -march=native.
// Replace with explicit AVX2/AVX-512 intrinsics only if the benchmark curve
// shows distance computation is the bottleneck.
inline float l2_sqr(const float* a, const float* b, size_t d) {
  float s = 0.f;
  for (size_t i = 0; i < d; ++i) {
    float t = a[i] - b[i];
    s += t * t;
  }
  return s;
}

}  // namespace vs

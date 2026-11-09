#pragma once
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace vs {

// Row-major matrix of T with `n` rows of `d` columns.
template <class T> struct Matrix {
  size_t n = 0, d = 0;
  std::vector<T> data;
  const T* row(size_t i) const { return data.data() + i * d; }
};

// Reads .fvecs / .ivecs: each vector is [int32 dim][dim * 4 bytes].
template <class T> Matrix<T> read_vecs(const std::string& path) {
  static_assert(sizeof(T) == 4, "fvecs/ivecs store 4-byte elements");
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  Matrix<T> m;
  int32_t dim = 0;
  while (std::fread(&dim, sizeof(dim), 1, f) == 1) {
    if (dim <= 0) throw std::runtime_error("bad dim in " + path);
    if (m.d == 0) m.d = static_cast<size_t>(dim);
    else if (m.d != static_cast<size_t>(dim)) throw std::runtime_error("ragged file " + path);
    size_t off = m.data.size();
    m.data.resize(off + m.d);
    if (std::fread(m.data.data() + off, sizeof(T), m.d, f) != m.d)
      throw std::runtime_error("truncated file " + path);
    ++m.n;
  }
  std::fclose(f);
  return m;
}

}  // namespace vs

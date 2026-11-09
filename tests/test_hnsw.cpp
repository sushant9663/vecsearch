// Correctness tests: recall sanity, filter correctness, parallel build, concurrent insert+query.
#include <atomic>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>
#include "vs/bruteforce.hpp"
#include "vs/hnsw.hpp"

using namespace vs;
static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { std::printf("FAIL: %s\n", msg); ++failures; } } while (0)

static std::vector<float> rand_data(size_t n, size_t d, uint64_t seed) {
  std::mt19937_64 g(seed); std::normal_distribution<float> N;
  std::vector<float> v(n * d); for (auto& x : v) x = N(g); return v;
}
static double recall(const Index& idx, const BruteForce& bf, const std::vector<float>& qs, size_t d, int k) {
  size_t nq = qs.size() / d, hit = 0;
  std::vector<uint32_t> a(k), b(k);
  for (size_t q = 0; q < nq; ++q) {
    idx.search(&qs[q * d], k, a.data()); bf.search(&qs[q * d], k, b.data());
    for (int i = 0; i < k; ++i) for (int j = 0; j < k; ++j) if (a[i] == b[j]) { ++hit; break; }
  }
  return double(hit) / (nq * k);
}

int main() {
  const size_t n = 6000, d = 32, nq = 100; const int k = 10;
  auto base = rand_data(n, d, 1), qs = rand_data(nq, d, 2);
  BruteForce bf; bf.build(base.data(), n, d);

  Hnsw h1; h1.build(base.data(), n, d); h1.set_search_param(100);
  double r1 = recall(h1, bf, qs, d, k);
  std::printf("single-thread build recall@10 = %.3f\n", r1);
  CHECK(r1 > 0.95, "single-thread recall too low");

  Hnsw h4; h4.set_build_threads(4); h4.build(base.data(), n, d); h4.set_search_param(100);
  double r4 = recall(h4, bf, qs, d, k);
  std::printf("4-thread build recall@10 = %.3f\n", r4);
  CHECK(r4 > 0.93, "parallel-build recall too low");

  // filter: every returned id must satisfy the predicate
  auto even = [](uint32_t id) { return id % 2 == 0; };
  std::vector<uint32_t> out(k); bool ok = true;
  for (size_t q = 0; q < nq; ++q) {
    int c = h1.search_filtered(&qs[q * d], k, out.data(), 100, even);
    for (int i = 0; i < c; ++i) ok &= (out[i] % 2 == 0);
    ok &= (c == k);
  }
  CHECK(ok, "in-graph filter returned a non-matching id or fewer than k");

  // concurrent: half pre-built, other half inserted while queries run
  Hnsw hc; hc.init(n, d, true); hc.set_search_param(100);
  for (size_t i = 0; i < n / 2; ++i) hc.insert(&base[i * d], uint32_t(i));
  std::atomic<size_t> next{n / 2}; std::atomic<bool> stop{false}; std::atomic<size_t> served{0};
  std::vector<std::thread> th;
  for (int t = 0; t < 3; ++t) th.emplace_back([&] {
    size_t i; while ((i = next.fetch_add(1)) < n) hc.insert(&base[i * d], uint32_t(i)); });
  for (int t = 0; t < 2; ++t) th.emplace_back([&, t] {
    std::vector<uint32_t> o(k); size_t q = t;
    while (!stop) { hc.search(&qs[(q++ % nq) * d], k, o.data()); ++served; } });
  for (int t = 0; t < 3; ++t) th[t].join();
  stop = true; for (size_t t = 3; t < th.size(); ++t) th[t].join();
  double rc = recall(hc, bf, qs, d, k);
  std::printf("concurrent build recall@10 = %.3f (%zu queries served during inserts), size=%zu\n", rc, served.load(), hc.size());
  CHECK(hc.size() == n, "not all inserts completed");
  CHECK(rc > 0.93, "recall after concurrent inserts too low");

  std::printf(failures ? "TESTS FAILED\n" : "ALL TESTS PASSED\n");
  return failures ? 1 : 0;
}

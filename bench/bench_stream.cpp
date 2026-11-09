// Phase 3 benchmark: query latency and insert throughput under mixed load.
//   bench_stream --base B --query Q --gt G [--initial-frac 0.5] [--insert-threads 2]
//     [--query-threads 2] [--insert-rate 0] [--ef 100] [--baseline-secs 10] [--label x] [--out f.csv]
// Rows written:
//   static_full : index fully built first, queries only (the "Phase 1 under the same threads" baseline)
//   mixed       : first initial-frac built, rest inserted while query threads run
// recall_final is measured single-threaded after all inserts, against the full-base ground truth.
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "vs/hnsw.hpp"
#include "vs/io.hpp"

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

struct Result { double ins_per_s = 0, qps = 0, p50 = 0, p99 = 0, recall = 0, build_s = 0; size_t nq_done = 0; };

static double final_recall(const vs::Hnsw& g, const vs::Matrix<float>& qs, const vs::Matrix<int32_t>& gt, int k) {
  std::vector<uint32_t> o(k); size_t hit = 0;
  for (size_t q = 0; q < qs.n; ++q) {
    g.search(qs.row(q), k, o.data());
    for (int i = 0; i < k; ++i) for (int j = 0; j < k; ++j) if (o[i] == (uint32_t)gt.row(q)[j]) { ++hit; break; }
  }
  return double(hit) / (qs.n * k);
}

int main(int argc, char** argv) {
  std::string base_p, query_p, gt_p, out_p, label = "run";
  double initial_frac = 0.5, insert_rate = 0, baseline_secs = 10;
  int ins_t = 2, q_t = 2, ef = 100, k = 10, M = 16, efc = 200, build_t = 1;
  for (int i = 1; i < argc; ++i) {
    auto is = [&](const char* n) { return std::strcmp(argv[i], n) == 0 && i + 1 < argc; };
    if (is("--base")) base_p = argv[++i]; else if (is("--query")) query_p = argv[++i];
    else if (is("--gt")) gt_p = argv[++i]; else if (is("--out")) out_p = argv[++i];
    else if (is("--label")) label = argv[++i]; else if (is("--initial-frac")) initial_frac = std::atof(argv[++i]);
    else if (is("--insert-rate")) insert_rate = std::atof(argv[++i]); else if (is("--baseline-secs")) baseline_secs = std::atof(argv[++i]);
    else if (is("--insert-threads")) ins_t = std::atoi(argv[++i]); else if (is("--query-threads")) q_t = std::atoi(argv[++i]);
    else if (is("--ef")) ef = std::atoi(argv[++i]); else if (is("--k")) k = std::atoi(argv[++i]);
    else if (is("--M")) M = std::atoi(argv[++i]); else if (is("--efc")) efc = std::atoi(argv[++i]);
    else if (is("--build-threads")) build_t = std::atoi(argv[++i]);
    else { std::cerr << "bad arg " << argv[i] << "\n"; return 2; }
  }
  if (base_p.empty() || query_p.empty() || gt_p.empty()) { std::cerr << "need --base --query --gt\n"; return 2; }
  auto base = vs::read_vecs<float>(base_p);
  auto qs = vs::read_vecs<float>(query_p);
  auto gt = vs::read_vecs<int32_t>(gt_p);
  const size_t n = base.n, d = base.d, m0 = static_cast<size_t>(n * initial_frac);

  // Runs query threads until `stop` (set externally or by timeout); returns per-query latencies.
  auto run_queries = [&](const vs::Hnsw& g, std::atomic<bool>& stop, double max_secs,
                         std::vector<std::vector<float>>& lats, std::vector<std::thread>& th) {
    lats.assign(q_t, {});
    auto t0 = Clock::now();
    for (int t = 0; t < q_t; ++t) th.emplace_back([&, t, t0, max_secs] {
      std::vector<uint32_t> o(k); lats[t].reserve(1 << 16);
      for (size_t i = t * 977;; ++i) {
        if (stop.load(std::memory_order_relaxed)) break;
        if (max_secs > 0 && secs(t0, Clock::now()) > max_secs) break;
        auto a = Clock::now();
        g.search(qs.row(i % qs.n), k, o.data());
        lats[t].push_back(static_cast<float>(secs(a, Clock::now()) * 1e6));
      }
    });
  };
  auto summarize = [&](std::vector<std::vector<float>>& lats, double elapsed, Result& r) {
    std::vector<float> all; for (auto& l : lats) all.insert(all.end(), l.begin(), l.end());
    r.nq_done = all.size(); r.qps = all.size() / elapsed;
    if (all.empty()) return;
    std::sort(all.begin(), all.end());
    r.p50 = all[all.size() / 2]; r.p99 = all[std::min(all.size() - 1, static_cast<size_t>(all.size() * 0.99))];
  };

  std::ostream* out = &std::cout; std::ofstream fo;
  if (!out_p.empty()) { fo.open(out_p, std::ios::app); out = &fo; }
  if (out_p.empty() || fo.tellp() == 0)
    *out << "label,mode,insert_threads,query_threads,insert_rate,ef,inserted,insert_per_s,queries,qps,p50_us,p99_us,recall_final,build_s\n";
  auto emit = [&](const char* mode, int it, size_t ins, const Result& r) {
    *out << label << "," << mode << "," << it << "," << q_t << "," << insert_rate << "," << ef << "," << ins << ","
         << r.ins_per_s << "," << r.nq_done << "," << r.qps << "," << r.p50 << "," << r.p99 << "," << r.recall << "," << r.build_s << "\n";
    out->flush();
  };

  // ---- static_full baseline ----
  {
    vs::Hnsw g(vs::HnswParams{M, efc, 100}); g.set_build_threads(build_t); g.set_force_threadsafe(true);
    auto t0 = Clock::now(); g.build(base.data.data(), n, d);
    Result r; r.build_s = secs(t0, Clock::now()); g.set_search_param(ef);
    std::atomic<bool> stop{false};
    auto t1 = Clock::now();
    std::vector<std::vector<float>> lats; std::vector<std::thread> th;
    run_queries(g, stop, baseline_secs, lats, th);
    for (auto& t : th) t.join();
    summarize(lats, secs(t1, Clock::now()), r);
    r.recall = final_recall(g, qs, gt, k);
    emit("static_full", 0, n, r);
  }
  // ---- mixed ----
  {
    vs::Hnsw g(vs::HnswParams{M, efc, 100}); g.init(n, d, true); g.set_search_param(ef);
    auto t0 = Clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256) num_threads(build_t) if (build_t > 1)
#endif
    for (long i = 0; i < static_cast<long>(m0); ++i) g.insert(base.row(i), static_cast<uint32_t>(i));
    Result r; r.build_s = secs(t0, Clock::now());
    std::atomic<size_t> next{m0}; std::atomic<bool> stop{false}; std::atomic<int> alive{ins_t};
    auto t1 = Clock::now();
    std::vector<std::vector<float>> lats; std::vector<std::thread> qth;
    run_queries(g, stop, 0, lats, qth);
    std::vector<std::thread> ith;
    for (int t = 0; t < ins_t; ++t) ith.emplace_back([&] {
      auto next_t = Clock::now();
      auto gap = std::chrono::duration<double>(insert_rate > 0 ? ins_t / insert_rate : 0.0);
      size_t i;
      while ((i = next.fetch_add(1)) < n) {
        if (insert_rate > 0) { next_t += std::chrono::duration_cast<Clock::duration>(gap); std::this_thread::sleep_until(next_t); }
        g.insert(base.row(i), static_cast<uint32_t>(i));
      }
    });
    for (auto& t : ith) t.join();
    double elapsed = secs(t1, Clock::now());
    stop = true;
    for (auto& t : qth) t.join();
    summarize(lats, elapsed, r);
    r.ins_per_s = (n - m0) / elapsed;
    r.recall = final_recall(g, qs, gt, k);
    emit("mixed", ins_t, n - m0, r);
  }
  return 0;
}

// Benchmark harness: recall@k vs QPS, build time, memory, latency percentiles.
// Usage:
//   bench --base sift_base.fvecs --query sift_query.fvecs --gt sift_groundtruth.ivecs \
//         --index bruteforce [--k 10] [--sweep 10,20,40,80] [--repeat 3] [--threads 1]
//         [--max-queries N] [--M 16] [--efc 200] [--build-threads 1] [--threadsafe 0|1] [--out results.csv]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "vs/bruteforce.hpp"
#include "vs/hnsw.hpp"
#include "vs/index.hpp"
#include "vs/io.hpp"

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// Resident set size in MB (Linux). Index memory ~= RSS after build - RSS before build.
static double rss_mb() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("VmRSS:", 0) == 0) return std::stod(line.substr(6)) / 1024.0;
  return -1.0;
}

struct IndexCfg { int M = 16, efc = 200, build_threads = 1; bool threadsafe = false; };

static std::unique_ptr<vs::Index> make_index(const std::string& name, const IndexCfg& c) {
  if (name == "bruteforce") return std::make_unique<vs::BruteForce>();
  if (name == "hnsw") {
    auto h = std::make_unique<vs::Hnsw>(vs::HnswParams{c.M, c.efc, 100});
    h->set_build_threads(c.build_threads);
    h->set_force_threadsafe(c.threadsafe);
    h->set_name(std::string("hnsw_M") + std::to_string(c.M) + (c.threadsafe ? "_locked" : ""));
    return h;
  }
  std::cerr << "unknown index: " << name << "\n";
  std::exit(2);
}

static std::vector<int> parse_list(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stoi(tok));
  return v;
}

int main(int argc, char** argv) {
  std::string base_p, query_p, gt_p, index_name = "bruteforce", out_p, sweep_s = "0";
  int k = 10, repeat = 3, threads = 1;
  long max_queries = -1;
  IndexCfg cfg;
  for (int i = 1; i < argc; ++i) {
    auto arg = [&](const char* n) { return std::strcmp(argv[i], n) == 0 && i + 1 < argc; };
    if (arg("--base")) base_p = argv[++i];
    else if (arg("--query")) query_p = argv[++i];
    else if (arg("--gt")) gt_p = argv[++i];
    else if (arg("--index")) index_name = argv[++i];
    else if (arg("--k")) k = std::atoi(argv[++i]);
    else if (arg("--sweep")) sweep_s = argv[++i];
    else if (arg("--repeat")) repeat = std::atoi(argv[++i]);
    else if (arg("--threads")) threads = std::atoi(argv[++i]);
    else if (arg("--max-queries")) max_queries = std::atol(argv[++i]);
    else if (arg("--M")) cfg.M = std::atoi(argv[++i]);
    else if (arg("--efc")) cfg.efc = std::atoi(argv[++i]);
    else if (arg("--build-threads")) cfg.build_threads = std::atoi(argv[++i]);
    else if (arg("--threadsafe")) cfg.threadsafe = std::atoi(argv[++i]) != 0;
    else if (arg("--out")) out_p = argv[++i];
    else { std::cerr << "bad arg: " << argv[i] << "\n"; return 2; }
  }
  if (base_p.empty() || query_p.empty() || gt_p.empty()) {
    std::cerr << "need --base --query --gt\n"; return 2;
  }
#ifdef _OPENMP
  omp_set_num_threads(threads);
#endif

  auto base = vs::read_vecs<float>(base_p);
  auto queries = vs::read_vecs<float>(query_p);
  auto gt = vs::read_vecs<int32_t>(gt_p);
  if (base.d != queries.d) { std::cerr << "dim mismatch\n"; return 2; }
  if (gt.n != queries.n || static_cast<int>(gt.d) < k) { std::cerr << "ground truth shape mismatch\n"; return 2; }
  size_t nq = queries.n;
  if (max_queries > 0 && static_cast<size_t>(max_queries) < nq) nq = max_queries;
  std::cerr << "base " << base.n << "x" << base.d << ", queries " << nq << ", k=" << k
            << ", threads=" << threads << "\n";

  auto index = make_index(index_name, cfg);
  double rss0 = rss_mb();
  auto t0 = Clock::now();
  index->build(base.data.data(), base.n, base.d);
  double build_s = secs(t0, Clock::now());
  double index_mb = rss_mb() - rss0;  // ~0 for brute force: it only references the base array
  std::cerr << "build " << build_s << " s, extra RSS " << index_mb << " MB\n";

  std::ostream* out = &std::cout;
  std::ofstream fout;
  if (!out_p.empty()) { fout.open(out_p); out = &fout; }
  *out << "index,param,k,threads,recall,qps,p50_us,p99_us,build_s,index_mb\n";

  std::vector<uint32_t> res(nq * k);
  std::vector<double> lat(nq);
  for (int param : parse_list(sweep_s)) {
    index->set_search_param(param);
    double best_qps = 0;
    std::vector<double> best_lat;
    for (int r = 0; r < repeat; ++r) {  // keep the fastest run to reduce noise
      auto ts = Clock::now();
#pragma omp parallel for schedule(static) if (threads > 1)
      for (long q = 0; q < static_cast<long>(nq); ++q) {
        auto a = Clock::now();
        index->search(queries.row(q), k, res.data() + q * k);
        lat[q] = secs(a, Clock::now()) * 1e6;
      }
      double qps = nq / secs(ts, Clock::now());
      if (qps > best_qps) { best_qps = qps; best_lat = lat; }
    }
    // recall@k = |returned ∩ true top-k| / k, averaged over queries
    size_t hit = 0;
    for (size_t q = 0; q < nq; ++q) {
      const int32_t* g = gt.row(q);
      for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j)
          if (res[q * k + j] == static_cast<uint32_t>(g[i])) { ++hit; break; }
    }
    double recall = static_cast<double>(hit) / (nq * k);
    std::sort(best_lat.begin(), best_lat.end());
    double p50 = best_lat[nq / 2], p99 = best_lat[std::min(nq - 1, static_cast<size_t>(nq * 0.99))];
    *out << index->name() << "," << param << "," << k << "," << threads << "," << recall << ","
         << best_qps << "," << p50 << "," << p99 << "," << build_s << "," << index_mb << "\n";
    out->flush();
  }
  return 0;
}

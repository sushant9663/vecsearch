// Phase 2 benchmark: pre-filter vs post-filter vs in-graph at 1% / 10% / 50% selectivity.
//   bench_filtered --base B --query Q [--attr-mode random|correlated] [--filters label,range]
//     [--selectivities 0.01,0.1,0.5] [--sweep 32,64,...] [--max-queries 1000] [--repeat 1] ...
// Ground truth for a filtered query is the exact top-k among matching points.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "vs/filtered.hpp"
#include "vs/io.hpp"

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

static std::vector<double> parse_d(const std::string& s) {
  std::vector<double> v; std::stringstream ss(s); std::string t;
  while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::stod(t));
  return v;
}

struct Stats { double qps, p50, p99; };
template <class Fn>
static Stats measure(size_t nq, int repeat, Fn&& fn) {  // fn(q) runs one query
  Stats best{0, 0, 0};
  std::vector<double> lat(nq);
  for (int r = 0; r < repeat; ++r) {
    auto ts = Clock::now();
    for (size_t q = 0; q < nq; ++q) { auto a = Clock::now(); fn(q); lat[q] = secs(a, Clock::now()) * 1e6; }
    double qps = nq / secs(ts, Clock::now());
    if (qps > best.qps) {
      std::vector<double> s = lat; std::sort(s.begin(), s.end());
      best = {qps, s[nq / 2], s[std::min(nq - 1, static_cast<size_t>(nq * 0.99))]};
    }
  }
  return best;
}

int main(int argc, char** argv) {
  std::string base_p, query_p, out_p, mode = "random", filters_s = "label,range";
  std::string sels_s = "0.01,0.1,0.5", sweep_s = "32,64,128,256,512,1024";
  int k = 10, M = 16, efc = 200, repeat = 1, threads = 1;
  long max_q = 1000, max_visit = 0;
  for (int i = 1; i < argc; ++i) {
    auto is = [&](const char* n) { return std::strcmp(argv[i], n) == 0 && i + 1 < argc; };
    if (is("--base")) base_p = argv[++i]; else if (is("--query")) query_p = argv[++i];
    else if (is("--out")) out_p = argv[++i]; else if (is("--attr-mode")) mode = argv[++i];
    else if (is("--filters")) filters_s = argv[++i]; else if (is("--selectivities")) sels_s = argv[++i];
    else if (is("--sweep")) sweep_s = argv[++i]; else if (is("--k")) k = std::atoi(argv[++i]);
    else if (is("--M")) M = std::atoi(argv[++i]); else if (is("--efc")) efc = std::atoi(argv[++i]);
    else if (is("--repeat")) repeat = std::atoi(argv[++i]); else if (is("--threads")) threads = std::atoi(argv[++i]);
    else if (is("--max-queries")) max_q = std::atol(argv[++i]); else if (is("--max-visit")) max_visit = std::atol(argv[++i]);
    else { std::cerr << "bad arg " << argv[i] << "\n"; return 2; }
  }
  if (base_p.empty() || query_p.empty()) { std::cerr << "need --base --query\n"; return 2; }
#ifdef _OPENMP
  omp_set_num_threads(threads);
#endif
  auto base = vs::read_vecs<float>(base_p);
  auto queries = vs::read_vecs<float>(query_p);
  size_t nq = std::min<size_t>(queries.n, max_q > 0 ? max_q : queries.n), d = base.d;

  vs::Hnsw g(vs::HnswParams{M, efc, 100});
  g.set_build_threads(threads);
  auto t0 = Clock::now(); g.build(base.data.data(), base.n, d);
  std::cerr << "built HNSW in " << secs(t0, Clock::now()) << " s\n";
  auto attrs = vs::Attributes::generate(base.data.data(), base.n, d, mode == "correlated");
  vs::FilterIndex fidx(attrs);
  vs::FilteredSearch fs(g, base.data.data(), d, attrs, fidx);

  std::ostream* out = &std::cout; std::ofstream fo;
  if (!out_p.empty()) { fo.open(out_p); out = &fo; }
  *out << "strategy,filter,selectivity,attr_mode,ef,k,recall,qps,p50_us,p99_us,avg_returned\n";
  const uint32_t NONE = std::numeric_limits<uint32_t>::max();

  std::vector<std::string> ftypes; { std::stringstream ss(filters_s); std::string t; while (std::getline(ss, t, ',')) ftypes.push_back(t); }
  for (const auto& ft : ftypes) for (double sel : parse_d(sels_s)) {
    vs::Filter f = vs::make_filter(ft == "range", sel);
    // ground truth (exact, over matching set)
    std::vector<uint32_t> gt(nq * k, NONE); std::vector<int> gtc(nq, 0);
#pragma omp parallel for schedule(dynamic, 8) if (threads > 1)
    for (long q = 0; q < static_cast<long>(nq); ++q) gtc[q] = fs.prefilter(queries.row(q), f, k, gt.data() + q * k);
    std::vector<uint32_t> res(nq * k);
    auto report = [&](const char* name, int ef, Stats s) {
      size_t hit = 0, denom = 0, ret = 0;
      for (size_t q = 0; q < nq; ++q) {
        denom += gtc[q];
        for (int i = 0; i < k; ++i) {
          uint32_t r = res[q * k + i]; if (r == NONE) continue; ++ret;
          for (int j = 0; j < gtc[q]; ++j) if (gt[q * k + j] == r) { ++hit; break; }
        }
      }
      *out << name << "," << ft << "," << sel << "," << mode << "," << ef << "," << k << ","
           << double(hit) / std::max<size_t>(denom, 1) << "," << s.qps << "," << s.p50 << "," << s.p99
           << "," << double(ret) / nq << "\n";
      out->flush();
    };
    auto run = [&](auto&& strategy) {
      return measure(nq, repeat, [&](size_t q) {
        uint32_t* o = res.data() + q * k; std::fill(o, o + k, NONE); strategy(queries.row(q), o); });
    };
    report("prefilter", 0, run([&](const float* q, uint32_t* o) { fs.prefilter(q, f, k, o); }));
    for (double efd : parse_d(sweep_s)) {
      int ef = static_cast<int>(efd);
      report("postfilter", ef, run([&](const float* q, uint32_t* o) { fs.postfilter(q, f, k, o, ef); }));
      size_t mv = max_visit > 0 ? static_cast<size_t>(max_visit) : SIZE_MAX;
      report("ingraph", ef, run([&](const float* q, uint32_t* o) { fs.ingraph(q, f, k, o, ef, mv); }));
    }
    std::cerr << "done " << ft << " sel=" << sel << "\n";
  }
  return 0;
}

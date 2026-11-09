#pragma once
// HNSW (Malkov & Yashunin 2016), written from scratch.
//
// Design points worth knowing for the interview:
//  * Level-0 neighbor lists live in one flat array (count + up to 2M ids per node) for cache
//    locality; upper levels are small per-node arrays allocated only for the ~1/M of nodes that
//    reach them.
//  * Capacity is fixed up front so inserts never reallocate; this is what makes concurrent
//    inserts + queries possible without a global lock.
//  * Concurrency (Phase 3): one spinlock per node guards that node's neighbor lists. A thread
//    never holds two node locks at once, so there is no lock-ordering/deadlock problem. Readers
//    copy a neighbor list under the node's lock, then release it before computing distances.
//    The entry point and its level are packed into one atomic word so a reader can never see a
//    new max level with an old entry point. When locking is off (default for static builds),
//    all lock code is skipped, so Phase 1 numbers are not penalized.
//  * Filtered search (Phase 2) is a template parameter of the level-0 search: results only
//    include nodes that pass the predicate, while navigation still uses every node.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "vs/distance.hpp"
#include "vs/index.hpp"

namespace vs {

struct HnswParams {
  int M = 16;
  int ef_construction = 200;
  uint64_t seed = 100;
};

struct NoFilter {
  bool operator()(uint32_t) const { return true; }
};

class SpinLock {
 public:
  void lock() { while (f_.test_and_set(std::memory_order_acquire)) { } }
  void unlock() { f_.clear(std::memory_order_release); }
 private:
  std::atomic_flag f_ = ATOMIC_FLAG_INIT;
};

class Hnsw : public Index {
 public:
  using Pair = std::pair<float, uint32_t>;  // (distance, id)
  static constexpr int kMaxLinks = 128;     // upper bound for 2*M

  explicit Hnsw(HnswParams p = {}) : p_(p) {
    if (2 * p.M > kMaxLinks || p.M < 2) throw std::invalid_argument("M out of range");
    M_ = p.M;
    maxM0_ = 2 * p.M;
    mult_ = 1.0 / std::log(static_cast<double>(p.M));
  }

  // ---- configuration -------------------------------------------------------------------
  void set_build_threads(int t) { build_threads_ = t; }
  void set_force_threadsafe(bool b) { force_threadsafe_ = b; }
  void set_search_param(int ef) override { ef_search_ = ef; }
  const char* name() const override { return name_.c_str(); }
  void set_name(std::string n) { name_ = std::move(n); }
  size_t size() const { return count_.load(std::memory_order_acquire); }
  size_t capacity() const { return cap_; }
  bool thread_safe() const { return threadsafe_; }

  // Allocates storage for `capacity` vectors. Must be called before insert().
  void init(size_t capacity, size_t d, bool thread_safe) {
    cap_ = capacity; d_ = d; threadsafe_ = thread_safe;
    stride0_ = 1 + maxM0_; strideU_ = 1 + M_;
    data_.assign(cap_ * d_, 0.f);
    l0_.assign(cap_ * stride0_, 0u);
    levels_.assign(cap_, 0);
    upper_.clear(); upper_.resize(cap_);
    locks_.reset(new SpinLock[cap_]);
    entry_.store(kNone); count_.store(0);
  }

  // Index interface: copies vectors into the index and inserts them (id = row number).
  void build(const float* data, size_t n, size_t d) override {
    init(n, d, force_threadsafe_ || build_threads_ > 1);
    if (n == 0) return;
    insert(data, 0);  // first node establishes the entry point
#ifdef _OPENMP
    if (build_threads_ > 1) {
#pragma omp parallel for schedule(dynamic, 256) num_threads(build_threads_)
      for (long i = 1; i < static_cast<long>(n); ++i) insert(data + static_cast<size_t>(i) * d, static_cast<uint32_t>(i));
      return;
    }
#endif
    for (size_t i = 1; i < n; ++i) insert(data + i * d, static_cast<uint32_t>(i));
  }

  // ---- insertion (thread-safe iff threadsafe_) -------------------------------------------
  // `id` must be unique and < capacity.
  void insert(const float* v, uint32_t id) {
    const int level = random_level(id);
    levels_[id] = static_cast<uint8_t>(level);
    if (level > 0) upper_[id].reset(new uint32_t[static_cast<size_t>(level) * strideU_]());
    std::memcpy(vec(id), v, d_ * sizeof(float));
    const float* q = vec(id);

    std::unique_lock<std::mutex> glock(global_, std::defer_lock);
    if (threadsafe_) {
      uint64_t e = entry_.load();
      if (e == kNone || level > level_of(e)) glock.lock();  // rare: ~1/M of inserts
    }
    uint64_t e = entry_.load();
    if (e == kNone) {  // very first node
      entry_.store(pack(level, id));
      count_.fetch_add(1);
      return;
    }
    const int curmax = level_of(e);
    Pair cur{dist(q, vec(id_of(e))), id_of(e)};
    for (int lev = curmax; lev > level; --lev) cur = greedy(q, cur, lev);

    std::vector<Pair> cands, sel(kMaxLinks);
    for (int lev = std::min(level, curmax); lev >= 0; --lev) {
      auto top = search_layer(q, {cur}, p_.ef_construction, lev, NoFilter{}, SIZE_MAX);
      to_sorted(top, cands);
      int sn = select(cands.data(), static_cast<int>(cands.size()), M_, sel.data());
      {  // publish our own list at this level
        SpinGuard g(locks_[id], threadsafe_);
        uint32_t* l = links(id, lev);
        l[0] = static_cast<uint32_t>(sn);
        for (int i = 0; i < sn; ++i) l[1 + i] = sel[i].second;
      }
      for (int i = 0; i < sn; ++i) add_backlink(sel[i].second, id, lev);
      cur = cands.front();  // closest found: entry for the next level down
    }
    if (level > curmax) entry_.store(pack(level, id));
    count_.fetch_add(1);
  }

  // ---- search ----------------------------------------------------------------------------
  void search(const float* q, int k, uint32_t* out) const override {
    int c = search_impl(q, k, out, ef_search_, NoFilter{}, SIZE_MAX);
    for (int i = c; i < k; ++i) out[i] = std::numeric_limits<uint32_t>::max();
  }
  // Unfiltered, returns up to n results with explicit ef (used by post-filtering).
  int search_topn(const float* q, int n, uint32_t* out, int ef) const {
    return search_impl(q, n, out, ef, NoFilter{}, SIZE_MAX);
  }
  // In-graph filtering: results are restricted to ids where pass(id) is true.
  // max_visit bounds distance computations at level 0 (SIZE_MAX = unbounded).
  template <class F>
  int search_filtered(const float* q, int k, uint32_t* out, int ef, const F& pass,
                      size_t max_visit = SIZE_MAX) const {
    return search_impl(q, k, out, ef, pass, max_visit);
  }

 private:
  static constexpr uint64_t kNone = ~0ull;
  static uint64_t pack(int level, uint32_t id) { return (static_cast<uint64_t>(level) << 32) | id; }
  static int level_of(uint64_t e) { return static_cast<int>(e >> 32); }
  static uint32_t id_of(uint64_t e) { return static_cast<uint32_t>(e & 0xffffffffu); }

  struct SpinGuard {
    SpinLock& l; bool on;
    SpinGuard(SpinLock& x, bool enabled) : l(x), on(enabled) { if (on) l.lock(); }
    ~SpinGuard() { if (on) l.unlock(); }
  };
  struct Visited {
    std::vector<uint32_t> tag; uint32_t cur = 0;
    void reset(size_t n) {
      if (tag.size() < n) { tag.assign(n, 0); cur = 0; }
      if (++cur == 0) { std::fill(tag.begin(), tag.end(), 0u); cur = 1; }
    }
    bool test_and_mark(uint32_t i) { if (tag[i] == cur) return true; tag[i] = cur; return false; }
  };
  static Visited& visited(size_t n) { static thread_local Visited v; v.reset(n); return v; }

  float* vec(uint32_t id) { return data_.data() + static_cast<size_t>(id) * d_; }
  const float* vec(uint32_t id) const { return data_.data() + static_cast<size_t>(id) * d_; }
  float dist(const float* a, const float* b) const { return l2_sqr(a, b, d_); }
  int maxm(int lev) const { return lev == 0 ? maxM0_ : M_; }
  uint32_t* links(uint32_t id, int lev) {
    return lev == 0 ? l0_.data() + static_cast<size_t>(id) * stride0_
                    : upper_[id].get() + static_cast<size_t>(lev - 1) * strideU_;
  }
  const uint32_t* links(uint32_t id, int lev) const {
    return lev == 0 ? l0_.data() + static_cast<size_t>(id) * stride0_
                    : upper_[id].get() + static_cast<size_t>(lev - 1) * strideU_;
  }

  // Copies a node's neighbor list into buf under the node's lock; returns the count.
  int read_links(uint32_t id, int lev, uint32_t* buf) const {
    SpinGuard g(locks_[id], threadsafe_);
    const uint32_t* l = links(id, lev);
    int c = static_cast<int>(l[0]);
    std::memcpy(buf, l + 1, static_cast<size_t>(c) * sizeof(uint32_t));
    return c;
  }

  int random_level(uint32_t id) const {  // deterministic per (seed, id): splitmix64
    uint64_t z = p_.seed + 0x9E3779B97F4A7C15ULL * (static_cast<uint64_t>(id) + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    double u = (static_cast<double>(z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    return std::min(static_cast<int>(-std::log(u) * mult_), 30);
  }

  // Greedy 1-NN walk used on upper levels.
  Pair greedy(const float* q, Pair cur, int lev) const {
    uint32_t buf[kMaxLinks];
    for (bool changed = true; changed;) {
      changed = false;
      int c = read_links(cur.second, lev, buf);
      for (int i = 0; i < c; ++i) {
        float d = dist(q, vec(buf[i]));
        if (d < cur.first) { cur = {d, buf[i]}; changed = true; }
      }
    }
    return cur;
  }

  // Best-first search on one level (Algorithm 2). Returns a max-heap of <= ef results that
  // pass the filter. Navigation visits all nodes regardless of the filter.
  template <class F>
  std::priority_queue<Pair> search_layer(const float* q, const std::vector<Pair>& eps, size_t ef,
                                         int lev, const F& pass, size_t max_visit) const {
    Visited& vis = visited(cap_);
    std::priority_queue<Pair> top;
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> cand;
    for (const Pair& e : eps) {
      vis.test_and_mark(e.second);
      cand.push(e);
      if (pass(e.second)) top.push(e);
    }
    while (top.size() > ef) top.pop();
    uint32_t buf[kMaxLinks];
    size_t visits = 0;
    while (!cand.empty()) {
      Pair c = cand.top();
      if (top.size() >= ef && c.first > top.top().first) break;
      cand.pop();
      int cnt = read_links(c.second, lev, buf);
      for (int i = 0; i < cnt; ++i) __builtin_prefetch(vec(buf[i]));
      for (int i = 0; i < cnt; ++i) {
        uint32_t nb = buf[i];
        if (vis.test_and_mark(nb)) continue;
        float d = dist(q, vec(nb));
        ++visits;
        if (top.size() < ef || d < top.top().first) {
          cand.emplace(d, nb);
          if (pass(nb)) { top.emplace(d, nb); if (top.size() > ef) top.pop(); }
        }
      }
      if (visits >= max_visit) break;
    }
    return top;
  }

  static void to_sorted(std::priority_queue<Pair>& top, std::vector<Pair>& out) {
    out.resize(top.size());
    for (size_t i = out.size(); i-- > 0;) { out[i] = top.top(); top.pop(); }
  }

  // Neighbor-selection heuristic (Algorithm 4): keep a candidate only if it is closer to the
  // base point than to every already-selected neighbor. c must be sorted ascending.
  int select(const Pair* c, int cn, int m, Pair* out) const {
    if (cn <= m) { std::copy(c, c + cn, out); return cn; }
    int on = 0;
    for (int i = 0; i < cn && on < m; ++i) {
      bool good = true;
      for (int j = 0; j < on; ++j)
        if (dist(vec(c[i].second), vec(out[j].second)) < c[i].first) { good = false; break; }
      if (good) out[on++] = c[i];
    }
    return on;
  }

  // Adds id to n's list at `lev`; if full, re-selects n's neighbors with the heuristic.
  void add_backlink(uint32_t n, uint32_t id, int lev) {
    SpinGuard g(locks_[n], threadsafe_);
    uint32_t* l = links(n, lev);
    int cnt = static_cast<int>(l[0]), mmax = maxm(lev);
    if (cnt < mmax) { l[1 + cnt] = id; l[0] = static_cast<uint32_t>(cnt + 1); return; }
    Pair c[kMaxLinks + 1], out[kMaxLinks + 1];
    const float* vn = vec(n);
    c[0] = {dist(vn, vec(id)), id};
    for (int i = 0; i < cnt; ++i) c[1 + i] = {dist(vn, vec(l[1 + i])), l[1 + i]};
    std::sort(c, c + cnt + 1);
    int on = select(c, cnt + 1, mmax, out);
    l[0] = static_cast<uint32_t>(on);
    for (int i = 0; i < on; ++i) l[1 + i] = out[i].second;
  }

  template <class F>
  int search_impl(const float* q, int k, uint32_t* out, int ef, const F& pass, size_t max_visit) const {
    uint64_t e = entry_.load();
    if (e == kNone) return 0;
    Pair cur{dist(q, vec(id_of(e))), id_of(e)};
    for (int lev = level_of(e); lev > 0; --lev) cur = greedy(q, cur, lev);
    auto top = search_layer(q, {cur}, static_cast<size_t>(std::max(ef, k)), 0, pass, max_visit);
    std::vector<Pair> r;
    to_sorted(top, r);
    int c = std::min<int>(k, static_cast<int>(r.size()));
    for (int i = 0; i < c; ++i) out[i] = r[i].second;
    return c;
  }

  HnswParams p_;
  int M_ = 16, maxM0_ = 32, ef_search_ = 64, build_threads_ = 1;
  double mult_ = 1.0;
  bool threadsafe_ = false, force_threadsafe_ = false;
  size_t cap_ = 0, d_ = 0, stride0_ = 0, strideU_ = 0;
  std::string name_ = "hnsw";
  std::vector<float> data_;
  std::vector<uint32_t> l0_;
  std::vector<uint8_t> levels_;
  std::vector<std::unique_ptr<uint32_t[]>> upper_;
  std::unique_ptr<SpinLock[]> locks_;
  std::atomic<uint64_t> entry_{kNone};
  std::atomic<size_t> count_{0};
  std::mutex global_;
};

}  // namespace vs

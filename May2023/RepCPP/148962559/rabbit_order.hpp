
#pragma once

#include <numa.h>

#include <omp.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <parallel/algorithm>

#include <boost/algorithm/cxx11/all_of.hpp>
#include <boost/atomic.hpp>
#include <boost/optional/optional.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/algorithm/count_if.hpp>
#include <boost/range/algorithm/remove_if.hpp>
#include <boost/range/algorithm/sort.hpp>
#include <boost/range/algorithm/unique.hpp>
#include <boost/range/irange.hpp>
#include <boost/range/numeric.hpp>

#define RO_DIE rabbit_order::aux::die_t(__FILE__, __LINE__, __func__)

namespace rabbit_order {
namespace aux {



struct die_t {
die_t(const char *file, uint64_t line, const char *func) {
std::cerr << file << ':' << line << '(' << func << ") [FATAL ERROR] ";
}

~die_t() {
std::cerr << std::endl;
exit(EXIT_FAILURE);
}
};

template<typename T>
die_t operator<<(die_t d, const T &x) {
std::cerr << x;
return d;
}


struct free_functor {
void operator()(void *p) const { std::free(p); }
};

template<typename T> using unique_c_ptr = std::unique_ptr<T, free_functor>;

template<typename T, typename = std::enable_if_t<std::is_array<T>::value> >
unique_c_ptr<T> make_aligned_unique(const size_t n, const size_t align) {
typedef typename std::remove_extent<T>::type elem_t;
const size_t z = sizeof(elem_t) * n;
elem_t *p;
if (posix_memalign(reinterpret_cast<void **>(&p), align, z) != 0 ||
p == nullptr) {
RO_DIE << "posix_memalign(3) failed";
}
return std::unique_ptr<T, free_functor>(p, free_functor());
}


template<typename T>
struct block {
size_t size;
T body[0];
};

template<typename T>
T *alloc_interleaved(const size_t nelem) {
const size_t z = sizeof(block<T>) + nelem * sizeof(T);
block<T> *const b = reinterpret_cast<block<T> *>(numa_alloc_interleaved(z));
if (b == NULL)
RO_DIE << "numa_alloc_interleaved(3) failed";
b->size = z;
return b->body;  
}

template<typename T>
void free_body(T *const body) {
if (body != nullptr) {
block<T> *const b = reinterpret_cast<block<T> *>(
reinterpret_cast<uint8_t *>(body) - offsetof(block<T>, body));
assert(b->size > offsetof(block<T>, body));
numa_free(b, b->size);
}
}

template<typename T>
struct free_body_functor {
void operator()(T *p) const { free_body(p); }
};

template<typename T>
using numa_unique_ptr = std::unique_ptr<
T, free_body_functor<typename std::remove_extent<T>::type> >;

template<typename T, typename = std::enable_if_t<std::is_array<T>::value> >
numa_unique_ptr<T> make_unique_interleaved(const size_t n) {
return numa_unique_ptr<T>(
alloc_interleaved<typename std::remove_extent<T>::type>(n));
}

template<typename T>
union atomix {
boost::atomic<T> a;
T raw;

atomix() : a(T()) {}

atomix(const atomix<T> &x) : a(static_cast<T>(x)) {}

atomix(T x) : a(x) {}

operator T() const { return a.load(boost::memory_order_acquire); }

T fetch_add(T x) { return a.fetch_add(x, boost::memory_order_acq_rel); }

T exchange(T x) { return a.exchange(x, boost::memory_order_acq_rel); }

bool compare_exchange_weak(T &exp, T x) {
return a.compare_exchange_weak(exp, x, boost::memory_order_acq_rel);
}

atomix &operator=(const atomix<T> &x) {
a.store(x, boost::memory_order_release);
return *this;
}

atomix &operator=(T x) {
a.store(x, boost::memory_order_release);
return *this;
}

const T *operator->() const {
assert(sizeof(*this) == sizeof(T) && a.is_lock_free());

return &raw;
}

T *operator->() {
return const_cast<T *>(static_cast<const atomix<T> *>(this)->operator->());
}
};


template<typename InputIt, typename OutputIt, typename Equal, typename Accum>
OutputIt uniq_accum(InputIt first,
const InputIt last,
OutputIt dest,
Equal equal,
Accum accum) {
if (first != last) {
auto x = *first;
while (++first != last) {
if (equal(x, *first)) {
x = accum(x, *first);
} else {
*dest++ = x;
x = *first;
}
}
*dest++ = x;
}
return dest;
}

double now_sec() {
return static_cast<std::chrono::duration<double> >(
std::chrono::system_clock::now().time_since_epoch()).count();
}

template<typename T>
std::vector<T> join(const std::vector<std::deque<T> > &xss) {
size_t len = 0;
for (auto &xs : xss) len += xs.size();

std::vector<T> ys;
ys.reserve(len);
for (auto &xs : xss) boost::copy(xs, std::back_inserter(ys));
return ys;
}


typedef uint32_t vint;  

constexpr vint vmax = std::numeric_limits<vint>::max();  

typedef std::pair<vint, float> edge;

struct atom {
atomix<float> str;    
atomix<vint> child;  

atom() : str(0.0), child(vmax) {}

atom(float s) : str(s), child(vmax) {}

atom(float s, vint c) : str(s), child(c) {}
} __attribute__((aligned(8)));

struct vertex {
atomix<atom> a;
atomix<vint> sibling;
vint united_child;

vertex(float str) : a(atom(str)), sibling(vmax), united_child(vmax) {}
};

struct graph {
numa_unique_ptr<atomix<vint>[]> coms;     
unique_c_ptr<vertex[]> vs;       
std::vector<std::vector<edge> > es;       
double tot_wgt;  
boost::optional<std::vector<vint> > tops;     

atomix<size_t> n_reunite;    
atomix<size_t> n_fail_lock;  
atomix<size_t> n_fail_cas;   
atomix<size_t> tot_nbrs;

graph() = default;

graph(graph &&x) = default;

graph &operator=(graph &&) = default;

graph(std::vector<std::vector<edge> > _es)
: coms(), vs(), es(std::move(_es)), tot_wgt(), tops(), n_reunite(),
n_fail_lock(), n_fail_cas(), tot_nbrs() {
const vint nvtx = static_cast<vint>(es.size());
vs = make_aligned_unique<vertex[]>(nvtx, sizeof(vertex));
coms = make_unique_interleaved<atomix<vint>[]>(nvtx);

double w = 0.0;
#pragma omp parallel for reduction(+:w)
for (vint v = 0; v < nvtx; ++v) {
float s = 0.0f;
for (auto &e : es[v]) s += e.second;
::new(&vs[v]) vertex(s);
w += s;
coms[v] = v;
}
tot_wgt = w;
}

vint n() const { return static_cast<vint>(es.size()); }
};


bool is_toplevel(const graph &, const vint);

bool is_merged(const graph &, const vint);

bool check_result(graph *const);

vint trace_com(const vint v, graph *const g) {
vint com = v;
for (;;) {
const vint c = g->coms[com];
if (c == com) break;
com = c;
}

if (v != com && g->coms[v] != com)
g->coms[v] = com;

return com;
}

template<typename InputIt, typename OutputIt>
OutputIt compact(InputIt it, const InputIt last, OutputIt result) {
if (it == last)
return result;

std::sort(it, last, [](auto &e0, auto &e1) { return e0.first < e1.first; });
return uniq_accum(it, last, result,
[](auto x, auto y) { return x.first == y.first; },
[](auto x, auto y) { return edge{x.first, x.second + y.second}; });
}

void unite(const vint v, std::vector<edge> *const nbrs, graph *const g) {
ptrdiff_t icmb = 0;

nbrs->clear();

const auto push_edges = [v, nbrs, g, &icmb](const vint u) {
const size_t cap = nbrs->capacity();
constexpr size_t npre = 8;  
auto &es = g->es[u];

for (size_t i = 0; i < es.size() && i < npre; ++i)
__builtin_prefetch(&g->coms[es[i].first], 0, 3);
for (size_t i = 0; i < es.size(); ++i) {
if (i + npre < es.size())
__builtin_prefetch(&g->coms[es[i + npre].first], 0, 3);
const vint c = trace_com(es[i].first, g);
if (c != v)  
nbrs->push_back({c, es[i].second});
}

#ifdef DEBUG
if (nbrs->size() > cap)
std::cerr << "WARNING: edge accumulation buffer is reallocated\n";
#else
static_cast<void>(cap);
#endif

if (nbrs->size() - icmb >= 2048) {
const auto it = nbrs->begin() + icmb;
icmb = compact(it, nbrs->end(), it) - nbrs->begin();
nbrs->resize(icmb);
}
};

push_edges(v);

while (g->vs[v].united_child != g->vs[v].a->child) {
const vint c = g->vs[v].a->child;
vint w;
for (w = c; w != vmax && w != g->vs[v].united_child; w = g->vs[w].sibling)
push_edges(w);

g->vs[v].united_child = c;
}

g->tot_nbrs.fetch_add(nbrs->size());

g->es[v].clear();
compact(nbrs->begin(), nbrs->end(), std::back_inserter(g->es[v]));
}

vint find_best(const graph &g, const vint v, const double vstr) {
double dmax = 0.0;
vint best = v;
for (const edge e : g.es[v]) {
const double d =
static_cast<double>(e.second) - vstr * g.vs[e.first].a->str / g.tot_wgt;
if (dmax < d) {
dmax = d;
best = e.first;
}
}
return best;
}

vint merge(const vint v, std::vector<edge> *const nbrs, graph *const g) {
unite(v, nbrs, g);

const float vstr = g->vs[v].a->str.exchange(-1);

if (g->vs[v].a->child != g->vs[v].united_child) {
unite(v, nbrs, g);
g->n_reunite.fetch_add(1);
}

const vint u = find_best(*g, v, vstr);
if (u == v) {
g->vs[v].a->str = vstr;
} else {
atom ua = g->vs[u].a;  
if (ua.str < 0.0) {
g->vs[v].a->str = vstr;
g->n_fail_lock.fetch_add(1);
return vmax;
}

g->vs[v].sibling = ua.child;

const atom _ua(ua.str + vstr, v);
if (!g->vs[u].a.compare_exchange_weak(ua, _ua)) {
g->vs[v].sibling = vmax;
g->vs[v].a->str = vstr;
g->n_fail_cas.fetch_add(1);
return vmax;
}

g->coms[v] = u;
}

assert(u != v || is_toplevel(*g, v));
assert(u == v || is_merged(*g, v));

return u;
}

std::unique_ptr<std::pair<vint, vint>[]> merge_order(const graph &g) {
auto ord = std::make_unique<std::pair<vint, vint>[]>(g.n());
#pragma omp parallel for
for (vint v = 0; v < g.n(); ++v)
ord[v] = {v, static_cast<vint>(g.es[v].size())};

__gnu_parallel::sort(&ord[0], &ord[g.n()],
[](auto x, auto y) { return x.second < y.second; });
return ord;
}

template<typename OutputIt, typename G>
void descendants(const G &g, vint v, OutputIt it) {
*it++ = v;
while ((v = g.vs[v].a->child) != vmax)
*it++ = v;
}

graph aggregate(std::vector<std::vector<edge> > adj) {
graph g(std::move(adj));
const auto ord = merge_order(g);
const int np = omp_get_max_threads();
size_t npend = 0;
double tmax = 0.0, ttotal = 0.0;
std::vector<std::deque<vint> > topss(np);

#pragma omp parallel reduction(+: npend) reduction(max: tmax) reduction(+: ttotal)
{
const double tstart = now_sec();
const int tid = omp_get_thread_num();
std::deque<vint> tops, pends;

std::vector<edge> nbrs;
nbrs.reserve(g.n() * 2);  

#pragma omp for schedule(static, 1)
for (vint i = 0; i < g.n(); ++i) {
pends.erase(boost::remove_if(pends, [&g, &tops, &nbrs](auto w) {
const vint u = merge(w, &nbrs, &g);
if (u == w) tops.push_back(w);
return u != vmax;  
}), pends.end());

const vint v = ord[i].first;
const vint u = merge(v, &nbrs, &g);
if (u == v) tops.push_back(v);
else if (u == vmax) pends.push_back(v);
}

ttotal = now_sec() - tstart;
tmax = ttotal;

#pragma omp barrier
#pragma omp critical
{
npend = pends.size();
for (const vint v : pends) {
const vint u = merge(v, &nbrs, &g);
if (u == v) tops.push_back(v);
assert(u != vmax);  
}
topss[tid] = std::move(tops);
}
}

g.tops = join(topss);

assert(([&g]() {
auto tops = *g.tops;
return g.tops->size() == boost::size(boost::unique(boost::sort(tops)));
})());

static_cast<void>(npend);  
static_cast<void>(tmax);

assert(check_result(&g));
return g;
}

std::unique_ptr<vint[]> compute_perm(const graph &g) {
auto perm = std::make_unique<vint[]>(g.n());
auto coms = std::make_unique<vint[]>(g.n());
const vint ncom = static_cast<vint>(g.tops->size());
std::vector<vint> offsets(ncom + 1);

const int np = omp_get_max_threads();
const vint ntask = std::min<vint>(ncom, 128 * np);
#pragma omp parallel
{
std::deque<vint> stack;

#pragma omp for schedule(dynamic, 1)
for (vint i = 0; i < ntask; ++i) {
for (vint comid = i; comid < ncom; comid += ntask) {
vint newid = 0;

descendants(g, (*g.tops)[comid], std::back_inserter(stack));

while (!stack.empty()) {
const vint v = stack.back();
stack.pop_back();

coms[v] = comid;
perm[v] = newid++;

if (g.vs[v].sibling != vmax)
descendants(g, g.vs[v].sibling, std::back_inserter(stack));
}

offsets[comid + 1] = newid;
}
}
}

boost::partial_sum(offsets, offsets.begin());
assert(offsets.back() == g.n());

#pragma omp parallel for schedule(static)
for (vint v = 0; v < g.n(); ++v)
perm[v] += offsets[coms[v]];

assert(([&g, &perm]() {
std::vector<vint> sorted(&perm[0], &perm[g.n()]);
return boost::equal(boost::sort(sorted),
boost::irange(static_cast<vint>(0), g.n()));
})());

return perm;
}


bool is_toplevel(const graph &g, const vint v) {
return g.vs[v].a->str >= 0.0 &&    
g.vs[v].sibling == vmax &&  
g.coms[v] == v;
}

bool is_merged(const graph &g, const vint v) {
return g.vs[v].a->str < 0.0 &&   
g.coms[v] != v;
};

bool check_result(graph *const pg) {
static_cast<void>(pg);

#ifndef NDEBUG
auto &g = *pg;
const auto vall = boost::irange(static_cast<vint>(0), g.n());
const auto istop = [&g](const vint v) { return is_toplevel(g, v); };

{
std::unordered_set<vint> topids;
for (vint v = 0; v < g.n(); ++v) topids.insert(trace_com(v, &g));
std::vector<vint> got(topids.begin(), topids.end());
auto ans = *g.tops;
assert(boost::equal(boost::sort(ans), boost::sort(got)));
}

assert(boost::algorithm::all_of(*g.tops, istop));
assert(boost::count_if(vall, istop) == static_cast<intmax_t>(g.tops->size()));

assert(boost::algorithm::all_of(vall, [&g](auto v) {
const vint c = trace_com(v, &g);
return is_toplevel(g, c);
}));

assert(boost::algorithm::all_of(vall, [&g](auto v) {
return is_toplevel(g, v) || is_merged(g, v);
}));
#endif

return true;
}

}  

using aux::vint;
using aux::now_sec;
using aux::edge;
using aux::trace_com;
using aux::aggregate;
using aux::compute_perm;

}  


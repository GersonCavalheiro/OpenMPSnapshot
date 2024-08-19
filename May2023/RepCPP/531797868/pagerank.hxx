#pragma once
#include <utility>
#include <vector>
#include <cmath>
#include <algorithm>
#include "_main.hxx"
#include "csr.hxx"
#include "vertices.hxx"
#include "transpose.hxx"
#include "dfs.hxx"
#include "components.hxx"

#ifdef OPENMP
#include <omp.h>
#endif

using std::tuple;
using std::vector;
using std::move;
using std::abs;
using std::max;





enum NormFunction {
L0_NORM = 0,
L1_NORM = 1,
L2_NORM = 2,
LI_NORM = 3
};


template <class V>
struct PagerankOptions {
int repeat;
int toleranceNorm;
V   tolerance;
V   damping;
int maxIterations;

PagerankOptions(int repeat=1, int toleranceNorm=LI_NORM, V tolerance=1e-10, V damping=0.85, int maxIterations=500) :
repeat(repeat), toleranceNorm(toleranceNorm), tolerance(tolerance), damping(damping), maxIterations(maxIterations) {}
};





template <class V>
struct PagerankResult {
vector<V> ranks;
int   iterations;
float time;

PagerankResult() :
ranks(), iterations(0), time(0) {}

PagerankResult(vector<V>&& ranks, int iterations=0, float time=0) :
ranks(ranks), iterations(iterations), time(time) {}

PagerankResult(vector<V>& ranks, int iterations=0, float time=0) :
ranks(move(ranks)), iterations(iterations), time(time) {}
};






template <class H, class V>
inline void pagerankInitialize(vector<V>& a, const H& xt) {
size_t N = xt.order();
xt.forEachVertexKey([&](auto v) {
a[v] = V(1)/N;
});
}


template <class H, class V>
inline void pagerankInitializeFrom(vector<V>& a, const H& xt, const vector<V>& q) {
xt.forEachVertexKey([&](auto v) {
a[v] = q[v];
});
}


#ifdef OPENMP
template <class H, class V>
inline void pagerankInitializeOmp(vector<V>& a, const H& xt) {
using  K = typename H::key_type;
size_t S = xt.span();
size_t N = xt.order();
#pragma omp parallel for schedule(auto)
for (K v=0; v<S; ++v) {
if (!xt.hasVertex(v)) continue;
a[v] = V(1)/N;
}
}

template <class H, class V>
inline void pagerankInitializeFromOmp(vector<V>& a, const H& xt, const vector<V>& q) {
using  K = typename H::key_type;
size_t S = xt.span();
#pragma omp parallel for schedule(auto)
for (K v=0; v<S; ++v) {
if (!xt.hasVertex(v)) continue;
a[v] = q[v];
}
}
#endif






template <class H, class V>
inline void pagerankFactor(vector<V>& a, const H& xt, V P) {
xt.forEachVertex([&](auto v, auto d) {
a[v] = d>0? P/d : 0;
});
}


#ifdef OPENMP
template <class H, class V>
inline void pagerankFactorOmp(vector<V>& a, const H& xt, V P) {
using  K = typename H::key_type;
size_t S = xt.span();
#pragma omp parallel for schedule(auto)
for (K v=0; v<S; ++v) {
if (!xt.hasVertex(v)) continue;
K  d = xt.vertexValue(v);
a[v] = d>0? P/d : 0;
}
}
#endif






template <class H, class V>
inline V pagerankTeleport(const H& xt, const vector<V>& r, V P) {
size_t N = xt.order();
V a = (1-P)/N;
xt.forEachVertex([&](auto v, auto d) {
if (d==0) a += P * r[v]/N;
});
return a;
}


#ifdef OPENMP
template <class H, class V>
inline V pagerankTeleportOmp(const H& xt, const vector<V>& r, V P) {
using  K = typename H::key_type;
size_t S = xt.span();
size_t N = xt.order();
V a = (1-P)/N;
#pragma omp parallel for schedule(auto) reduction(+:a)
for (K v=0; v<S; ++v) {
if (!xt.hasVertex(v)) continue;
K   d = xt.vertexValue(v);
if (d==0) a += P * r[v]/N;
}
return a;
}
#endif






template <class H, class K, class V>
inline V pagerankCalculateRankDelta(vector<V>& a, const H& xt, const vector<V>& r, const vector<V>& c, K v, V C0) {
V av = C0;
V rv = r[v];
xt.forEachEdgeKey(v, [&](auto u) {
av += c[u];
});
a[v] = av;
return av - rv;
}



template <class H, class V, class FA, class FP>
inline void pagerankCalculateRanks(vector<V>& a, const H& xt, const vector<V>& r, const vector<V>& c, V C0, V E, FA fa, FP fp) {
xt.forEachVertexKey([&](auto v) {
if (!fa(v)) return;
if (abs(pagerankCalculateRankDelta(a, xt, r, c, v, C0)) > E) fp(v);
});
}


#ifdef OPENMP
template <class H, class V, class FA, class FP>
inline void pagerankCalculateRanksOmp(vector<V>& a, const H& xt, const vector<V>& r, const vector<V>& c, V C0, V E, FA fa, FP fp) {
using  K = typename H::key_type;
size_t S = xt.span();
#pragma omp parallel for schedule(dynamic, 2048)
for (K v=0; v<S; ++v) {
if (!xt.hasVertex(v) || !fa(v)) continue;
if (abs(pagerankCalculateRankDelta(a, xt, r, c, v, C0)) > E) fp(v);
}
}
#endif






template <class V>
inline V pagerankError(const vector<V>& x, const vector<V>& y, int EF) {
switch (EF) {
case 1:  return l1Norm(x, y);
case 2:  return l2Norm(x, y);
default: return liNorm(x, y);
}
}


#ifdef OPENMP
template <class V>
inline V pagerankErrorOmp(const vector<V>& x, const vector<V>& y, int EF) {
switch (EF) {
case 1:  return l1NormOmp(x, y);
case 2:  return l2NormOmp(x, y);
default: return liNormOmp(x, y);
}
}
#endif






template <class G, class FT>
inline auto pagerankAffectedTraversal(const G& x, const G& y, FT ft) {
auto fn = [](auto u) {};
vector<bool> vis(max(x.span(), y.span()));
y.forEachVertexKey([&](auto u) {
if (!ft(u)) return;
dfsVisitedForEachW(vis, x, u, fn);
dfsVisitedForEachW(vis, y, u, fn);
});
return vis;
}



template <class G, class K>
inline auto pagerankAffectedTraversal(const G& x, const G& y, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions) {
auto fn = [](K u) {};
vector<bool> vis(max(x.span(), y.span()));
for (const auto& [u, v] : deletions)
dfsVisitedForEachW(vis, x, u, fn);
for (const auto& [u, v] : insertions)
dfsVisitedForEachW(vis, y, u, fn);
return vis;
}






template <class G, class FT>
inline auto pagerankAffectedFrontier(const G& x, const G& y, FT ft) {
auto fn = [](auto u) {};
vector<bool> vis(max(x.span(), y.span()));
y.forEachVertexKey([&](auto u) {
if (!ft(u)) return;
x.forEachEdgeKey(u, [&](auto v) { vis[v] = true; });
y.forEachEdgeKey(u, [&](auto v) { vis[v] = true; });
});
return vis;
}



template <class G, class K>
inline auto pagerankAffectedFrontier(const G& x, const G& y, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions) {
auto fn = [](K u) {};
vector<bool> vis(max(x.span(), y.span()));
for (const auto& [u, v] : deletions) {
vis[v] = true;
y.forEachEdgeKey(u, [&](auto v) { vis[v] = true; });
}
for (const auto& [u, v] : insertions)
y.forEachEdgeKey(u, [&](auto v) { vis[v] = true; });
return vis;
}






template <bool ASYNC=false, class H, class V, class FL, class FA, class FP>
PagerankResult<V> pagerankSeq(const H& xt, const vector<V> *q, const PagerankOptions<V>& o, FL fl, FA fa, FP fp) {
using  K = typename H::key_type;
size_t S = xt.span();
size_t N = xt.order();
V   P  = o.damping;
V   E  = o.tolerance;
int L  = o.maxIterations, l = 0;
int EF = o.toleranceNorm;
vector<int> e(S); vector<V> a(S), r(S), c(S), f(S);
float t = measureDuration([&]() {
fillValueU(e, 0);
if (q) pagerankInitializeFrom(r, xt, *q);
else   pagerankInitialize    (r, xt);
if (!ASYNC) copyValuesW(a, r);
pagerankFactor(f, xt, P); multiplyValuesW(c, r, f);  
l = fl(e, ASYNC? r : a, r, c, f, xt, P, E, L, EF, fa, fp);  
}, o.repeat);
return {r, l, t};
}





#ifdef OPENMP

template <bool ASYNC=false, class H, class V, class FL, class FA, class FP>
PagerankResult<V> pagerankOmp(const H& xt, const vector<V> *q, const PagerankOptions<V>& o, FL fl, FA fa, FP fp) {
using K  = typename H::key_type;
size_t S = xt.span();
size_t N = xt.order();
V   P  = o.damping;
V   E  = o.tolerance;
int L  = o.maxIterations, l = 0;
int EF = o.toleranceNorm;
vector<int> e(S); vector<V> a(S), r(S), c(S), f(S);
float t = measureDuration([&]() {
fillValueU(e, 0);
if (q) pagerankInitializeFromOmp(r, xt, *q);
else   pagerankInitializeOmp    (r, xt);
if (!ASYNC) copyValuesOmpW(a, r);
pagerankFactorOmp(f, xt, P); multiplyValuesOmpW(c, r, f);  
l = fl(e, ASYNC? r : a, r, c, f, xt, P, E, L, EF, fa, fp);  
}, o.repeat);
return {r, l, t};
}
#endif

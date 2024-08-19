#pragma once
#include <utility>
#include <cmath>
#include <array>
#include <omp.h>
#include "_main.hxx"
#include "Labelset.hxx"
#include "labelrank.hxx"

using std::pair;
using std::array;
using std::pow;





template <class G, class K, class V, size_t N>
void labelrankInitializeVertexW(ALabelset<K, V>& a, vector<Labelset<K, V, N>>& as, const G& x, K u, V e) {
V sumw = V(); a.clear();
x.forEachEdge(u, [&](auto v, auto w) {
a.set(v, w);
sumw += w;
});
labelsetReorderU(a);
labelsetCopyW(as[u], a);
labelsetMultiplyPowU(as[u], 1/sumw, e);
}



template <class G, class K, class V, size_t N>
void labelrankUpdateVertexW(ALabelset<K, V>& a, vector<Labelset<K, V, N>>& as, const vector<Labelset<K, V, N>>& ls, const G& x, K u, V e) {
V sumw = V(); a.clear();
x.forEachEdge(u, [&](auto v, auto w) {
labelsetCombineU(a, ls[v], w);
sumw += w;
});
labelsetReorderU(a);
labelsetCopyW(as[u], a);
labelsetMultiplyPowU(as[u], 1/sumw, e);
}



template <class G, class K, class V, size_t N>
bool labelrankIsVertexStable(const vector<Labelset<K, V, N>>& ls, const G& x, K u, V q) {
K count = K();
x.forEachEdgeKey(u, [&](auto v) {
if (labelsetIsSubset(ls[u], ls[v])) count++;
});
return count > q * x.degree(u);
}



template <class G, class K, class V, size_t N>
auto labelrankBestLabels(const vector<Labelset<K, V, N>>& ls, const G& x) {
vector<K> a(x.span());
x.forEachVertexKey([&](auto u) {
a[u] = ls[u][0].first;
});
return a;
}





template <size_t N, class G>
auto labelrankOmp(const G& x, const LabelrankOptions& o={}) {
using K = typename G::key_type;
using V = typename G::edge_value_type;
size_t S = x.span();
int    T = omp_get_max_threads();
vector<ALabelset<K, V>> la(T, ALabelset<K, V>(S));
vector<Labelset<K, V, N>> ls(S);
vector<Labelset<K, V, N>> ms(S);
float t = measureDurationMarked([&](auto mark) {
for (auto& a : la) a.clear();
ls.clear(); ls.resize(S);
ms.clear(); ms.resize(S);
mark([&]() {
#pragma omp parallel for schedule(monotonic:runtime)
for (K u=0; u<S; u++) {
int t = omp_get_thread_num();
if (!x.hasVertex(u)) continue;
labelrankInitializeVertexW(la[t], ls, x, u, V(o.inflation));
}
for (int i=0; i<o.maxIterations; ++i) {
#pragma omp parallel for schedule(monotonic:runtime)
for (K u=0; u<S; u++) {
int t = omp_get_thread_num();
if (!x.hasVertex(u)) continue;
if (labelrankIsVertexStable(ls, x, u, V(o.conditionalUpdate))) ms[u] = ls[u];
else labelrankUpdateVertexW(la[t], ms, ls, x, u, V(o.inflation));
}
swap(ls, ms);
}
});
}, o.repeat);
return LabelrankResult(labelrankBestLabels(ls, x), o.maxIterations, t);
}





template <size_t N, class G>
auto labelrankSeq(const G& x, const LabelrankOptions& o={}) {
using K = typename G::key_type;
using V = typename G::edge_value_type;
ALabelset<K, V> la(x.span());
vector<Labelset<K, V, N>> ls(x.span());
vector<Labelset<K, V, N>> ms(x.span());
float t = measureDurationMarked([&](auto mark) {
la.clear();
ls.clear(); ls.resize(x.span());
ms.clear(); ms.resize(x.span());
mark([&]() {
x.forEachVertexKey([&](auto u) {
labelrankInitializeVertexW(la, ls, x, u, V(o.inflation));
});
for (int i=0; i<o.maxIterations; ++i) {
x.forEachVertexKey([&](auto u) {
if (labelrankIsVertexStable(ls, x, u, V(o.conditionalUpdate))) ms[u] = ls[u];
else labelrankUpdateVertexW(la, ms, ls, x, u, V(o.inflation));
});
swap(ls, ms);
}
});
}, o.repeat);
return LabelrankResult(labelrankBestLabels(ls, x), o.maxIterations, t);
}

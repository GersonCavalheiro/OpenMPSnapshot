#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "dynamic.hxx"
#include "pagerank.hxx"

using std::vector;
using std::swap;





template <class G, class H, class T>
auto pagerankVertices(const G& x, const H& xt, const PagerankOptions<T>& o, const PagerankData<G> *D=nullptr) {
using K = typename G::key_type;
if (!o.splitComponents) return vertexKeys(xt);
return joinValuesVector(componentsD(x, xt, D));
}


template <class G, class H, class T>
auto pagerankDynamicVertices(const G& x, const H& xt, const G& y, const H& yt, const PagerankOptions<T>& o, const PagerankData<G> *D=nullptr) {
using K = typename G::key_type;
if (!o.splitComponents) return dynamicInVertices(x, xt, y, yt);
const auto& cs = componentsD(y, yt, D);
const auto& b  = blockgraphD(y, cs, D);
auto [is, n] = dynamicInComponentIndices(x, xt, y, yt, cs, b);
auto ks = joinAtVector<K>(cs, sliceIterable(is, 0, n)); size_t nv = ks.size();
joinAtU(ks, cs, sliceIterable(is, n));
return make_pair(ks, nv);
}





template <class G, class H, class T>
auto pagerankComponents(const G& x, const H& xt, const PagerankOptions<T>& o, const PagerankData<G> *D=nullptr) {
using K = typename G::key_type;
if (!o.splitComponents) return vector2d<K> {vertexKeys(xt)};
return componentsD(x, xt, D);
}


template <class G, class H>
auto pagerankDynamicComponentsDefault(const G& x, const H& xt, const G& y, const H& yt) {
using K = typename G::key_type;
vector2d<K> a;
auto [ks, n] = dynamicInVertices(x, xt, y, yt);
a.push_back(vector<K>(ks.begin(), ks.begin()+n));
a.push_back(vector<K>(ks.begin()+n, ks.end()));
return make_pair(a, size_t(1));
}

template <class G, class H, class T>
auto pagerankDynamicComponentsSplit(const G& x, const H& xt, const G& y, const H& yt, const PagerankOptions<T>& o, const PagerankData<G> *D=nullptr) {
using K = typename G::key_type;
const auto& cs = componentsD(y, yt, D);
const auto& b  = blockgraphD(y, cs, D);
auto [is, n] = dynamicInComponentIndices(x, xt, y, yt, cs, b);
vector2d<K> a;
for (auto i : is)
a.push_back(cs[i]);
return make_pair(a, n);
}

template <class G, class H, class T>
auto pagerankDynamicComponents(const G& x, const H& xt, const G& y, const H& yt, const PagerankOptions<T>& o, const PagerankData<G> *D=nullptr) {
if (o.splitComponents) return pagerankDynamicComponentsSplit(x, xt, y, yt, o, D);
return pagerankDynamicComponentsDefault(x, xt, y, yt);
}





template <class K, class T>
void pagerankFactorW(vector<T>& a, const vector<K>& vdata, K i, K n, T p) {
for (K u=i; u<i+n; u++) {
K  d = vdata[u];
a[u] = d>0? p/d : 0;
}
}





template <class K, class T>
T pagerankTeleport(const vector<T>& r, const vector<K>& vdata, K N, T p) {
T a = (1-p)/N;
for (K u=0; u<N; u++)
if (vdata[u] == 0) a += p*r[u]/N;
return a;
}





template <class K, class T>
void pagerankCalculateW(vector<T>& a, const vector<T>& c, const vector<K>& vfrom, const vector<K>& efrom, K i, K n, T c0) {
for (K v=i; v<i+n; v++)
a[v] = c0 + sumValuesAt(c, sliceIterable(efrom, vfrom[v], vfrom[v+1]));
}

template <class K, class T>
void pagerankCalculateOrderedU(vector<T>& e, vector<T>& r, const vector<T>& f, const vector<K>& vfrom, const vector<K>& efrom, K i, K n, T c0) {
for (K v=i; v<i+n; v++) {
T a = c0;
for (K u : sliceIterable(efrom, vfrom[v], vfrom[v+1]))
a += f[u] * r[u];
e[v] = a - r[v];
r[v] = a;
}
}





template <class K, class T>
T pagerankError(const vector<T>& x, const vector<T>& y, K i, K N, int EF) {
switch (EF) {
case 1:  return l1Norm(x, y, i, N);
case 2:  return l2Norm(x, y, i, N);
default: return liNorm(x, y, i, N);
}
}

template <class K, class T>
T pagerankError(const vector<T>& x, K i, K N, int EF) {
switch (EF) {
case 1:  return l1Norm(x, i, N);
case 2:  return l2Norm(x, i, N);
default: return liNorm(x, i, N);
}
}





template <class H, class J, class M, class FL, class T=float>
PagerankResult<T> pagerankSeq(const H& xt, const J& ks, size_t i, const M& ns, FL fl, const vector<T> *q, const PagerankOptions<T>& o) {
using K = typename H::key_type;
K    N  = xt.order();
T    p  = o.damping;
T    E  = o.tolerance;
int  L  = o.maxIterations, l = 0;
int  EF = o.toleranceNorm;
auto vfrom = sourceOffsetsAs(xt, ks, K());
auto efrom = destinationIndicesAs(xt, ks, K());
auto vdata = vertexData(xt, ks);
vector<T> a(N), r(N), c(N), f(N), qc;
if (q) qc = compressContainer(xt, *q, ks);
float t = measureDuration([&]() {
if (q) copyValuesW(r, qc);   
else fillValueU(r, T(1)/N);
pagerankFactorW(f, vdata, 0, N, p); multiplyValuesW(c, r, f, 0, N);           
l = fl(a, r, c, f, vfrom, efrom, vdata, K(i), ns, N, p, E, L, EF, K(), K());  
}, o.repeat);
return {decompressContainer(xt, r, ks), l, t};
}

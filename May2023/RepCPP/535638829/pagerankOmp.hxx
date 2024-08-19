#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "pagerank.hxx"
#include "pagerankSeq.hxx"

using std::vector;
using std::swap;





template <class K, class T>
void pagerankFactorOmpW(vector<T>& a, const vector<K>& vdata, K i, K n, T p) {
if (n<SIZE_MIN_OMPM) { pagerankFactorW(a, vdata, i, n, p); return; }
#pragma omp parallel for schedule(auto)
for (K u=i; u<i+n; u++) {
K  d = vdata[u];
a[u] = d>0? p/d : 0;
}
}





template <class K, class T>
T pagerankTeleportOmp(const vector<T>& r, const vector<K>& vdata, K N, T p) {
if (N<SIZE_MIN_OMPR) return pagerankTeleport(r, vdata, N, p);
T a = (1-p)/N;
#pragma omp parallel for schedule(auto) reduction(+:a)
for (K u=0; u<N; u++)
if (vdata[u] == 0) a += p*r[u]/N;
return a;
}





template <class K, class T>
void pagerankCalculateOmpW(vector<T>& a, const vector<T>& c, const vector<K>& vfrom, const vector<K>& efrom, K i, K n, T c0) {
if (n<SIZE_MIN_OMPM) { pagerankCalculateW(a, c, vfrom, efrom, i, n, c0); return; }
#pragma omp parallel for schedule(dynamic, 2048)
for (K v=i; v<i+n; v++)
a[v] = c0 + sumValuesAt(c, sliceIterable(efrom, vfrom[v], vfrom[v+1]));
}

template <class K, class T>
void pagerankCalculateOrderedOmpU(vector<T>& e, vector<T>& r, const vector<T>& f, const vector<K>& vfrom, const vector<K>& efrom, K i, K n, T c0) {
if (n<SIZE_MIN_OMPM) { pagerankCalculateOrderedU(e, r, f, vfrom, efrom, i, n, c0); return; }
#pragma omp parallel for schedule(dynamic, 2048)
for (K v=i; v<i+n; v++) {
T a = c0;
for (K u : sliceIterable(efrom, vfrom[v], vfrom[v+1]))
a += f[u] * r[u];
e[v] = a - r[v];
r[v] = a;
}
}





template <class K, class T>
T pagerankErrorOmp(const vector<T>& x, const vector<T>& y, K i, K N, int EF) {
switch (EF) {
case 1:  return l1NormOmp(x, y, i, N);
case 2:  return l2NormOmp(x, y, i, N);
default: return liNormOmp(x, y, i, N);
}
}

template <class K, class T>
T pagerankErrorOmp(const vector<T>& x, K i, K N, int EF) {
switch (EF) {
case 1:  return l1NormOmp(x, i, N);
case 2:  return l2NormOmp(x, i, N);
default: return liNormOmp(x, i, N);
}
}





template <class H, class J, class M, class FL, class T=float>
PagerankResult<T> pagerankOmp(const H& xt, const J& ks, size_t i, const M& ns, FL fl, const vector<T> *q, const PagerankOptions<T>& o) {
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
if (q) copyValuesOmpW(r, qc);  
else fillValueOmpU(r, T(1)/N);
pagerankFactorOmpW(f, vdata, 0, N, p); multiplyValuesOmpW(c, r, f, 0, N);  
l = fl(a, r, c, f, vfrom, efrom, vdata, i, ns, N, p, E, L, EF, K(), K());  
}, o.repeat);
return {decompressContainer(xt, r, ks), l, t};
}

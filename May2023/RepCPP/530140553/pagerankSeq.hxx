#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "pagerank.hxx"

using std::vector;
using std::swap;





template <class T>
void pagerankFactorW(vector<T>& a, const vector<int>& vdata, int i, int n, T p) {
for (int u=i; u<i+n; u++) {
int d = vdata[u];
a[u] = d>0? p/d : 0;
}
}





template <class T>
T pagerankTeleport(const vector<T>& r, const vector<int>& vdata, int N, T p) {
T a = (1-p)/N;
for (int u=0; u<N; u++)
if (vdata[u] == 0) a += p*r[u]/N;
return a;
}





template <class T>
void pagerankCalculateW(vector<T>& a, const vector<T>& c, const vector<int>& vfrom, const vector<int>& efrom, int i, int n, T c0) {
for (int v=i; v<i+n; v++)
a[v] = c0 + sumValuesAt(c, sliceIterable(efrom, vfrom[v], vfrom[v+1]));
}

template <class T>
void pagerankCalculateOrderedU(vector<T>& e, vector<T>& r, const vector<T>& f, const vector<int>& vfrom, const vector<int>& efrom, int i, int n, T c0) {
for (int v=i; v<i+n; v++) {
T a = c0;
for (int u : sliceIterable(efrom, vfrom[v], vfrom[v+1]))
a += f[u] * r[u];
e[v] = a - r[v];
r[v] = a;
}
}





template <class T>
T pagerankError(const vector<T>& x, const vector<T>& y, int i, int N, int EF) {
switch (EF) {
case 1:  return l1Norm(x, y, i, N);
case 2:  return l2Norm(x, y, i, N);
default: return liNorm(x, y, i, N);
}
}

template <class T>
T pagerankError(const vector<T>& x, int i, int N, int EF) {
switch (EF) {
case 1:  return l1Norm(x, i, N);
case 2:  return l2Norm(x, i, N);
default: return liNorm(x, i, N);
}
}





template <class H, class J, class M, class FL, class T=float>
PagerankResult<T> pagerankSeq(const H& xt, const J& ks, int i, const M& ns, FL fl, const vector<T> *q, const PagerankOptions<T>& o) {
int  N  = xt.order();
T    p  = o.damping;
T    E  = o.tolerance;
int  L  = o.maxIterations, l = 0;
int  EF = o.toleranceNorm;
auto vfrom = sourceOffsetsAs(xt, ks, int());
auto efrom = destinationIndicesAs(xt, ks, int());
auto vdata = vertexData(xt, ks);
vector<T> a(N), r(N), c(N), f(N), qc;
if (q) qc = compressContainer(xt, *q, ks);
float t = measureDuration([&]() {
if (q) copyValuesW(r, qc);   
else fillValueU(r, T(1)/N);
pagerankFactorW(f, vdata, 0, N, p); multiplyValuesW(c, r, f, 0, N);  
l = fl(a, r, c, f, vfrom, efrom, vdata, i, ns, N, p, E, L, EF);      
}, o.repeat);
return {decompressContainer(xt, r, ks), l, t};
}

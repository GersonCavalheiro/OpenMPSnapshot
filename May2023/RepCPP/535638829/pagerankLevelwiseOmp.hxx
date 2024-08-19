#pragma once
#include <utility>
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "transpose.hxx"
#include "components.hxx"
#include "sort.hxx"
#include "dynamic.hxx"
#include "pagerank.hxx"
#include "pagerankOmp.hxx"
#include "pagerankMonolithicOmp.hxx"

using std::vector;
using std::swap;
using std::move;





template <bool O, bool D, class K, class T, class J, bool F=false>
int pagerankLevelwiseOmpLoopU(vector<T>& a, vector<T>& r, vector<T>& c, const vector<T>& f, const vector<K>& vfrom, const vector<K>& efrom, const vector<K>& vdata, K i, const J& ns, K N, T p, T E, int L, int EF, K EI=K(), K EN=K()) {
float l = 0;
if (D) return 0;
for (K n : ns) {
if (n<=0) { i += -n; continue; }
T    E1 = EF<=2? E*n/N : E;
int  l1 = pagerankMonolithicOmpLoopU<O, D>(a, r, c, f, vfrom, efrom, vdata, i, n, N, p, E1, L, EF);
l += l1 * float(n)/N;
i += n;
if (!O) swap(a, r);
}
return int(l + 0.5f);
}





template <bool O, bool D, class G, class H, class T=float>
PagerankResult<T> pagerankLevelwiseOmp(const G& x, const H& xt, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *C=nullptr) {
using K = typename G::key_type;
K    N  = xt.order();  if (N==0) return PagerankResult<T>::initial(xt, q);
const auto& cs = componentsD(x, xt, C);
const auto& b  = blockgraphD(x, cs, C);
const auto& bt = blockgraphTransposeD(b, C);
auto gs = levelwiseGroupedComponentsFrom(cs, b, bt);
auto ns = transformIterable(gs, [&](const auto& g) { return g.size(); });
auto ks = joinValuesVector(gs);
return pagerankOmp(xt, ks, 0, ns, pagerankLevelwiseOmpLoopU<O, D, K, T, decltype(ns)>, q, o);
}
template <bool O, bool D, class G, class T=float>
PagerankResult<T> pagerankLevelwiseOmp(const G& x, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *C=nullptr) {
auto xt = transposeWithDegree(x);
return pagerankLevelwiseOmp(x, xt, q, o, C);
}





template <bool O, bool D, class G, class H, class T=float>
PagerankResult<T> pagerankLevelwiseOmpDynamic(const G& x, const H& xt, const G& y, const H& yt, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *C=nullptr) {
using K = typename G::key_type;
K    N  = yt.order();  if (N==0) return PagerankResult<T>::initial(yt, q);
const auto& cs = componentsD(x, xt, C);
const auto& b  = blockgraphD(x, cs, C);
const auto& bt = blockgraphTransposeD(b, C);
auto gi = levelwiseGroupIndices(b, bt);
auto [is, n] = dynamicInComponentIndices(x, xt, y, yt, cs, b);  if (n==0) return PagerankResult<T>::initial(yt, q);
auto ig = groupValuesVector(sliceIterable(is, 0, n), [&](K i) { return gi[i]; });
auto gs = joinAt2dVector(cs, ig);
auto ns = transformIterable(gs, [&](const auto& g) { return g.size(); });
auto ks = joinValuesVector(gs); joinAtU(ks, cs, sliceIterable(is, n));
return pagerankOmp(yt, ks, 0, ns, pagerankLevelwiseOmpLoopU<O, D, K, T, decltype(ns)>, q, o);
}
template <class G, class T=float>
PagerankResult<T> pagerankLevelwiseOmpDynamic(const G& x, const G& y, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *C=nullptr) {
auto xt = transposeWithDegree(x);
auto yt = transposeWithDegree(y);
return pagerankLevelwiseOmpDynamic(x, xt, y, yt, q, o, C);
}

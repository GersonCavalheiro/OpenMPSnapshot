#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "transpose.hxx"
#include "dynamic.hxx"
#include "pagerank.hxx"
#include "pagerankSeq.hxx"
#include "pagerankOmp.hxx"

using std::vector;
using std::swap;





template <bool O, bool D, class K, class T, bool F=false>
int pagerankMonolithicOmpLoopU(vector<T>& a, vector<T>& r, vector<T>& c, const vector<T>& f, const vector<K>& vfrom, const vector<K>& efrom, const vector<K>& vdata, K i, K n, K N, T p, T E, int L, int EF, K EI=K(), K EN=K()) {
int l = 0;
if (F) EI = 0;
if (F) EN = N;
while (!O && l<L) {
T c0 = D? pagerankTeleportOmp(r, vdata, N, p) : (1-p)/N;
pagerankCalculateOmpW(a, c, vfrom, efrom, i, n, c0);    
multiplyValuesOmpW(c, a, f, i, n);                      
T el = pagerankErrorOmp(a, r, EN? EI:i, EN? EN:n, EF);  
swap(a, r); ++l;                                        
if (el<E) break;                                        
}
while (O && l<L) {
T c0 = D? pagerankTeleportOmp(r, vdata, N, p) : (1-p)/N;
pagerankCalculateOrderedOmpU(a, r, f, vfrom, efrom, i, n, c0);  
T el = pagerankErrorOmp(a, EN? EI:i, EN? EN:n, EF); ++l;        
if (el<E) break;                                                
}
return l;
}





template <bool O, bool D, class G, class H, class T=float>
PagerankResult<T> pagerankMonolithicOmp(const G& x, const H& xt, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *C=nullptr) {
using K = typename G::key_type;
K    N  = xt.order();  if (N==0) return PagerankResult<T>::initial(xt, q);
auto ks = pagerankVertices(x, xt, o, C);
return pagerankOmp(xt, ks, K(0), N, pagerankMonolithicOmpLoopU<O, D, K, T>, q, o);
}

template <bool O, bool D, class G, class T=float>
PagerankResult<T> pagerankMonolithicOmp(const G& x, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *C=nullptr) {
auto xt = transposeWithDegree(x);
return pagerankMonolithicOmp<O, D>(x, xt, q, o, C);
}





template <bool O, bool D, class G, class H, class T=float>
PagerankResult<T> pagerankMonolithicOmpDynamic(const G& x, const H& xt, const G& y, const H& yt, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *C=nullptr) {
using K = typename G::key_type;
K     N = yt.order();                                        if (N==0) return PagerankResult<T>::initial(yt, q);
auto [ks, n] = pagerankDynamicVertices(x, xt, y, yt, o, C);  if (n==0) return PagerankResult<T>::initial(yt, q);
return pagerankOmp(yt, ks, K(0), n, pagerankMonolithicOmpLoopU<O, D, K, T>, q, o);
}

template <bool O, bool D, class G, class T=float>
PagerankResult<T> pagerankMonolithicOmpDynamic(const G& x, const G& y, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *C=nullptr) {
auto xt = transposeWithDegree(x);
auto yt = transposeWithDegree(y);
return pagerankMonolithicOmpDynamic<O, D>(x, xt, y, yt, q, o, C);
}

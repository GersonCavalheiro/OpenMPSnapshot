#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "transpose.hxx"
#include "dynamic.hxx"
#include "pagerank.hxx"
#include "pagerankSeq.hxx"
#include "pagerankOmp.hxx"
#include "pagerankMonolithicSeq.hxx"

using std::vector;
using std::swap;
using std::min;





template <bool O, bool D, class T>
int pagerankBarrierfreeOmpLoopU(vector<T>& a, vector<T>& r, vector<T>& c, const vector<T>& f, const vector<int>& vfrom, const vector<int>& efrom, const vector<int>& vdata, int i, int n, int N, T p, T E, int L, int EF) {
float l = 0;
if (!O) return 0;
int TS = omp_get_max_threads();
int DN = ceilDiv(n, TS);
#pragma omp parallel for schedule(static, 1)
for (int t=0; t<TS; t++) {
int ti = i + t*DN;
int tI = min(ti + DN, i + n);
int tn = tI - ti;
int tl = pagerankMonolithicSeqLoopU<O, D, T>(a, r, c, f, vfrom, efrom, vdata, ti, tn, N, p, E, L, EF);
l += tl * tn/n;
}
return int(l + 0.5f);
}





template <bool O, bool D, class G, class H, class T=float>
PagerankResult<T> pagerankBarrierfreeOmp(const G& x, const H& xt, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}) {
int  N  = xt.order();  if (N==0) return PagerankResult<T>::initial(xt, q);
auto ks = vertexKeys(xt);
return pagerankOmp(xt, ks, 0, N, pagerankBarrierfreeOmpLoopU<O, D, T>, q, o);
}

template <bool O, bool D, class G, class T=float>
PagerankResult<T> pagerankBarrierfreeOmp(const G& x, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}) {
auto xt = transposeWithDegree(x);
return pagerankBarrierfreeOmp<O, D>(x, xt, q, o);
}





template <bool O, bool D, class G, class H, class T=float>
PagerankResult<T> pagerankBarrierfreeOmpDynamic(const G& x, const H& xt, const G& y, const H& yt, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}) {
int  N = yt.order();                             if (N==0) return PagerankResult<T>::initial(yt, q);
auto [ks, n] = dynamicInVertices(x, xt, y, yt);  if (n==0) return PagerankResult<T>::initial(yt, q);
return pagerankOmp(yt, ks, 0, n, pagerankBarrierfreeOmpLoopU<O, D, T>, q, o);
}

template <bool O, bool D, class G, class T=float>
PagerankResult<T> pagerankBarrierfreeOmpDynamic(const G& x, const G& y, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}) {
auto xt = transposeWithDegree(x);
auto yt = transposeWithDegree(y);
return pagerankBarrierfreeOmpDynamic<O, D>(x, xt, y, yt, q, o);
}

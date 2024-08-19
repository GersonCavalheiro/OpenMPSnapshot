#pragma once
#include <utility>
#include <algorithm>
#include <vector>
#include "_main.hxx"
#include "vertices.hxx"
#include "pagerank.hxx"

using std::tuple;
using std::vector;
using std::swap;






template <bool ASYNC=false, bool DEAD=false, class H, class K, class V, class FA, class FP>
inline int pagerankBasicSeqLoop(vector<int>& e, vector<V>& a, vector<V>& r, vector<V>& c, const vector<V>& f, const H& xt, V P, V E, int L, int EF, FA fa, FP fp) {
size_t N = xt.order();
int    l = 0;
while (l<L) {
V C0 = DEAD? pagerankTeleport(xt, r, P) : (1-P)/N;
pagerankCalculateRanks(a, xt, r, c, C0, E, fa, fp); ++l;  
multiplyValuesW(c, a, f);        
V el = pagerankError(a, r, EF);  
if (!ASYNC) swap(a, r);          
if (el<E) break;                 
}
return l;
}


#ifdef OPENMP
template <bool ASYNC=false, bool DEAD=false, class H, class K, class V, class FA, class FP>
inline int pagerankBasicOmpLoop(vector<int>& e, vector<V>& a, vector<V>& r, vector<V>& c, const vector<V>& f, const H& xt, V P, V E, int L, int EF, FA fa, FP fp) {
size_t N = xt.order();
int    l = 0;
while (l<L) {
V C0 = DEAD? pagerankTeleportOmp(xt, r, P) : (1-P)/N;
pagerankCalculateRanksOmp(a, xt, r, c, C0, E, fa, fp); ++l;  
multiplyValuesOmpW(c, a, f);        
V el = pagerankErrorOmp(a, r, EF);  
if (!ASYNC) swap(a, r);             
if (el<E) break;                    
}
return l;
}
#endif






template <bool ASYNC=false, bool DEAD=false, class H, class V>
inline PagerankResult<V> pagerankBasicSeq(const H& xt, const vector<V> *q, const PagerankOptions<V>& o) {
using K = typename H::key_type;
if (xt.empty()) return {};
auto fa = [](K u) { return true; };
auto fp = [](K u) {};
return pagerankSeq<ASYNC>(xt, q, o, pagerankBasicSeqLoop<ASYNC, DEAD, H, K, V, decltype(fa), decltype(fp)>, fa, fp);
}


#ifdef OPENMP
template <bool ASYNC=false, bool DEAD=false, class H, class V>
inline PagerankResult<V> pagerankBasicOmp(const H& xt, const vector<V> *q, const PagerankOptions<V>& o) {
using K = typename H::key_type;
if (xt.empty()) return {};
auto fa = [](K u) { return true; };
auto fp = [](K u) {};
return pagerankOmp<ASYNC>(xt, q, o, pagerankBasicOmpLoop<ASYNC, DEAD, H, K, V, decltype(fa), decltype(fp)>, fa, fp);
}
#endif






template <bool ASYNC=false, bool DEAD=false, class G, class H, class K, class V>
inline PagerankResult<V> pagerankBasicDynamicTraversalSeq(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
if (yt.empty()) return {};
auto vaff = pagerankAffectedTraversal(x, y, deletions, insertions);
auto fa   = [&](K u) { return vaff[u]==true; };
auto fp   = [ ](K u) {};
return pagerankSeq<ASYNC>(yt, q, o, pagerankBasicSeqLoop<ASYNC, DEAD, H, K, V, decltype(fa), decltype(fp)>, fa, fp);
}


#ifdef OPENMP
template <bool ASYNC=false, bool DEAD=false, class G, class H, class K, class V>
inline PagerankResult<V> pagerankBasicDynamicTraversalOmp(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
if (yt.empty()) return {};
auto vaff = pagerankAffectedTraversal(x, y, deletions, insertions);
auto fa   = [&](K u) { return vaff[u]==true; };
auto fp   = [ ](K u) {};
return pagerankOmp<ASYNC>(yt, q, o, pagerankBasicOmpLoop<ASYNC, DEAD, H, K, V, decltype(fa), decltype(fp)>, fa, fp);
}
#endif






template <bool ASYNC=false, bool DEAD=false, class G, class H, class K, class V>
inline PagerankResult<V> pagerankBasicDynamicFrontierSeq(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
if (yt.empty()) return {};
auto vaff = pagerankAffectedFrontier(x, y, deletions, insertions);
auto fa   = [&](K u) { return vaff[u]==true; };
auto fp   = [&](K u) { y.forEachEdgeKey(u, [&](auto v) { vaff[v] = true; }); };
return pagerankSeq<ASYNC>(yt, q, o, pagerankBasicSeqLoop<ASYNC, DEAD, H, K, V, decltype(fa), decltype(fp)>, fa, fp);
}


#ifdef OPENMP
template <bool ASYNC=false, bool DEAD=false, class G, class H, class K, class V>
inline PagerankResult<V> pagerankBasicDynamicFrontierOmp(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
if (yt.empty()) return {};
auto vaff = pagerankAffectedFrontier(x, y, deletions, insertions);
auto fa   = [&](K u) { return vaff[u]==true; };
auto fp   = [&](K u) { y.forEachEdgeKey(u, [&](auto v) { vaff[v] = true; }); };
return pagerankOmp<ASYNC>(yt, q, o, pagerankBasicOmpLoop<ASYNC, DEAD, H, K, V, decltype(fa), decltype(fp)>, fa, fp);
}
#endif

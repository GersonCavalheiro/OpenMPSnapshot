#pragma once






template <class G, class K>
inline double edgeWeight(const G& x, K u) {
double a = 0;
x.forEachEdgeValue(u, [&](auto w) { a += w; });
return a;
}



template <class G>
inline double edgeWeight(const G& x) {
double a = 0;
x.forEachVertexKey([&](auto u) { a += edgeWeight(x, u); });
return a;
}

template <class G>
inline double edgeWeightOmp(const G& x) {
using K = typename G::key_type;
double a = 0;
size_t S = x.span();
#pragma omp parallel for schedule(auto) reduction(+:a)
for (K u=0; u<S; ++u) {
if (!x.hasVertex(u)) continue;
a += edgeWeight(x, u);
}
return a;
}





template <class G, class K>
inline void degreesW(vector<K>& a, const G& x) {
x.forEachVertexKey([&](auto u) { a[u] = x.degree(u); });
}

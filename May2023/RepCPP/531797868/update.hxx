#pragma once
#include <cstdint>
#include <utility>
#include <vector>

#ifdef OPENMP
#include <omp.h>
#endif

using std::pair;
using std::vector;





template <class G, class K, class E, class FT>
inline void addEdgeU(G &a, K u, K v, E w, FT ft) {
a.addEdge(u, v, w, ft);
}
template <class G, class K, class E>
inline void addEdgeU(G& a, K u, K v, E w=E()) {
a.addEdge(u, v, w);
}


#ifdef OPENMP
template <class G, class K, class E>
inline void addEdgeOmpU(G& a, K u, K v, E w=E()) {
auto ft = [](K u) { return belongsOmp(u); };
a.addEdge(u, v, w, ft);
}
#endif





template <class G, class K, class FT>
inline void removeEdgeU(G &a, K u, K v, FT ft) {
a.removeEdge(u, v, ft);
}
template <class G, class K>
inline void removeEdgeU(G& a, K u, K v) {
a.removeEdge(u, v);
}


#ifdef OPENMP
template <class G, class K>
inline void removeEdgeOmpU(G& a, K u, K v) {
auto ft = [](K u) { return belongsOmp(u); };
a.removeEdge(u, v, ft);
}
#endif





template <class G>
inline void updateU(G& a) {
a.update();
}


#ifdef OPENMP
template <class G>
inline void updateOmpU(G& a) {
using  K = typename G::key_type;
using  E = typename G::edge_value_type;
size_t S = a.span();
int THREADS = omp_get_max_threads();
vector<vector<pair<K, E>>*> bufs(THREADS);
for (int i=0; i<THREADS; ++i)
bufs[i] = new vector<pair<K, E>>();
#pragma omp parallel for schedule(auto)
for (K u=0; u<S; ++u) {
int t = omp_get_thread_num();
a.updateEdges(u, bufs[t]);
}
a.update();
for (int i=0; i<THREADS; ++i)
delete bufs[i];
}
#endif

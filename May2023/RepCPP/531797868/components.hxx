#pragma once
#include <vector>
#include "_main.hxx"
#include "vertices.hxx"
#include "dfs.hxx"

using std::vector;





template <class G, class H>
inline auto components(const G& x, const H& xt) {
using K = typename G::key_type;
vector2d<K> a; vector<K> vs;
vector<bool> vis(x.span());
x.forEachVertexKey([&](auto u) {
if (vis[u]) return;
dfsEndVisitedForEachW(vis, x, u, [&](auto v) { vs.push_back(v); });
});
fillValueU(vis, false);
while (!vs.empty()) {
auto u = vs.back(); vs.pop_back();
if (vis[u]) continue;
vector<K> c;
dfsVisitedForEachW(vis, xt, u, [&](auto v) { c.push_back(v); });
a.push_back(move(c));
}
return a;
}





template <class G, class K>
inline auto componentIds(const G& x, const vector2d<K>& cs) {
auto a = createContainer(x, K()); K i = 0;
for (const auto& c : cs) {
for (K u : c)
a[u] = i;
i++;
}
return a;
}





template <class H, class G, class K>
inline void blockgraphW(H& a, const G& x, const vector2d<K>& cs) {
auto c = componentIds(x, cs);
x.forEachVertexKey([&](auto u) {
a.addVertex(c[u]);
x.forEachEdgeKey(u, [&](auto v) {
if (c[u] != c[v]) a.addEdge(c[u], c[v]);
});
});
a.update();
}
template <class G, class K>
inline auto blockgraph(const G& x, const vector2d<K>& cs) {
G a; blockgraphW(a, x, cs);
return a;
}

#pragma once
#include <vector>

using std::vector;






template <class B=bool, class G, class K, class F>
inline void dfsVisitedForEachW(vector<B>& vis, const G& x, K u, F fn) {
if (vis[u]) return;
vis[u] = B(1); fn(u);
x.forEachEdgeKey(u, [&](K v) {
if (!vis[v]) dfsVisitedForEachW(vis, x, v, fn);
});
}
template <class B=bool, class G, class K, class F>
inline vector<B> dfsVisitedForEach(const G& x, K u, F fn) {
vector<B> vis(x.span());
dfsVisitedForEachW(vis, x, u, fn);
return vis;
}


template <class B=bool, class G, class K, class F>
inline void dfsVisitedForEachNonRecursiveW(vector<B>& vis, const G& x, K u, F fn) {
if (vis[u]) return;
vector<K> stack;
stack.reserve(x.span());
stack.push_back(u);
do {
K u = stack.back(); stack.pop_back();
vis[u] = B(1); fn(u);
x.forEachEdgeKey(u, [&](K v) {
if (!vis[v]) stack.push_back(v);
});
} while (!stack.empty());
}
template <class B=bool, class G, class K, class F>
inline vector<B> dfsVisitedForEachNonRecursiveW(const G& x, K u, F fn) {
vector<B> vis(x.span());
dfsVisitedForEachNonRecursiveW(vis, x, u, fn);
return vis;
}

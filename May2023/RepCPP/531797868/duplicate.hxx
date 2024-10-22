#pragma once





template <class H, class G>
inline void duplicateW(H& a, const G& x) {
a.reserve(x.span());
x.forEachVertex([&](auto u, auto d) { a.addVertex(u, d); });
x.forEachVertexKey([&](auto u) {
x.forEachEdge(u, [&](auto v, auto w) { a.addEdge(u, v, w); });
});
a.update();
}
template <class G>
inline auto duplicate(const G& x) {
G a = x;
return a;
}

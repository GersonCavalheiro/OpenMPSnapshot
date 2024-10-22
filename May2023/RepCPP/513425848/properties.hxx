#pragma once
#include <tuple>
#include "vertices.hxx"

using std::make_tuple;





template <class G, class F>
auto degreesDo(const G& x, F fn) {
using K = typename G::key_type;
auto  a = createContainer(x, K());
x.forEachVertexKey([&](auto u) { a[u] = x.degree(u); fn(u, x.degree(u)); });
return a;
}
template <class G>
inline auto degrees(const G& x) {
auto fn = [](auto u, auto d) {};
return degreesDo(x, fn);
}


template <class G>
auto minDegree(const G& x) {
using K = typename G::key_type;
K min = x.order();
x.forEachVertexKey([&](auto u) {
auto d = x.degree(u);
if (d<min) min = d;
});
return min;
}

template <class G>
auto maxDegree(const G& x) {
using K = typename G::key_type;
K max = 0;
x.forEachVertexKey([&](auto u) {
auto d = x.degree(u);
if (d>max) max = d;
});
return max;
}

template <class G>
inline float avgDegree(const G& x) {
size_t N = x.order();
return N==0? 0 : x.size()/float(N);
}


template <class G>
auto minMaxAvgDegree(const G& x) {
using K = typename G::key_type;
K min = x.order();
K max = 0;
x.forEachVertexKey([&](auto u) {
auto d = x.degree(u);
if (d<min) min = d;
if (d>max) max = d;
});
return make_tuple(min, max, avgDegree(x));
}






template <class G, class K>
inline auto edgeWeight(const G& x, K u) {
using E = typename G::edge_value_type; E a = E();
x.forEachEdgeValue(u, [&](auto w) { a += w; });
return a;
}



template <class G>
auto edgeWeight(const G& x) {
using E = typename G::edge_value_type; E a = E();
x.forEachVertexKey([&](auto u) { a += edgeWeight(x, u); });
return a;
}

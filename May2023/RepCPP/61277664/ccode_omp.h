#include <stdexcept>
#include <sstream>
#include <string>
#include "defines.h"

void cmap_3clique(Graph &g, uint64_t &total,
std::vector<cmap8_t> &cmaps) {

std::cout << "3-clique using cmap\n";
uint64_t counter = 0;

#pragma omp parallel for schedule(dynamic, 1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
uint64_t local_counter = 0;
auto y0 = g.N(v0);

auto tid = omp_get_thread_num();
auto &cmap = cmaps[tid];

for (auto u : y0) {
#if USE_DAG == 0
if (u >= v0) break;
#endif
cmap.set(u, 1);
}

for (auto v1 : y0) {
#if USE_DAG == 0
if (v1 >= v0) break;
#endif
auto y1 = g.N(v1);

for (auto u : y1) {
#if USE_DAG == 0
if (u >= v1) break;
#endif
local_counter += (cmap.get(u) == 1);
}
}

for (auto u : y0) {
#if USE_DAG == 0
if (u >= v0) break;
#endif
cmap.set(u, 0);
}

counter += local_counter;
}

total = counter;
}

void cmap_4clique(Graph &g, uint64_t &total,
std::vector<cmap8_t> &cmaps) {
std::cout << "4-clique using cmap\n";
uint64_t counter = 0;
#ifdef STATS_EDGES_VISITED
uint64_t edges = 0;
#endif
#pragma omp parallel for schedule(dynamic, 1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
uint64_t local_counter = 0;
auto y0 = g.N(v0);
auto tid = omp_get_thread_num();
auto &cmap = cmaps[tid];

for (auto u : y0) {
#if USE_DAG == 0
if (u >= v0) break;
#endif
cmap.set(u, 1);
}

for (auto v1 : y0) {
#if USE_DAG == 0
if (v1 >= v0) break;
#endif
auto y1 = g.N(v1);

VertexSet y0y1;
y0y1.clear();
for (auto u : y1) {
#if USE_DAG == 0
if (u >= v1) break;
#endif
if (cmap.get(u) == 1) {
cmap.set(u, 2);
y0y1.add(u);
}
}

for (auto v2 : y0y1) {
for (auto v3 : g.N(v2)) {
#if USE_DAG == 0
if (v3 >= v2) break;
#endif
local_counter += (cmap.get(v3) == 2);
}
}
for (auto u : y0y1) cmap.set(u, 1);
}

for (auto u : y0) {
#if USE_DAG == 0
if (u >= v0) break;
#endif
cmap.set(u, 0);
}
counter += local_counter;
}
total = counter;
}

void cmap_5clique(Graph &g, uint64_t &total,
std::vector<cmap8_t> &cmaps) {
std::cout << "5-clique using cmap\n";
uint64_t counter = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
auto y0 = g.N(v0);
uint64_t local_counter = 0;
#if 0
for (auto v1 : y0) {
auto y1 = g.N(v1);
auto y0y1 = y0 & y1;
for (auto v2 : y0y1) {
auto y2 = g.N(v2);
auto y0y1y2 = y0y1 & y2;
for (auto v3 : y0y1y2)
local_counter += intersection_num(y0y1y2, g.N(v3));
}
}
#else
auto tid = omp_get_thread_num();
auto &cmap = cmaps[tid];
for (auto u : y0) cmap.set(u, 1);
for (auto v1 : y0) {
auto y1 = g.N(v1);
VertexSet y0y1;
y0y1.clear();
for (auto u : y1) {
if (cmap.get(u) == 1) {
cmap.set(u, 2);
y0y1.add(u);
}
}
for (auto v2 : y0y1) {
VertexSet y0y1y2;
y0y1y2.clear();
for (auto u : g.N(v2)) {
if (cmap.get(u) == 2) {
cmap.set(u, 3);
y0y1y2.add(u);
}
}
for (auto v3 : y0y1y2) {
for (auto v4 : g.N(v3)) {
local_counter += (cmap.get(v4) == 3);
}
}
for (auto u : y0y1y2) cmap.set(u, 2);
}
for (auto u : y0y1) cmap.set(u, 1);
}
for (auto u : y0) cmap.set(u, 0);
#endif
counter += local_counter;
}
total = counter;
}

void cmap_4clique(Graph &g, uint64_t &total,
std::vector<cmap8_t> &cmaps,
std::vector<EmbList> &emb_lists) {
std::cout << "4-clique using cmap and embedding list\n";
uint64_t counter = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0 ++) {
auto tid = omp_get_thread_num();
auto &cmap = cmaps[tid];
auto &emb_list = emb_lists[tid];
auto y0 = g.N(v0);
for (auto u : y0) cmap.set(u, 1);
for (auto v1 : y0) {
emb_list.set_size(2, 0);
for (auto u : g.N(v1)) {
if (cmap.get(u) == 1) {
cmap.set(u, 2);
emb_list.add_emb(2, u);
}
}
for (vidType emb_id = 0; emb_id < emb_list.size(2); emb_id++) {
auto v2 = emb_list.get_vertex(2, emb_id);
for (auto v3 : g.N(v2)) {
counter += (cmap.get(v3) == 2);
}
}
for (vidType emb_id = 0; emb_id < emb_list.size(2); emb_id++) {
auto v = emb_list.get_vertex(2, emb_id);
cmap.set(v, 1);
}
}
for (auto u : y0) cmap.set(u, 0);
}
total = counter;
}

void cmap_5clique(Graph &g, uint64_t &total,
std::vector<cmap8_t> &cmaps,
std::vector<EmbList> &emb_lists) {
std::cout << "5-clique using cmap and embedding list\n";
uint64_t counter = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0 ++) {
uint64_t local_counter = 0;
auto tid = omp_get_thread_num();
auto &cmap = cmaps[tid];
auto &emb_list = emb_lists[tid];
auto y0 = g.N(v0);
for (auto u : y0) cmap.set(u, 1);
for (auto v1 : y0) {
emb_list.set_size(2, 0);
for (auto u : g.N(v1)) {
if (cmap.get(u) == 1) {
cmap.set(u, 2);
emb_list.add_emb(2, u);
}
}
for (vidType id2 = 0; id2 < emb_list.size(2); id2++) {
auto v2 = emb_list.get_vertex(2, id2);
emb_list.set_size(3, 0);
for (auto u : g.N(v2)) {
if (cmap.get(u) == 2) {
cmap.set(u, 3);
emb_list.add_emb(3, u);
}
}
for (vidType id3 = 0; id3 < emb_list.size(3); id3++) {
auto v3 = emb_list.get_vertex(3, id3);
for (auto v4 : g.N(v3)) {
local_counter += (cmap.get(v4) == 3);
}
}
for (vidType id3 = 0; id3 < emb_list.size(3); id3++) {
auto v = emb_list.get_vertex(3, id3);
cmap.set(v, 2);
}
}
for (vidType id2 = 0; id2 < emb_list.size(2); id2++) {
auto v = emb_list.get_vertex(2, id2);
cmap.set(v, 1);
}
}
for (auto u : y0) cmap.set(u, 0);
counter += local_counter;
}
total = counter;
}

void cmap_kclique(Graph &g, unsigned k, uint64_t &total,
std::vector<cmap8_t> &cmaps) {
switch (k) {
case 3:
cmap_3clique(g, total, cmaps);
break;
case 4:
cmap_4clique(g, total, cmaps);
break;
case 5:
static_assert(USE_DAG == 1, "5-clique w/o DAG not implemented yet!\n");
cmap_5clique(g, total, cmaps);
break;
default:
std::stringstream sstream;
sstream << k << "-clique not implemented with cmap yet!";
throw runtime_error{sstream.str()};
}
}


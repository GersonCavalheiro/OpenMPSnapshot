#include "cc.h"
#include "timer.h"
#include "platform_atomics.h"
#include <random>
#include <iostream>
#include <algorithm>
#include <unordered_map>

void Link(IndexT u, IndexT v, IndexT *comp) {
IndexT p1 = comp[u];
IndexT p2 = comp[v];
while (p1 != p2) {
IndexT high = p1 > p2 ? p1 : p2;
IndexT low = p1 + (p2 - high);
IndexT p_high = comp[high];
if ((p_high == low) || (p_high == high && compare_and_swap(comp[high], high, low)))
break;
p1 = comp[comp[high]];
p2 = comp[low];
}
}

void Compress(int m, IndexT *comp) {
#pragma omp parallel for schedule(static, 2048)
for (IndexT n = 0; n < m; n++) {
while (comp[n] != comp[comp[n]]) {
comp[n] = comp[comp[n]];
}
}
}

void Afforest(Graph &g, CompT *comp, int32_t neighbor_rounds = 2) {
auto m = g.V();
for (int r = 0; r < neighbor_rounds; ++r) {
#pragma omp parallel for
for (IndexT src = 0; src < m; src ++) {
for (IndexT dst : g.out_neigh(src, r)) {
Link(src, dst, comp);
break;
}
}
Compress(m, comp);
}

IndexT c = SampleFrequentElement(m, comp);

if (!g.is_directed()) {
#pragma omp parallel for schedule(dynamic, 2048)
for (IndexT u = 0; u < m; u ++) {
if (comp[u] == c) continue;
for (auto v : g.out_neigh(u, neighbor_rounds)) {
Link(u, v, comp);
}
}
} else {
#pragma omp parallel for schedule(dynamic, 2048)
for (IndexT u = 0; u < m; u ++) {
if (comp[u] == c) continue;
for (auto v : g.out_neigh(u, neighbor_rounds)) {
Link(u, v, comp);
}
for (auto v : g.in_neigh(u)) {
Link(u, v, comp);
}
}
}
Compress(m, comp);
}

void CCSolver(Graph &g, CompT *comp) {
auto m = g.V();
int num_threads = 1;
#pragma omp parallel
{
num_threads = omp_get_num_threads();
}
printf("Launching OpenMP CC solver (%d threads) ...\n", num_threads);

#pragma omp parallel for
for (int n = 0; n < m; n ++) comp[n] = n;

Timer t;
t.Start();
Afforest(g, comp);
t.Stop();

printf("\truntime [omp_afforest] = %f ms.\n", t.Millisecs());
return;
}

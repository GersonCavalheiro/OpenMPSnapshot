
#include "kcl.h"
#include <omp.h>
#include "timer.h"
#include "subgraph.h"
#define KCL_VARIANT "omp_base"

void mksub(unsigned n, unsigned core, unsigned *cd, unsigned *adj, unsigned u, subgraph* &sg, unsigned char k) {
static unsigned *old = NULL, *mynew = NULL;
#pragma omp threadprivate(mynew,old)
if (old == NULL) {
mynew = (unsigned *)malloc(n * sizeof(unsigned));
old = (unsigned *)malloc(core * sizeof(unsigned));
for (unsigned i = 0; i < n; i ++) mynew[i] = (unsigned)-1;
}
for (unsigned i = 0; i < sg->n[k-1]; i ++) sg->lab[i] = 0;
unsigned v;
unsigned j = 0;
for (unsigned i = cd[u]; i < cd[u+1]; i ++) {
v = adj[i];
mynew[v] = j;
old[j] = v;
sg->lab[j] = k-1;
sg->vertices[k-1][j] = j;
sg->d[k-1][j] = 0;
j ++;
}
sg->n[k-1] = j;
for (unsigned i = 0; i < sg->n[k-1]; i ++) {
v = old[i];
for (unsigned l = cd[v]; l < cd[v+1]; l ++) {
unsigned w = adj[l];
j = mynew[w];
if (j != (unsigned)-1) {
sg->adj[sg->core*i+sg->d[k-1][i]++] = j;
}
}
}
for (unsigned i = cd[u]; i < cd[u+1]; i ++)
mynew[adj[i]] = (unsigned)-1;
}

void kclique_thread(unsigned l, subgraph * &sg, long long *n) {
if (l == 2) {
for(unsigned i = 0; i < sg->n[2]; i++) { 
unsigned u = sg->vertices[2][i];
unsigned end = u * sg->core + sg->d[2][u];
for (unsigned j = u * sg->core; j < end; j ++) {
(*n) ++; 
}
}
return;
}
printf("TODO\n");
for(unsigned i = 0; i < sg->n[l]; i ++) {
unsigned u = sg->vertices[l][i];
sg->n[l-1] = 0;
unsigned end = u*sg->core+sg->d[l][u];
for (unsigned j = u*sg->core; j < end; j ++) {
unsigned v = sg->adj[j];
if (sg->lab[v] == l) {
sg->lab[v] = l-1;
sg->vertices[l-1][sg->n[l-1]++] = v;
sg->d[l-1][v] = 0;
}
}
for (unsigned j = 0; j < sg->n[l-1]; j ++) {
unsigned v = sg->vertices[l-1][j];
end = sg->core * v + sg->d[l][v];
for (unsigned k = sg->core * v; k < end; k ++) {
unsigned w = sg->adj[k];
if (sg->lab[w] == l-1) {
sg->d[l-1][v] ++;
}
else {
sg->adj[k--] = sg->adj[--end];
sg->adj[end] = w;
}
}
}
kclique_thread(l-1, sg, n);
for (unsigned j = 0; j < sg->n[l-1]; j ++) {
unsigned v = sg->vertices[l-1][j];
sg->lab[v] = l;
}
}
}

void KCLSolver(Graph &g, unsigned k, long long *total) {
int num_threads = 1;
#pragma omp parallel
{
num_threads = omp_get_num_threads();
}
printf("Launching OpenMP KCL solver (%d threads) ...\n", num_threads);
long long num = 0;

Timer t;
t.Start();
#pragma omp parallel reduction(+:num)
{
subgraph *sg = new subgraph;
sg->allocate(g.core, k);
#pragma omp for schedule(dynamic, 1) nowait
for (unsigned i = 0; i < g.n; i ++) {
mksub(g.n, g.core, g.cd, g.adj, i, sg, k);
kclique_thread(k-1, sg, &num);
}
}
t.Stop();

*total = num;
printf("Number of %d-cliques: %lld\n", k, num);
printf("\truntime [%s] = %f ms.\n", KCL_VARIANT, t.Millisecs());
return;
}

#include "spath.h"
void dijkstra(int **spaths, int *weights, int numV, int nodes, int pos) {
int i, j, k, current;
#pragma omp parallel shared(spaths)
{
#pragma omp for private(i,j,k,current)
for (i = 0; i < nodes; i++) {
int node = pos/numV + i;
int offset = i * numV;
bool *spt = allocate(numV * sizeof(bool)); 
for (j = 0; j < numV; j++) {
(*spaths)[j + offset] = -1;
spt[j] = false;
}
(*spaths)[offset + node] = 0;
current = node;
for (j = 0; j < numV; j++) {
for (k = 0; k < numV; k++) {
int direct = weights[(numV * current) + k]; 
if (k == current || direct == 0) {
continue;
}
int dist = (*spaths)[offset + current] + direct;
if ((*spaths)[offset + k] == -1 || dist < (*spaths)[offset + k]) {
(*spaths)[offset + k] = dist;
}
}
spt[current] = true;
int lowest = -1;
for (k = 0; k < numV; k++) {
if (!spt[k] && (*spaths)[offset + k] != -1 && (lowest == -1 || (*spaths)[offset + k] < lowest)) {
lowest = (*spaths)[offset + k];
current = k;
}
}
}
free(spt);
}
}
}

#pragma once
#include <omp.h>
#include "standard.h"

namespace dijkstra {
class openmp: public standard
{
private:
int maxThreads;
public:
openmp(int V);
~openmp();
void minDistance(int *dist, bool *sptSet, int start, int end, int *local_private_md, int *local_private_u);
void printSolution(int *dist);
void init(int *dist, bool *sptSet, int src);
void dijkstra(int **graph, int src);
void updateDist(int *dist, bool *sptSet, int **graph, int start, int end, int u);
};
}
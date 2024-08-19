#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

void pageRank(Graph g, double *solution, double damping, double convergence)
{


int numNodes = num_nodes(g);
double equal_prob = 1.0 / numNodes;
double *score_new = new double[numNodes];
bool converged = false;
double global_diff = 0.0;
double sum_no_out = 0.0;

#pragma omp parallel for reduction(+ \
: sum_no_out)
for (int i = 0; i < numNodes; ++i)
{
solution[i] = equal_prob;
if (outgoing_size(g, i) == 0)
sum_no_out += solution[i];
}

while (!converged)
{
#pragma omp parallel for
for (int vi = 0; vi < numNodes; vi++)
{
score_new[vi] = 0.0;

const Vertex *start = incoming_begin(g, vi);
const Vertex *end = incoming_end(g, vi);
for (const Vertex *v = start; v != end; v++)
score_new[vi] += solution[*v] / outgoing_size(g, *v);

score_new[vi] = (damping * score_new[vi]) + (1.0 - damping) / numNodes;
score_new[vi] += damping * sum_no_out / numNodes;
}

global_diff = 0.0;
sum_no_out = 0.0;
#pragma omp parallel for reduction(+ \
: global_diff, sum_no_out)
for (int vi = 0; vi < numNodes; vi++)
{
global_diff += abs(score_new[vi] - solution[vi]);
solution[vi] = score_new[vi];
if (outgoing_size(g, vi) == 0)
sum_no_out += solution[vi];
}

converged = (global_diff < convergence);
}

delete[] score_new;


}


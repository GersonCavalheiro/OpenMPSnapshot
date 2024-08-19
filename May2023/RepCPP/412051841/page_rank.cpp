#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

void pageRank(Graph g, double *solution, double damping, double convergence) {


int numNodes = num_nodes(g);
double equal_prob = 1.0 / numNodes;
#pragma omp parallel for
for (int i = 0; i < numNodes; i++) {
solution[i] = equal_prob;
}

double *old_solution = (double *) malloc(numNodes * sizeof(double));

double sum_of_no_outgoing, global_diff;

bool converged = false;
while (!converged) {
memcpy(old_solution, solution, numNodes * sizeof(double));

sum_of_no_outgoing = 0.0;
global_diff = 0.0;
#pragma omp parallel
{
#pragma omp for reduction (+:sum_of_no_outgoing)
for (int no_outgoing = 0; no_outgoing < numNodes; no_outgoing++) {
if (outgoing_size(g, no_outgoing) == 0)
sum_of_no_outgoing += damping * old_solution[no_outgoing] / numNodes;
}

#pragma omp for reduction (+:global_diff)
for (int vi = 0; vi < numNodes; vi++) {
const Vertex *start = incoming_begin(g, vi);
const Vertex *end = incoming_end(g, vi);
double sum = 0.0;
for (const Vertex *incoming = start; incoming != end; incoming++) {
sum += old_solution[*incoming] / outgoing_size(g, *incoming);
}
solution[vi] = (damping * sum) + (1.0 - damping) / numNodes + sum_of_no_outgoing;

global_diff += fabs(old_solution[vi] - solution[vi]);
}
}

converged = (global_diff < convergence);
}
delete old_solution;
}

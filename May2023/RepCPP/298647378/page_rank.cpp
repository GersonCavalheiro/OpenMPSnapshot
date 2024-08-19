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
double *old_solution;
old_solution = (double*)malloc(sizeof(double) * g->num_nodes);

#pragma omp parallel for
for (int i = 0; i < numNodes; ++i)
{
solution[i] = equal_prob;
}

bool converged = false;
while (!converged){
double global_diff = 0.0;
double no_outgoing_score = 0.0;
memcpy(old_solution, solution, g->num_nodes * sizeof(double));

#pragma omp parallel for reduction(+:no_outgoing_score)
for (int j = 0; j < numNodes; j++){
if (outgoing_size(g, j) == 0) no_outgoing_score += damping * old_solution[j] / numNodes;
}

#pragma omp parallel for reduction(+:global_diff)
for (int i = 0; i < numNodes; i++){
const Vertex* in_start = incoming_begin(g, i);
const Vertex* in_end = incoming_end(g, i);
double sum = 0.0;
for (const Vertex* v = in_start; v != in_end; v++){
sum += old_solution[*v] / (double)outgoing_size(g, *v);
}

sum = (damping * sum) + (1.0 - damping) / numNodes;

sum += no_outgoing_score;
solution[i] = sum;
global_diff += fabs(sum - old_solution[i]);
}
converged = global_diff < convergence;
printf("%lf \n", global_diff);
}

delete old_solution;


}

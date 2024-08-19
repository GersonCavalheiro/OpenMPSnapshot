#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "ompdist/graph.h"
#include "ompdist/graph_gen.h"
#include "ompdist/utils.h"
#include "ompdist/msr.h"
#include "config.h"
typedef struct {
int x;              
int new_x;          
} processor;
void initialize_graph(graph* g, int* xvals) {
for (int i = 0; i < g->N; i++) {
node* cur = elem_at(&g->vertices, i);
processor* p = malloc(sizeof(processor));
p->x = xvals[i];
cur->data = p;
}
}
void calculate_temporary_x(graph* g) {
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < g->N; i++) {
node* cur = elem_at(&g->vertices, i);
processor* p = cur->data;
int new_x = p->x;
for (int j = 0; j < cur->degree; j++) {
node* neighbor = *((node**) elem_at(&cur->neighbors, j));
processor* neighbor_p = neighbor->data;
if (new_x < neighbor_p->x)
new_x = neighbor_p->x;
}
p->new_x = new_x;
}
}
int propagate_temporary_x(graph* g) {
int something_changed = 0;
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < g->N; i++) {
node* cur = elem_at(&g->vertices, i);
processor* p = cur->data;
if (p->new_x != p->x)
something_changed = 1;
p->x = p->new_x;
}
return something_changed;
}
int verify_and_print_solution(graph* g) {
int disagreement = 0;
int leader = -1;
for (int i = 0; i < g->N; i++) {
node* cur = elem_at(&g->vertices, i);
processor* p = cur->data;
if (leader == -1)
leader = p->x;
if (leader != p->x) {
INFO("Node %d disagrees; wants %d as leader instead of %d\n", i, p->x, leader);
disagreement = 1;
}
}
if (!disagreement)
INFO("Correct! All nodes have agreed to have %d as the leader.\n", leader);
return disagreement;
}
int main(int argc, char* argv[]) {
int N;
int M;
int* xvals;
graph* g;
int iterate;
int iterations = 1;
if ((iterate = input_through_argv(argc, argv))) {
FILE* in = fopen(argv[2], "r");
fscanf(in, "%d\n", &N);
g = new_graph(N, 0);
xvals = malloc(N * sizeof(int));
g->M = M = read_graph(g, in);
fscanf(in, "\n");
for (int i = 0; i < N; i++)
fscanf(in, "%d", &xvals[i]);
fclose(in);
sscanf(argv[3], "%d", &iterations);
}
else {
N = 16;
M = 64;
if (argc > 1) {
sscanf(argv[1], "%d", &N);
sscanf(argv[2], "%d", &M);
}
g = generate_new_connected_graph(N, M);
xvals = malloc(N * sizeof(int));
for (int i = 0; i < N; i++)
xvals[i] = i;
}
long long duration = 0;
double total_energy = 0;
int verification;
for (int i = 0; i < iterations; i++) {
begin_timer();
init_energy_measure();
int something_changed = 1;
int num_rounds = 0;
initialize_graph(g, xvals);
while (something_changed) {
calculate_temporary_x(g);
something_changed = propagate_temporary_x(g);
num_rounds++;
}
total_energy += total_energy_used();
duration += time_elapsed();
verification = verify_and_print_solution(g);
for (int v = 0; v < g->N; v++) {
node* cur = elem_at(&g->vertices, v);
free(cur->data);
}
}
free(xvals);
if (iterate)
printf("%.2lf %.2lf\n", ((double) duration) / iterations, total_energy / iterations);
return verification;
}

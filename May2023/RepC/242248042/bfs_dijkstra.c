#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "ompdist/utils.h"
#include "ompdist/vector.h"
#include "ompdist/graph.h"
#include "ompdist/graph_gen.h"
#include "ompdist/msr.h"
#include "config.h"
#define START 1
#define JOIN  2
int ROOT;
typedef struct {
int parent_label;
int phase_discovered;
} payload;
void initialize_graph(graph* g) {
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < g->N; i++) {
node* cur = elem_at(&g->vertices, i);
payload* data = malloc(sizeof(payload));
data->parent_label = -1;
data->phase_discovered = -1;
cur->data = data;
}
node* root = elem_at(&g->vertices, ROOT);
payload* data = root->data;
data->phase_discovered = 0;
}
int broadcast_start(graph* g, int p) {
int nobody_was_discovered = 1;
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < g->N; i++) {
node* cur = elem_at(&g->vertices, i);
payload* data = cur->data;
if (data->phase_discovered == p) {
for (int j = 0; j < cur->degree; j++) {
node* neighbor = *((node**) elem_at(&cur->neighbors, j));
payload* neighbor_data = neighbor->data;
if (neighbor_data->phase_discovered < 0) {
neighbor_data->phase_discovered = p+1;
neighbor_data->parent_label = cur->label;
nobody_was_discovered = 0;
}
}
}
}
return nobody_was_discovered;
}
void print_solution(graph* g) {
int max_distance = 0;
for (int i = 0; i < g->N; i++) {
node* cur = elem_at(&g->vertices, i);
payload* data = cur->data;
if (data->phase_discovered > max_distance)
max_distance = data->phase_discovered;
INFO("node_%d: parent = %d, dist = %d\n", cur->label, data->parent_label, data->phase_discovered);
}
INFO("max_distance = %d\n", max_distance);
}
int main(int argc, char* argv[]) {
int N;
int M;
graph* g;
int iterate;
int iterations = 1;
if ((iterate = input_through_argv(argc, argv))) {
FILE* in = fopen(argv[2], "r");
fscanf(in, "%d\n", &N);
g = new_graph(N, 0);
fscanf(in, "%d\n", &ROOT);
g->M = M = read_graph(g, in);
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
ROOT = 0;
}
long long duration = 0;
double total_energy = 0;
for (int i = 0; i < iterations; i++) {
begin_timer();
init_energy_measure();
int p = 0;
int nobody_was_discovered = 0;
initialize_graph(g);
while (!nobody_was_discovered) {
nobody_was_discovered = broadcast_start(g, p);
p++;
}
total_energy += total_energy_used();
duration += time_elapsed();
}
if (iterate)
printf("%.2lf %.2lf\n", ((double) duration) / iterations, total_energy / iterations);
return 0;
}

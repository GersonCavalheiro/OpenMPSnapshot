#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_DISTANCE -1
#define NOT_VISITED_VERTEX 0
#define THRESHOLD 0.25
#define DYNAMIC_CHUNK 1024

void vertex_set_clear(vertex_set *list) {
list->count = 0;
}

void vertex_set_init(vertex_set *list, int count) {
list->max_vertices = count;
list->vertices = (int *) calloc(list->max_vertices, sizeof(int));
vertex_set_clear(list);
}

void top_down_step(
Graph &g,
vertex_set *frontier,
int *distances,
int &current_frontier) {
int num_of_frontiers = 0;

#pragma omp parallel for reduction (+:num_of_frontiers) schedule (dynamic, DYNAMIC_CHUNK)
for (int node = 0; node < g->num_nodes; node++) {
if (frontier->vertices[node] == current_frontier) {
const int start_edge = g->outgoing_starts[node];
const int end_edge = (node == g->num_nodes - 1)
? g->num_edges
: g->outgoing_starts[node + 1];

for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
const int outgoing = g->outgoing_edges[neighbor];

if (frontier->vertices[outgoing] == NOT_VISITED_VERTEX) {
num_of_frontiers++;
distances[outgoing] = distances[node] + 1;
frontier->vertices[outgoing] = current_frontier + 1;
}
}
}
}
frontier->count = num_of_frontiers;
}

void bfs_top_down(Graph graph, solution *sol) {

vertex_set list;
vertex_set_init(&list, graph->num_nodes);

vertex_set *frontier = &list;

memset(sol->distances, NOT_VISITED_DISTANCE, graph->num_nodes * sizeof(int));

int num_of_hops = 1;
frontier->vertices[frontier->count++] = num_of_hops;
sol->distances[ROOT_NODE_ID] = 0;

while (frontier->count != 0) {

#ifdef VERBOSE
double start_time = CycleTimer::currentSeconds();
#endif

vertex_set_clear(frontier);

top_down_step(graph, frontier, sol->distances, num_of_hops);

#ifdef VERBOSE
double end_time = CycleTimer::currentSeconds();
printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

num_of_hops++;
}
}

void bottom_up_step(
Graph &g,
vertex_set *frontier,
int *distances,
int &num_of_hops) {
int num_of_frontiers = 0;

#pragma omp parallel for reduction (+:num_of_frontiers) schedule (dynamic, DYNAMIC_CHUNK)
for (int node = 0; node < g->num_nodes; node++) {
if (frontier->vertices[node] == NOT_VISITED_VERTEX) {
const int start_edge = g->incoming_starts[node];
const int end_edge = (node == g->num_nodes - 1)
? g->num_edges
: g->incoming_starts[node + 1];

for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
const int incoming = g->incoming_edges[neighbor];

if (frontier->vertices[incoming] == num_of_hops) {
num_of_frontiers++;
distances[node] = distances[incoming] + 1;
frontier->vertices[node] = num_of_hops + 1;
break;
}
}
}
}
frontier->count = num_of_frontiers;
}

void bfs_bottom_up(Graph graph, solution *sol) {
vertex_set list;
vertex_set_init(&list, graph->num_nodes);

vertex_set *frontier = &list;

memset(sol->distances, NOT_VISITED_DISTANCE, graph->num_nodes * sizeof(int));

int num_of_hops = 1;
frontier->vertices[frontier->count++] = num_of_hops;
sol->distances[ROOT_NODE_ID] = 0;

while (frontier->count != 0) {

#ifdef VERBOSE
double start_time = CycleTimer::currentSeconds();
#endif

vertex_set_clear(frontier);

bottom_up_step(graph, frontier, sol->distances, num_of_hops);

#ifdef VERBOSE
double end_time = CycleTimer::currentSeconds();
printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

num_of_hops++;
}
}

void bfs_hybrid(Graph graph, solution *sol) {

vertex_set list;
vertex_set_init(&list, graph->num_nodes);

vertex_set *frontier = &list;

memset(sol->distances, NOT_VISITED_DISTANCE, graph->num_nodes * sizeof(int));

int num_of_hops = 1;
frontier->vertices[frontier->count++] = num_of_hops;
sol->distances[ROOT_NODE_ID] = 0;

const double threshold_count = graph->num_nodes * THRESHOLD;

while (frontier->count != 0) {

#ifdef VERBOSE
double start_time = CycleTimer::currentSeconds();
#endif

if (frontier->count > threshold_count) {
vertex_set_clear(frontier);
bottom_up_step(graph, frontier, sol->distances, num_of_hops);
} else {
vertex_set_clear(frontier);
top_down_step(graph, frontier, sol->distances, num_of_hops);
}

#ifdef VERBOSE
double end_time = CycleTimer::currentSeconds();
printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

num_of_hops++;
}
}

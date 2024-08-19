#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
list->max_vertices = count;
list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
vertex_set_clear(list);
}

void top_down_step(
Graph g,
vertex_set *frontier,
vertex_set *new_frontier,
int *distances)
{
#pragma omp parallel
{
int local_count = 0;
Vertex *local_frontier = new Vertex[g->num_nodes];

#pragma omp for
for (int i = 0; i < frontier->count; i++)
{
int node = frontier->vertices[i];

int start_edge = g->outgoing_starts[node];
int end_edge = (node == g->num_nodes - 1)
? g->num_edges
: g->outgoing_starts[node + 1];

for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
{
int outgoing = g->outgoing_edges[neighbor];

if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1))
local_frontier[local_count++] = outgoing;
}
}
#pragma omp critical
{
memcpy(new_frontier->vertices + new_frontier->count, local_frontier, sizeof(int) * local_count);
new_frontier->count += local_count;
}

delete[] local_frontier;
}
}

void bfs_top_down(Graph graph, solution *sol)
{
vertex_set list1;
vertex_set list2;
vertex_set_init(&list1, graph->num_nodes);
vertex_set_init(&list2, graph->num_nodes);

vertex_set *frontier = &list1;
vertex_set *new_frontier = &list2;

#pragma omp parallel for
for (int i = 0; i < graph->num_nodes; i++)
sol->distances[i] = NOT_VISITED_MARKER;

frontier->vertices[frontier->count++] = ROOT_NODE_ID;
sol->distances[ROOT_NODE_ID] = 0;

while (frontier->count != 0)
{

#ifdef VERBOSE
double start_time = CycleTimer::currentSeconds();
#endif
vertex_set_clear(new_frontier);

top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
double end_time = CycleTimer::currentSeconds();
printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

#pragma omp single
{
vertex_set *tmp = frontier;
frontier = new_frontier;
new_frontier = tmp;
}
}
}

void bottom_up_step(
Graph g,
vertex_set *frontier,
vertex_set *new_frontier,
int *distances)
{

int cur_distance = distances[frontier->vertices[0]]; 

#pragma omp parallel
{
int local_count = 0;
int *local_frontier = new Vertex[g->num_nodes];

#pragma omp for
for (int i = 0; i < g->num_nodes; i++)
{
if (distances[i] == NOT_VISITED_MARKER)
{
int start_edge = g->incoming_starts[i];
int end_edge = (i == g->num_nodes - 1)
? g->num_edges
: g->incoming_starts[i + 1];

for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
{
int incoming = g->incoming_edges[neighbor];
if (distances[incoming] == cur_distance)
{
distances[i] = distances[incoming] + 1;
local_frontier[local_count++] = i;
break;
}
}
}
}
#pragma omp critical
{
memcpy(new_frontier->vertices + new_frontier->count, local_frontier, sizeof(int) * local_count);
new_frontier->count += local_count;
}

delete[] local_frontier;
}
}

void bfs_bottom_up(Graph graph, solution *sol)
{
vertex_set list1;
vertex_set list2;
vertex_set_init(&list1, graph->num_nodes);
vertex_set_init(&list2, graph->num_nodes);

vertex_set *frontier = &list1;
vertex_set *new_frontier = &list2;

#pragma omp parallel for
for (int i = 0; i < graph->num_nodes; i++)
sol->distances[i] = NOT_VISITED_MARKER;

frontier->vertices[frontier->count++] = ROOT_NODE_ID;
sol->distances[ROOT_NODE_ID] = 0;

while (frontier->count != 0)
{

#ifdef VERBOSE
double start_time = CycleTimer::currentSeconds();
#endif
vertex_set_clear(new_frontier);

bottom_up_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
double end_time = CycleTimer::currentSeconds();
printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

#pragma omp single
{
vertex_set *tmp = frontier;
frontier = new_frontier;
new_frontier = tmp;
}
}
}

void bfs_hybrid(Graph graph, solution *sol)
{
vertex_set list1;
vertex_set list2;
vertex_set_init(&list1, graph->num_nodes);
vertex_set_init(&list2, graph->num_nodes);

vertex_set *frontier = &list1;
vertex_set *new_frontier = &list2;

#pragma omp parallel for
for (int i = 0; i < graph->num_nodes; i++)
sol->distances[i] = NOT_VISITED_MARKER;

frontier->vertices[frontier->count++] = ROOT_NODE_ID;
sol->distances[ROOT_NODE_ID] = 0;

while (frontier->count != 0)
{

#ifdef VERBOSE
double start_time = CycleTimer::currentSeconds();
#endif
vertex_set_clear(new_frontier);

int num_unvisited = 0;

if (frontier->count < (graph->num_nodes / 24))
{
top_down_step(graph, frontier, new_frontier, sol->distances);
}
else
{
bottom_up_step(graph, frontier, new_frontier, sol->distances);
}

#ifdef VERBOSE
double end_time = CycleTimer::currentSeconds();
printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

#pragma omp single
{
vertex_set *tmp = frontier;
frontier = new_frontier;
new_frontier = tmp;
}
}
}


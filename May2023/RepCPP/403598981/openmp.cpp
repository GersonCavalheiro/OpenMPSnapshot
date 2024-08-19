#include "openmp.h"
using namespace dijkstra;

openmp::openmp(int V): standard(V) {}

openmp::~openmp() {
}

void openmp::minDistance(int *dist, bool *sptSet, int start, int end, int *local_private_md, int *local_private_u)
{
*local_private_md = INT_MAX;
*local_private_u = -1;
int i, min_index;
for (i = start; i <= end; i++)
if (sptSet[i] == false && dist[i] <= *local_private_md) {
*local_private_md = dist[i], *local_private_u = i;
}
}

void openmp::printSolution(int *dist)
{
printf("Vertex Distance from Source\n");
for (int i = 0; i < V; i++)
printf("%d tt %d\n", i, dist[i]);
}


void openmp::init(int *dist, bool *sptSet, int src) {
int i;
for (i = 0; i < V; i++) {
dist[i] = INT_MAX, sptSet[i] = false;
}

dist[src] = 0;
}

void openmp::updateDist(int *dist, bool *sptSet, int **graph, int start, int end, int u) {
int i;
for (i = start; i <= end; i++)

if (!sptSet[i] && graph[u][i] && dist[u] != INT_MAX && dist[u] + graph[u][i] < dist[i])
dist[i] = dist[u] + graph[u][i];
}

void openmp::dijkstra(int **graph, int src)
{
int *dist = new int[V]; 

bool *sptSet = new bool[V]; 

init(dist, sptSet, src);
int nth, u, md, local_id, local_start, local_end, local_step, local_u, local_md;

# pragma omp parallel private ( local_id, local_start, local_end, local_step, local_u, local_md ) shared (u, md, V, sptSet, dist, graph )
{
local_id = omp_get_thread_num ();
nth = omp_get_num_threads (); 
local_start = ( local_id * V ) / nth;
local_end  = (( local_id + 1 ) * V ) / nth - 1;



for (local_step = 0; local_step < V - 1; local_step++) {
# pragma omp single 
{
md = INT_MAX;
u = -1; 
}
minDistance(dist, sptSet, local_start, local_end, &local_md, &local_u);

# pragma omp critical
{
if ( local_md < md )  
{
md = local_md;
u = local_u;
}
}
# pragma omp barrier
# pragma omp single 
{
if ( u != - 1 )
{
sptSet[u] = true;
}
}

# pragma omp barrier
if ( u != -1 )
{
updateDist( dist, sptSet, graph, local_start, local_end, u );
}
#pragma omp barrier
}


}
delete[] dist;
delete[] sptSet;
}


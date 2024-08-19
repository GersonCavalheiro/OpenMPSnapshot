#include <stdio.h>
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include "../utils/utils_sort.h"
#ifndef SIZE
#define SIZE 200
#endif
#ifndef OMP_THREADS
#define OMP_THREADS 2
#endif
int GRAPH[SIZE][SIZE];
int DIST[SIZE];
bool VISITED[SIZE];
int GRAPH_ORIG[SIZE][SIZE];
int DIST_ORIG[SIZE];
bool VISITED_ORIG[SIZE];
FILE *fp;
void init_random_graph()
{
for (int i = 0; i < SIZE; i++)
{
for (int j = i; j < SIZE; j++)
{
if (i - j == 0 || rand() % 5 == 0)
{ 
GRAPH[j][i] = GRAPH[i][j] = 0;
GRAPH_ORIG[j][i] = GRAPH_ORIG[i][j] = 0;
}
else
{
int r = rand() % 100;
GRAPH[j][i] = GRAPH[i][j] = r; 
GRAPH_ORIG[j][i] = GRAPH_ORIG[i][j] = r; 
}
}
}
}
int minDistance(int const dist[], bool const prev[])
{
int min = INT_MAX;
int min_index;
for (int v = 0; v < SIZE; v++)
{
if (!prev[v] && dist[v] <= min)
{
min = dist[v];
min_index = v;
}
}
return min_index;
}
void dijkstra(int *dist, int graph[SIZE][SIZE], bool *visited)
{
for (int i = 0; i < SIZE; i++)
{
dist[i] = INT_MAX;
visited[i] = false;
}
dist[0] = 0;
for (int count = 0; count < SIZE - 1; count++)
{
int u = minDistance(dist, visited);
visited[u] = true;
#pragma omp parallel for schedule(static) default(none) shared(dist, visited, graph,u) num_threads(OMP_THREADS)
for (int v = 0; v < SIZE; v++)
{
if (!visited[v] && graph[u][v] && dist[u] != INT_MAX && dist[u] + graph[u][v] < dist[v])
{
dist[v] += graph[u][v];
}
}
}
}
void sequential_dijkstra(int *dist, int graph[SIZE][SIZE], bool *visited)
{
for (int i = 0; i < SIZE; i++)
{
dist[i] = INT_MAX;
visited[i] = false;
}
dist[0] = 0;
for (int count = 0; count < SIZE - 1; count++)
{
int u = minDistance(dist, visited);
visited[u] = true;
#pragma omp parallel for schedule(static) default(none) shared(dist, visited, graph,u)
for (int v = 0; v < SIZE; v++)
{
if (!visited[v] && graph[u][v] && dist[u] != INT_MAX && dist[u] + graph[u][v] < dist[v])
{
dist[v] += graph[u][v];
}
}
}
}
int main()
{
init_random_graph();
sequential_dijkstra(DIST_ORIG,GRAPH_ORIG,VISITED_ORIG);
struct timeval begin, end;
gettimeofday(&begin, 0);
dijkstra(DIST, GRAPH, VISITED);
gettimeofday(&end, 0);
long seconds = end.tv_sec - begin.tv_sec;
long microseconds = end.tv_usec - begin.tv_usec;
double time_spent = seconds + microseconds * 1e-6;
int status = is_equal(DIST,DIST_ORIG, SIZE);
fp = fopen("log/c_std.csv", "a");
fprintf(fp, "OMP Dijkstra, %d, %d, %f, %d\n", SIZE, OMP_THREADS, time_spent, status);
fclose(fp);
return 0;
}
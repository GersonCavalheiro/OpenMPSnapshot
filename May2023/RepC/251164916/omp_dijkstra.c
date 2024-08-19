#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <omp.h>
#include <assert.h>
#include <time.h>
int minDistance(long dist[], bool sptSet[], long V)
{
int min = INT_MAX, min_index;
for (int v = 0; v < V; v++)
{
if ((sptSet[v] == false) && (dist[v] <= min))
{
min = dist[v], min_index = v;
}
}
return min_index;
}
void dijkstra(long src, long V, long **graph, long *dist)
{
bool sptSet[V];
for (int i = 0; i < V; i++)
{
dist[i] = INT_MAX, sptSet[i] = false;
}
dist[src] = 0;
for (int count = 0; count < V - 1; count++)
{
int u = minDistance(dist, sptSet, V);
sptSet[u] = true;
for (int v = 0; v < V; v++)
{
if (!sptSet[v] && graph[u][v] && (dist[u] != INT_MAX) && (dist[u] + graph[u][v] < dist[v]))
{
dist[v] = dist[u] + graph[u][v];
}
}
}
}
void solution(long **matrix, long **newmatrix, long nodes);
int main(int argc, char *argv[])
{
srand(13517020);
if (argc < 3)
{
fprintf(stderr, "error: missing command line arguments\n");
exit(1);
}
else
{
long nodes = atoi(argv[1]);
int thread_count = strtol(argv[2], NULL, 10);
long **matrix = (long **)malloc(nodes * sizeof(long *));
for (int i = 0; i < nodes; i++)
{
matrix[i] = (long *)malloc(nodes * sizeof(long));
}
assert(matrix != NULL);
for (int i = 0; i < nodes; i++)
{
for (int j = 0; j < nodes; j++)
{
if (i == j)
{
matrix[i][j] = 0;
}
else
{
matrix[i][j] = rand();
}
}
}
long **newmatrix = (long **)malloc(nodes * sizeof(long *));
for (int i = 0; i < nodes; i++)
{
newmatrix[i] = (long *)malloc(nodes * sizeof(long));
}
#pragma omp parallel num_threads(thread_count)
solution(matrix, newmatrix, nodes);
for (int i = 0; i < nodes; i++)
{
free(matrix[i]);
}
free(matrix);
for (int i = 0; i < nodes; i++)
{
free(newmatrix[i]);
}
free(newmatrix);
}
return 0;
}
void solution(long **matrix, long **newmatrix, long nodes)
{
clock_t begin = clock();
int numtasks, rank = 1;
rank = omp_get_thread_num();
numtasks = omp_get_num_threads();
for (int i = rank; i < nodes; i += numtasks)
{
dijkstra(i, nodes, matrix, newmatrix[i]);
}
#pragma omp barrier
if (rank == 0)
{
clock_t end = clock();
FILE *fp;
fp = fopen("old_matrix.txt", "w");
fprintf(fp, "Old matrix:\n");
for (int i = 0; i < nodes; i++)
{
for (int j = 0; j < nodes; j++)
{
fprintf(fp, "%ld ", matrix[i][j]);
}
fprintf(fp, "\n");
}
fclose(fp);
fp = fopen("result.txt", "w");
fprintf(fp, "New matrix:\n");
for (int i = 0; i < nodes; i++)
{
for (int j = 0; j < nodes; j++)
{
fprintf(fp, "%ld ", newmatrix[i][j]);
}
fprintf(fp, "\n");
}
fprintf(fp, "Solution found in: %.3f microseconds\n", ((double)(end - begin) / CLOCKS_PER_SEC) * 1000000);
printf("Solution found in: %.3f microseconds\n", ((double)(end - begin) / CLOCKS_PER_SEC) * 1000000);
fclose(fp);
}
}
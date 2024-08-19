#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
typedef struct
{
int x, y;
} Point;
#define ITERATIONS 10000
int getDistance(Point p1, Point p2)
{
return sqrt((p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y));
}
int main(int argc, char *argv[])
{
int i, j;
int iter;
int min;
int idx;
int sumX, sumY, n;
int size = 2;
int numOfPoints = 0;
Point *points = malloc(size * sizeof(Point));
int **distance;
int *assignedCluster;
FILE *file;
int clusters = 0;
Point *clusterPoints;
#pragma omp parallel shared(clusters)
{
#pragma omp single
clusters = omp_get_num_threads();
}
file = fopen("./points.txt", "r");
while (fscanf(file, "%d %d", &points[numOfPoints].x, &points[numOfPoints].y) == 2)
{
numOfPoints++;
if (numOfPoints > size)
{
size *= 2;
points = realloc(points, size * sizeof(Point));
}
}
clusterPoints = malloc(clusters * sizeof(Point));
assignedCluster = malloc(numOfPoints * sizeof(int));
distance = malloc(clusters * sizeof(int *));
for (i = 0; i < clusters; i++)
{
distance[i] = malloc(numOfPoints * sizeof(int));
}
for (i = 0; i < clusters; i++)
{
clusterPoints[i].x = rand();
clusterPoints[i].y = rand();
}
for (iter = 0; iter < ITERATIONS; iter++)
{
#pragma omp parallel shared(points, numOfPoints, clusters, clusterPoints, distance) private(i, j)
{
#pragma omp for schedule(static)
for (i = 0; i < clusters; i++)
{
for (j = 0; j < numOfPoints; j++)
{
distance[i][j] = getDistance(points[j], clusterPoints[i]);
}
}
}
#pragma omp parallel shared(numOfPoints, clusters, distance, assignedCluster) private(i, j, min, idx)
{
#pragma omp for schedule(static)
for (i = 0; i < numOfPoints; i++)
{
min = __INT_MAX__;
idx = -1;
for (j = 0; j < clusters; j++)
{
if (distance[j][i] < min)
{
min = distance[j][i];
idx = j;
}
}
assignedCluster[i] = idx;
}
}
#pragma omp parallel shared(points, numOfPoints, clusters, clusterPoints, assignedCluster) private(i, j, n, sumX, sumY)
{
#pragma omp for schedule(static)
for (i = 0; i < clusters; i++)
{
n = 0;
sumX = 0;
sumY = 0;
for (j = 0; j < numOfPoints; j++)
{
if (assignedCluster[j] == i)
{
n++;
sumX += points[j].x;
sumY += points[j].y;
}
}
if (n != 0)
{
clusterPoints[i].x = sumX / n;
clusterPoints[i].y = sumY / n;
}
}
}
}
for (i = 0; i < clusters; i++)
{
printf("Cluster %d:\n", i + 1);
for (j = 0; j < numOfPoints; j++)
{
if (assignedCluster[j] == i)
{
printf("(%d, %d)\n", points[j].x, points[j].y);
}
}
printf("\n");
}
}

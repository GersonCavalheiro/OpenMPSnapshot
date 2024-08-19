#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <math.h>
#include <float.h>

#define THREADS 3

struct Point {
int x, y;
int clusterNum;
};

struct FloatPoint {
float x, y;
};

struct Cluster {
struct Point *ps;
int size;
};

int main() {

int i, j, k, id, numClusters = THREADS;
struct Cluster *clusters = (struct Cluster *) malloc((numClusters) * sizeof(struct Cluster));

FILE *record;
char str[50], ch;
struct Point *points;
record = fopen("points.txt", "r");

if (NULL == record) {
printf("file can't be opened \n");
}

int count = 0, indx = 0;

while (!feof(record)) {
ch = fgetc(record);
if (ch == '\n') {
count++;
}
}
count += 1;

points = (struct Point *) malloc((count) * sizeof(struct Point));

record = fopen("points.txt", "r");

char *subtext;

while (fgets(str, 50, record) != NULL) {


subtext = strtok(str, ",");

points[indx].x = atoi(subtext + 1); 

subtext = strtok(NULL, ",");
points[indx].y = atoi(subtext);

indx++;
}
fclose(record);

struct FloatPoint centroid[numClusters];

for (i = 0; i < numClusters; ++i) {

centroid[i].x = rand() % 15;
centroid[i].y = rand() % 15;
}


float distance[numClusters][count];
int cnt = 10,flag=1;

while (cnt>0&&flag) {

#pragma omp parallel num_threads(THREADS) shared(distance) private(i, j, k)
{
id = omp_get_thread_num();

for (i = 0; i < count; ++i) {
float diffX = pow((float) points[i].x - centroid[id].x, 2);
float diffY = pow((float) points[i].y - centroid[id].y, 2.0);
distance[id][i] = sqrtf(diffX + diffY);
}
}

int cN = -1;
for (i = 0; i < count; ++i) {

float min = FLT_MAX;

for (j = 0; j < numClusters; ++j) {
if (distance[j][i] < min) {
min = distance[j][i];
cN = j;
}
}
points[i].clusterNum = cN + 1;
}

int z = 0, numOfPoints = 0;
struct Point tmp[count];

for (i = 1; i <= numClusters; ++i) {
for (j = 0; j < count; ++j) {

if (points[j].clusterNum == i) {
numOfPoints++;
tmp[z] = points[j];
z++;
}
}
clusters[i - 1].ps = (struct Point *) malloc((numOfPoints) * sizeof(struct Point));
clusters[i - 1].size = numOfPoints;

for (k = 0; k < numOfPoints; ++k) {
clusters[i - 1].ps[k] = tmp[k];
}
z = 0;
numOfPoints = 0;
}

for (i = 0; i < numClusters; ++i) {

printf("Cluster: %d \n", i + 1);

for (j = 0; j < clusters[i].size; ++j) {
printf("( %d , %d ) \n", clusters[i].ps[j].x, clusters[i].ps[j].y);
}
}
printf("--------------------- \n");

struct FloatPoint mean;
struct FloatPoint prevCentroid[numClusters];

for ( i = 0; i < numClusters ; ++i) {
prevCentroid[i] = centroid[i];
}

#pragma omp parallel num_threads(THREADS) shared(clusters) private(j, mean)
{
id = omp_get_thread_num();
mean.x = 0;
mean.y = 0;

for (j = 0; j < clusters[id].size; ++j) {
mean.x += clusters[id].ps[j].x;
mean.y += clusters[id].ps[j].y;
}

if(clusters[id].size!=0)
{
mean.x /= clusters[id].size;
mean.y /= clusters[id].size;
}

centroid[id].x = mean.x;
centroid[id].y = mean.y;
}

flag=0;
for ( i = 0; i < numClusters ; ++i) {
if(fabs(prevCentroid[i].x-centroid[i].x)>=0.01||fabs(prevCentroid[i].y-centroid[i].y)>=0.01)
flag=1;
}
cnt--;

}




}

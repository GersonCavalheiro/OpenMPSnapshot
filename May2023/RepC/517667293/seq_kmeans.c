#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "kmeans.h"
__inline static
float euclid_dist_2(int    numdims,  
float *coord1,   
float *coord2)   
{
int i;
float ans=0.0;
for (i=0; i<numdims; i++)
ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);
return(ans);
}
__inline static
int find_nearest_cluster(int     numClusters, 
int     numCoords,   
float  *object,      
float **clusters)    
{
int   index, i;
float dist, min_dist;
index    = 0;
min_dist = euclid_dist_2(numCoords, object, clusters[0]);
for (i=1; i<numClusters; i++) {
dist = euclid_dist_2(numCoords, object, clusters[i]);
if (dist < min_dist) { 
min_dist = dist;
index    = i;
}
}
return(index);
}
int seq_kmeans(float **objects,      
int     numCoords,    
int     numObjs,      
int     numClusters,  
float   threshold,    
int    *membership,   
float **clusters)     
{
int      i, j, index, loop=0;
int     *newClusterSize; 
float    delta;          
float  **newClusters;    
newClusterSize = (int*) calloc(numClusters, sizeof(int));
assert(newClusterSize != NULL);
newClusters    = (float**) malloc(numClusters *            sizeof(float*));
assert(newClusters != NULL);
newClusters[0] = (float*)  calloc(numClusters * numCoords, sizeof(float));
assert(newClusters[0] != NULL);
#pragma omp parallel
{
#pragma omp for schedule (static) nowait
for (i=0; i<numObjs; i++) membership[i] = -1;
#pragma omp for schedule (guided) nowait
for (i=1; i<numClusters; i++)
newClusters[i] = newClusters[0] + i*numCoords;
}
do {
delta = 0.0;
#pragma omp parallel
{   
#pragma omp for private(index, j) reduction(+:delta) schedule(dynamic,1)
for (i=0; i<numObjs; i++) {
index = find_nearest_cluster(numClusters, numCoords, objects[i],
clusters);
if (membership[i] != index) delta += 1.0;
membership[i] = index;
#pragma omp atomic
newClusterSize[index]++;
for (j=0; j<numCoords; j++)
#pragma omp atomic
newClusters[index][j] += objects[i][j];
}
#pragma omp for private(j) schedule(dynamic,1)
for (i=0; i<numClusters; i++) {
for (j=0; j<numCoords; j++) {
if (newClusterSize[i] > 0)
clusters[i][j] = newClusters[i][j] / newClusterSize[i];
newClusters[i][j] = 0.0;   
}
newClusterSize[i] = 0;   
}
#pragma omp single nowait
{
delta /= numObjs;
}
}
} while (delta > threshold && loop++ < 500);
free(newClusters[0]);
free(newClusters);
free(newClusterSize);
return 1;
}

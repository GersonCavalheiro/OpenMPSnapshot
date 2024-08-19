#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "kmeans.h"
inline static float euclid_dist_2(int    numdims,  
float * coord1,   
float * coord2)   
{
int i;
float ans = 0.0;
for(i=0; i<numdims; i++)
ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);
return ans;
}
inline static int find_nearest_cluster(int     numClusters, 
int     numCoords,   
float * object,      
float * clusters)    
{
int index, i;
float dist, min_dist;
index = 0;
min_dist = euclid_dist_2(numCoords, object, clusters);
for(i=1; i<numClusters; i++) {
dist = euclid_dist_2(numCoords, object, &clusters[i*numCoords]);
if (dist < min_dist) { 
min_dist = dist;
index    = i;
}
}
return index;
}
void kmeans(float * objects,          
int     numCoords,        
int     numObjs,          
int     numClusters,      
float   threshold,        
long    loop_threshold,   
int   * membership,       
float * clusters)         
{
int i, j, k;
int index, loop=0;
double timing = 0;
float delta;          
int * newClusterSize; 
float * newClusters;  
int nthreads;         
nthreads = omp_get_max_threads();
printf("OpenMP Kmeans - Reduction\t(number of threads: %d)\n", nthreads);
for (i=0; i<numObjs; i++)
membership[i] = -1;
newClusterSize = (typeof(newClusterSize)) calloc(numClusters, sizeof(*newClusterSize));
newClusters = (typeof(newClusters))  calloc(numClusters * numCoords, sizeof(*newClusters));
int * local_newClusterSize[nthreads];  
float * local_newClusters[nthreads];   
for (k=0; k<nthreads; k++)
{
local_newClusterSize[k] = (typeof(*local_newClusterSize)) calloc(numClusters, sizeof(**local_newClusterSize));
local_newClusters[k] = (typeof(*local_newClusters)) calloc(numClusters * numCoords, sizeof(**local_newClusters));
}
timing = wtime();
do {
for (i=0; i<numClusters; i++) {
for (j=0; j<numCoords; j++)
newClusters[i*numCoords + j] = 0.0;
newClusterSize[i] = 0;
}
delta = 0.0;
for (k=0; k<nthreads; k++)
{
for (i=0; i<numClusters; i++) {
for (j=0; j<numCoords; j++)
local_newClusters[k][i*numCoords + j] = 0.0;
local_newClusterSize[k][i] = 0;
}
}
#pragma omp parallel for shared(local_newClusters, local_newClusterSize) private(i, j, k, index) reduction(+:delta)
for (i=0; i<numObjs; i++)
{
index = find_nearest_cluster(numClusters, numCoords, &objects[i*numCoords], clusters);
if (membership[i] != index)
delta += 1.0;
membership[i] = index;
k = omp_get_thread_num();
local_newClusterSize[k][index]++;
for (j=0; j<numCoords; j++)
local_newClusters[k][index*numCoords + j] += objects[i*numCoords + j];
}
for (k=0; k<nthreads; k++)
{
for (i=0; i<numClusters; i++) {
for (j=0; j<numCoords; j++)
newClusters[i*numCoords + j] += local_newClusters[k][i*numCoords + j];
newClusterSize[i] += local_newClusterSize[k][i];
}
}
for (i=0; i<numClusters; i++) {
for (j=0; j<numCoords; j++) {
if (newClusterSize[i] > 1)
clusters[i*numCoords + j] = newClusters[i*numCoords + j] / newClusterSize[i];
}
}
delta /= numObjs;
loop++;
printf("\r\tcompleted loop %d", loop);
fflush(stdout);
} while (delta > threshold && loop < loop_threshold);
timing = wtime() - timing;
printf("\n        nloops = %3d   (total = %7.4fs)  (per loop = %7.4fs)\n", loop, timing, timing/loop);
for (k=0; k<nthreads; k++)
{
free(local_newClusterSize[k]);
free(local_newClusters[k]);
}
free(newClusters);
free(newClusterSize);
}
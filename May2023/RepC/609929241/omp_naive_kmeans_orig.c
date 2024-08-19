#include <stdio.h>
#include <stdlib.h>
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
int i, j;
int index, loop=0;
double timing = 0;
float delta;          
int * newClusterSize; 
float * newClusters;  
int nthreads;         
nthreads = omp_get_max_threads();
printf("OpenMP Kmeans - Naive\t(number of threads: %d)\n", nthreads);
for (i=0; i<numObjs; i++)
membership[i] = -1;
newClusterSize = (typeof(newClusterSize)) calloc(numClusters, sizeof(*newClusterSize));
newClusters = (typeof(newClusters))  calloc(numClusters * numCoords, sizeof(*newClusters));
timing = wtime();
do {
for (i=0; i<numClusters; i++) {
for (j=0; j<numCoords; j++)
newClusters[i*numCoords + j] = 0.0;
newClusterSize[i] = 0;
}
delta = 0.0;
for (i=0; i<numObjs; i++) {
index = find_nearest_cluster(numClusters, numCoords, &objects[i*numCoords], clusters);
if (membership[i] != index)
delta += 1.0;
membership[i] = index;
newClusterSize[index]++;
for (j=0; j<numCoords; j++)
newClusters[index*numCoords + j] += objects[i*numCoords + j];
}
for (i=0; i<numClusters; i++) {
for (j=0; j<numCoords; j++) {
if (newClusterSize[i] > 0)
clusters[i*numCoords + j] = newClusters[i*numCoords + j] / newClusterSize[i];
}
}
delta /= numObjs;
loop++;
printf("\r\tcompleted loop %d", loop);
fflush(stdout);
} while (delta > threshold && loop < 10);
timing = wtime() - timing;
printf("\n        nloops = %3d   (total = %7.4fs)  (per loop = %7.4fs)\n", loop, timing, timing/loop);
free(newClusters);
free(newClusterSize);
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc/malloc.h>
#include <omp.h>
#include "omp_kmeans.h"
#define MAX_ITERATIONS 200
#define THRESHOLD 1e-4
float *centroids_global;
float *cluster_points_global;
float delta_global = THRESHOLD + 1;
double current_itr_delta_global;
double findEuclideanDistance(float *pointA, float *pointB)
{
return sqrt(pow(((double)(*(pointB + 0)) - (double)(*(pointA + 0))), 2) + pow(((double)(*(pointB + 1)) - (double)(*(pointA + 1))), 2) + pow(((double)(*(pointB + 2)) - (double)(*(pointA + 2))), 2));
}
void threadedClustering(int tID, int N, int K, int num_threads, float *data_points, float **cluster_points, float **centroids, int *num_iterations)
{
printf("Inside threadedClustering %d\n", tID);
int length_per_thread = N / num_threads;
int start = tID * length_per_thread;
int end = start + length_per_thread;
if (end > N)
{
end = N;
length_per_thread = start - end;
}
double min_distance, current_distance;
int *current_cluster_id = (int *)malloc(sizeof(int) * length_per_thread); 
int iteration_count = 0;
while ((delta_global > THRESHOLD) && (iteration_count < MAX_ITERATIONS))
{
float *current_centroid = (float *)calloc(K * 3, sizeof(float)); 
int *cluster_count = (int *)calloc(K, sizeof(int));              
for (int i = start; i < end; i++)
{
min_distance = __DBL_MAX__; 
for (int j = 0; j < K; j++)
{
current_distance = findEuclideanDistance((data_points + (i * 3)), (centroids_global + (iteration_count * K + j) * 3));
if (current_distance < min_distance)
{
min_distance = current_distance;
current_cluster_id[i - start] = j;
}
}
cluster_count[current_cluster_id[i - start]]++;
current_centroid[current_cluster_id[i - start] * 3] += data_points[(i * 3)];
current_centroid[current_cluster_id[i - start] * 3 + 1] += data_points[(i * 3) + 1];
current_centroid[current_cluster_id[i - start] * 3 + 2] += data_points[(i * 3) + 2];
}
#pragma omp critical
{
for (int i = 0; i < K; i++)
{
if (cluster_count[i] == 0)
{
continue;
}
centroids_global[((iteration_count + 1) * K + i) * 3] = current_centroid[(i * 3)] / (float)cluster_count[i];
centroids_global[((iteration_count + 1) * K + i) * 3 + 1] = current_centroid[(i * 3) + 1] / (float)cluster_count[i];
centroids_global[((iteration_count + 1) * K + i) * 3 + 2] = current_centroid[(i * 3) + 2] / (float)cluster_count[i];
}
}
double current_delta = 0.0;
for (int i = 0; i < K; i++)
{
current_delta += findEuclideanDistance((centroids_global + (iteration_count * K + i) * 3), (centroids_global + ((iteration_count - 1) * K + i) * 3));
}
#pragma omp barrier
{
if (current_delta > current_itr_delta_global)
current_itr_delta_global = current_delta;
}
#pragma omp barrier
iteration_count++;
#pragma omp master
{
delta_global = current_itr_delta_global;
current_itr_delta_global = 0.0;
(*num_iterations)++;
}
}
for (int i = start; i < end; i++)
{
cluster_points_global[i * 4] = data_points[i * 3];
cluster_points_global[i * 4 + 1] = data_points[i * 3 + 1];
cluster_points_global[i * 4 + 2] = data_points[i * 3 + 2];
cluster_points_global[i * 4 + 3] = (float)current_cluster_id[i - start];
}
}
void kmeansClusteringOmp(int N, int K, int num_threads, float *data_points, float **cluster_points, float **centroids, int *num_iterations)
{
*cluster_points = (float *)malloc(sizeof(float) * N * 4);
cluster_points_global = *cluster_points;
centroids_global = (float *)calloc(MAX_ITERATIONS * K * 3, sizeof(float));
for (int i = 0; i < K; i++)
{
centroids_global[(i * 3)] = data_points[(i * 3)];
centroids_global[(i * 3) + 1] = data_points[(i * 3) + 1];
centroids_global[(i * 3) + 2] = data_points[(i * 3) + 2];
}
omp_set_num_threads(num_threads);
#pragma omp parallel
{
int tID = omp_get_thread_num();
printf("Thread no. %d created\n", tID);
threadedClustering(tID, N, K, num_threads, data_points, cluster_points, centroids, num_iterations);
}
int centroids_size = (*num_iterations + 1) * K * 3;
*centroids = (float *)calloc(centroids_size, sizeof(float));
for (int i = 0; i < centroids_size; i++)
{
(*centroids)[i] = centroids_global[i];
}
printf("Process Completed\n");
printf("Number of iterations: %d\n", *num_iterations);
for (int i = 0; i < K; i++)
{
printf("Final Centroids:\t(%f, %f, %f)\n", *(*centroids + ((*num_iterations) * K) + (i * 3)), *(*centroids + ((*num_iterations) * K) + (i * 3) + 1), *(*centroids + ((*num_iterations) * K) + (i * 3) + 2));
}
}
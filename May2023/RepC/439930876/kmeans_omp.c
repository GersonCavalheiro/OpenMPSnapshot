#include <stdio.h>
#include "kmeans_omp.h"
#include <stdlib.h>
#define SEED 100
void update_centroids_omp(struct KMeans *model, int threads) {
int cluster_count[model->no_clusters];
float sum_cluster_columns[model->no_clusters * model->columns];
for (int i = 0; i < model->no_clusters; i++) {
cluster_count[i] = 0;
for (int c = 0; c < model->columns; c++) {
sum_cluster_columns[i * model->columns + c] = 0;
}
}
#pragma omp parallel for num_threads(threads) reduction(+:cluster_count, sum_cluster_columns)
for (int row = 0; row < model->rows; row++) {
for (int k = 0; k < model->no_clusters; k++) {
if (model->data_clusters[row] == k) {
cluster_count[k] = cluster_count[k] + 1;
for (int i = 0; i < model->columns; i++) {
sum_cluster_columns[k * model->columns + i] += model->data[row * model->columns + i];
}
break;
}
}
}
for (int k = 0; k < model->no_clusters; k++) {
for (int c = 0; c < model->columns; c++) {
if (cluster_count[k] != 0)
model->centroids[k * model->columns + c] =
sum_cluster_columns[k * model->columns + c] / (float) cluster_count[k];
}
}
}
void fit_omp(struct KMeans *model, int threads) {
if (threads <= 0) {
printf("\nInvalid number of threads. Initializing with single thread.");
threads = 1;
}
printCentroids(model);
float **cluster_centroids = malloc(sizeof(float *) * model->no_clusters);
int iteration = 300;
for (int itr = 0; itr < iteration; itr++) {
for (int k = 0; k < model->no_clusters; k++) {
cluster_centroids[k] = getClusterCentroid(model, k);
}
#pragma omp parallel for num_threads(threads)
for (int i = 0; i < model->rows; i++) {
float min_dist = eucledianDist(cluster_centroids[0], model, i);;
model->data_clusters[i] = 0;
for (int k = 1; k < model->no_clusters; k++) {
float dist = eucledianDist(cluster_centroids[k], model, i);
if (dist <= min_dist) {
min_dist = dist;
model->data_clusters[i] = k;
}
}
}
update_centroids_omp(model, threads);
}
}

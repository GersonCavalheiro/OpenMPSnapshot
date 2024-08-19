#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <assert.h>
#include <math.h>
#include "utils.h"
double** init_centroids(double** dataset, int dataset_size, int num_features, int clusters) {
double centroid_min = min_matrix(dataset, dataset_size, num_features);
double centroid_max = max_matrix(dataset, dataset_size, num_features);
double** centroids = create_matrix(clusters, num_features);
#pragma omp parallel for collapse(2)
for (int i = 0; i < clusters; i++)
for (int j = 0; j < num_features; j++)
centroids[i][j] = random_double(centroid_min, centroid_max);
return centroids;
}
int find_closest_centroid(double* data_point, double** centroids, int clusters, int num_features) {
double error;
double temp;
int index;
error = INFINITY;
for (int i = 0; i < clusters; i++) {
temp = euclidean_distance(data_point, centroids[i], num_features);
if (temp < error) {
error = temp;
index = i;
}
}
return index;
}
double** standard_kmeans(double** dataset, int dataset_size, int num_features, int clusters, int max_iterations, int* assignments, int early_stop) {
double** centroids;
double** prev_centroids;
int* counter; 
int iteration = 0;
int updates = 0; 
centroids = init_centroids(dataset, dataset_size, num_features, clusters); 
prev_centroids = create_matrix(clusters, num_features);
counter = (int*) malloc(clusters * sizeof(int));
assert(counter != NULL);
memset(counter, 0, clusters * sizeof(int));
memset(assignments, -1, dataset_size * sizeof(int));
do {
updates = 0;
memset(counter, 0, clusters * sizeof(int));
#pragma omp parallel for reduction(+: updates)
for (int i = 0; i < dataset_size; i++) {
int cluster_index = find_closest_centroid(dataset[i], prev_centroids, clusters, num_features);
if (cluster_index != assignments[i]) updates += 1; 
assignments[i] = cluster_index;
}
#pragma omp parallel for
for (int i = 0; i < clusters; i++){
int my_counter = 0;
#pragma omp parallel for reduction(+: my_counter)
for (int j = 0; j < dataset_size; j++) {
if (assignments[j] == i) my_counter += 1;
}
counter[i] = my_counter;
}
memcpy(centroids, prev_centroids, clusters * num_features * sizeof(double));
#pragma omp parallel for schedule(dynamic,1) collapse(2)
for (int k = 0; k < clusters; k++) {
for (int j = 0; j < num_features; j++) {
if (counter[k] <= 0) continue;
double sum = 0.0;
#pragma omp parallel for reduction(+: sum)
for (int i = 0; i < dataset_size; i++) {
if (assignments[i] == k) {
sum += dataset[i][j];
}
}
centroids[k][j] = sum/counter[k];
}
}
iteration++;
} while (iteration < max_iterations && ((early_stop == 1) ? ((updates > 0) ? 1 : 0) : 1));
free(counter);
return centroids;
}
double distance_sum(double* point, double** points, int size, int dim) {
double sum = 0;
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < size; i++)
sum += euclidean_distance(point, points[i], dim);
return sum;
}
double** median_kmeans(double** dataset, int dataset_size, int num_features, int clusters, int max_iterations, int* assignments, int early_stop) {
double** centroids;
double** prev_centroids;
double*** data_assignments; 
int* counter;
int iteration = 0;
int updates = 0;
int centroid_min = min_matrix(dataset, dataset_size, num_features);
int centroid_max = max_matrix(dataset, dataset_size, num_features);
data_assignments = (double***) malloc (clusters * sizeof(double**));
for (int i = 0; i < clusters; i++){
data_assignments[i] = create_matrix(dataset_size, num_features);
}
centroids = init_centroids(dataset, dataset_size, num_features, clusters);
prev_centroids = create_matrix(clusters, num_features);
counter = (int*) malloc(clusters * sizeof(int));
assert(counter != NULL);
memset(counter, 0, clusters * sizeof(int));
memset(assignments, -1, dataset_size * sizeof(int));
do {
updates = 0;
memset(counter, 0, clusters * sizeof(int));
#pragma omp parallel for reduction(+: updates)
for (int i = 0; i < dataset_size; i++) {
int cluster_index = find_closest_centroid(dataset[i], prev_centroids, clusters, num_features);
if (cluster_index != assignments[i]) updates += 1; 
assignments[i] = cluster_index;
}
#pragma omp parallel for
for (int i = 0; i < clusters; i++){
int sum = 0;
for (int j = 0; j < dataset_size; j++) {
if (assignments[j] == i) {
memcpy(data_assignments[i][sum], dataset[j], num_features * sizeof(double));
sum += 1;
}
}
counter[i] = sum;
}
memcpy(centroids, prev_centroids, clusters * num_features * sizeof(double));
for (int i = 0; i< clusters; i++){
double min_dist = INFINITY;
double my_dist;
int index;
#pragma omp parallel private(my_dist, index)
{
#pragma omp for reduction(min:min_dist)
for (int j = 0; j < counter[i]; j++) {
double distance = distance_sum(data_assignments[i][j], data_assignments[i], counter[i], num_features);
if (distance < min_dist) {
min_dist = distance;
my_dist = distance;
index = j;
}
}
#pragma critical
if (my_dist == min_dist){
memcpy(centroids[i], data_assignments[i][index], num_features * sizeof(double));
}
}
}
iteration++;
} while (iteration < max_iterations && ((early_stop == 1) ? ((updates > 0) ? 1 : 0) : 1));
free(counter);
free(data_assignments);
return centroids;
}

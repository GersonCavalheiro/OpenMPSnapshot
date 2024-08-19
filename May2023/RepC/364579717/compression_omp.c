#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "image_io.h"
#include "compression.h"
void initialise_centers(byte_t *data, double *centers, int n_pixels, int n_channels, int n_clusters);
void assign_pixels(byte_t *data, double *centers, int *labels, double *distances, int *changed, int n_pixels, int n_channels, int n_clusters);
void update_centers(byte_t *data, double *centers, int *labels, double *distances, int n_pixels, int n_channels, int n_clusters);
void update_data(byte_t *data, double *centers, int *labels, int n_pixels, int n_channels);
void kmeans_compression_omp(byte_t *data, int width, int height, int n_channels, int n_clusters, int max_iterations, int n_threads) 
{
int n_pixels = width * height;
int *labels = malloc(n_pixels * sizeof(int));
double *centers = malloc(n_clusters * n_channels * sizeof(double));
double *distances = malloc(n_pixels * sizeof(double));
omp_set_num_threads(n_threads);
double initialise_centers_time = 0;
double assign_pixels_time = 0;
double update_centers_time = 0;
double update_data_time = 0;
double start_time = omp_get_wtime();
initialise_centers(data, centers, n_pixels, n_channels, n_clusters);
initialise_centers_time += omp_get_wtime() - start_time;
int have_clusters_changed = 0;
for (int i = 0; i < max_iterations; i++) {
start_time = omp_get_wtime();
assign_pixels(data, centers, labels, distances, &have_clusters_changed, n_pixels, n_channels, n_clusters);
assign_pixels_time += omp_get_wtime() - start_time;
if (!have_clusters_changed) {
break;
}
start_time = omp_get_wtime();
update_centers(data, centers, labels, distances, n_pixels, n_channels, n_clusters);
update_centers_time += omp_get_wtime() - start_time;
}
start_time = omp_get_wtime();
update_data(data, centers, labels, n_pixels, n_channels);
update_data_time += omp_get_wtime() - start_time;
free(centers);
free(labels);
free(distances);
}
void initialise_centers(byte_t *data, double *centers, int n_pixels, int n_channels, int n_clusters)
{
for (int cluster = 0; cluster < n_clusters; cluster++) {
int random_int = rand() % n_pixels;
for (int channel = 0; channel < n_channels; channel++) {
centers[cluster * n_channels + channel] = data[random_int * n_channels + channel];
}
}
}
void assign_pixels(byte_t *data, double *centers, int *labels, double *distances, int *changed, int n_pixels, int n_channels, int n_clusters)
{
int have_clusters_changed = 0;
int pixel, cluster, channel, min_cluster, min_distance;
double tmp;
#pragma omp parallel for schedule(static) private(pixel, cluster, channel, min_cluster, min_distance, tmp)
for (pixel = 0; pixel < n_pixels; pixel++) {
double min_distance = DBL_MAX;
for (cluster = 0; cluster < n_clusters; cluster++) {
double distance = 0;
for (int channel = 0; channel < n_channels; channel++) {
tmp = (double)(data[pixel * n_channels + channel] - centers[cluster * n_channels + channel]);
distance += (tmp * tmp);
}
if (distance < min_distance) {
min_distance = distance;
min_cluster = cluster;
}
}
distances[pixel] = min_distance;
if (labels[pixel] != min_cluster) {
labels[pixel] = min_cluster;
have_clusters_changed = 1;
}
}
*changed = have_clusters_changed;
}
void update_centers(byte_t *data, double *centers, int *labels, double *distances, int n_pixels, int n_channels, int n_clusters)
{
int *counts = malloc(n_clusters * sizeof(int));
for (int cluster = 0; cluster < n_clusters; cluster++) {
for (int channel = 0; channel < n_channels; channel++) {
centers[cluster * n_channels + channel] = 0;
}
counts[cluster] = 0;
}
int pixel, min_cluster, channel;
#pragma omp parallel for private(pixel, min_cluster, channel) reduction(+:centers[:n_clusters * n_channels], counts[:n_clusters])
for (pixel = 0; pixel < n_pixels; pixel++) {
min_cluster = labels[pixel];
for (channel = 0; channel < n_channels; channel++) {
centers[min_cluster * n_channels + channel] += data[pixel * n_channels + channel];
}
counts[min_cluster] += 1;
}
for (int cluster = 0; cluster < n_clusters; cluster++) {
if (counts[cluster]) {
for (int channel = 0; channel < n_channels; channel++) {
centers[cluster * n_channels + channel] /= counts[cluster];
}
} else {
double max_distance = 0;
int farthest_pixel = 0;
#pragma omp parallel
{
double max_distance_local = max_distance;
int farthest_pixel_local = farthest_pixel;
#pragma omp for nowait
for (int pixel = 0; pixel < n_pixels; pixel++) {
if (distances[pixel] > max_distance) {
max_distance_local = distances[pixel];
farthest_pixel_local = pixel;
}
}
#pragma omp critical
{
if (max_distance_local > max_distance) {
max_distance = max_distance_local;
farthest_pixel = farthest_pixel_local;
}
}
}
for (int channel = 0; channel < n_channels; channel++) {
centers[cluster * n_channels + channel] = data[farthest_pixel * n_channels + channel];
}
distances[farthest_pixel] = 0;
}
}
free(counts);
}
void update_data(byte_t *data, double *centers, int *labels, int n_pixels, int n_channels)
{
int pixel, min_cluster, channel;
#pragma omp parallel for schedule(static) private(pixel, channel, min_cluster)
for (pixel = 0; pixel < n_pixels; pixel++) {
min_cluster = labels[pixel];
for (channel = 0; channel < n_channels; channel++) {
data[pixel * n_channels + channel] = (byte_t)round(centers[min_cluster * n_channels + channel]);
}
}
}
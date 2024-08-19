
#pragma once
#include <assert.h>
#include <float.h>
#include <math.h>
#include <memory.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
const double PI = 3.1415926535897932;
size_t threads;
double *datasets;
size_t dataSize;
size_t dimensions;
size_t clusters;
double *centers;
int *labels;
double *means;
double *variances;
double *priorities;
double *probabilities;
void loadFile(const char *const fileName) {
FILE *file = fopen(fileName, "r");
datasets = (double *)calloc(dimensions * dataSize, sizeof(double));
for (int i = 0; i < dataSize; i++) {
for (int j = 0; j < dimensions; j++) {
#pragma warning(disable: 6031)
fscanf(file, "%lf", datasets + i * dimensions + j);
}
}
fclose(file);
}
const double getDistance(const size_t sampleId, const size_t clusterId) {
double distance = 0;
int i = 0;
#pragma warning(disable: 6993)
#pragma omp parallel for num_threads(threads) reduction(+: distance)
for (i = 0; i < dimensions; i++) {
distance += pow(datasets[sampleId * dimensions + i]
- centers[clusterId * dimensions + i], 2
);
}
return sqrt(distance);
}
const double getCost(const size_t sampleId) {
labels[sampleId] = -1;
double minimum = DBL_MAX;
for (int i = 0; i < clusters; i++) {
double temp = getDistance(sampleId, i);
if (temp < minimum) {
minimum = temp;
labels[sampleId] = i;
}
}
return minimum;
}
void kMeans_clustering() {
#pragma region Initialization
const int maxInterations = 100;
const int epsilon = 0.001;
centers = (double *)calloc(clusters * dimensions, sizeof(double));
for (int i = 0; i < clusters; i++) {
const size_t start = i * dataSize * dimensions / clusters;
for (int j = 0; j < dimensions; j++) {
centers[i * dimensions + j] = datasets[start + j];
}
}
labels = (int *)calloc(dataSize, sizeof(int));
#pragma warning(disable: 6387)
memset(labels, 0, dataSize * sizeof(int));
double costs[2] = {0, 0};
int *sampleCounts = (int *)calloc(clusters, sizeof(int));
memset(sampleCounts, 0, clusters * sizeof(int));
#pragma endregion
#pragma region Estimations
for (int i = 0; i < maxInterations; i++) {
memset(sampleCounts, 0, clusters * sizeof(int));
double *nextMeans =
(double *)calloc(clusters * dimensions, sizeof(double));
if (nextMeans == NULL) {
fprintf(stderr, "[ERROR] AllocateFailedException: nextMeans\n");
exit(EXIT_FAILURE);
}
memset(nextMeans, 0, clusters * dimensions * sizeof(double));
costs[0] = costs[1];
costs[1] = 0;
int j = 0;
#pragma omp parallel for num_threads(threads)
for (j = 0; j < dataSize; j++) {
costs[1] += getCost(j);
#pragma warning(disable: 6011)
sampleCounts[labels[j]] += 1;
for (int k = 0; k < dimensions; k++) {
nextMeans[labels[j] * dimensions + k] +=
datasets[j * dimensions + k];
}
}
costs[1] /= dataSize;
for (int j = 0; j < clusters; j++) {
if (sampleCounts[j] > 0) {
for (int k = 0; k < dimensions; k++) {
centers[j * dimensions + k] =
nextMeans[j * dimensions + k] / sampleCounts[j];
}
}
}
free(nextMeans);
if (fabs(costs[0] - costs[1]) < epsilon * costs[0]) {
break;
}
}
#pragma endregion
#pragma region Cleanup
free(sampleCounts);
#pragma endregion
}
void gaussian_clustering() {
#pragma region Initialization
const int maxIterations = 100;
const double epsilon = 0.001;
priorities = (double *)calloc(clusters, sizeof(double));
memset(priorities, 0, clusters * sizeof(double));
means = (double *)calloc(clusters * dimensions, sizeof(double));
memcpy(means, centers, clusters * dimensions * sizeof(double));
variances = (double *)calloc(clusters * dimensions, sizeof(double));
memset(variances, 0, clusters * dimensions * sizeof(double));
probabilities = (double *)calloc(dataSize, sizeof(double));
memset(probabilities, 0, dataSize * sizeof(double));
double *minVariances = (double *)calloc(dimensions, sizeof(double));
memset(minVariances, 0, dimensions * sizeof(double));
int *counts = (int *)calloc(clusters, sizeof(int));
memset(counts, 0, clusters * sizeof(int));
double *overallMeans = (double *)calloc(dimensions, sizeof(double));
memset(overallMeans, 0, dimensions * sizeof(double));
int i = 0;
#pragma omp parallel for num_threads(threads)
for (i = 0; i < dataSize; i++) {
counts[labels[i]] += 1;
for (int j = 0; j < dimensions; j++) {
const double axes = datasets[i * dimensions + j]
- centers[labels[i] * dimensions + j];
variances[labels[i] * dimensions + j] += pow(axes, 2);
overallMeans[j] += datasets[i * dimensions + j];
minVariances[j] += pow(datasets[i * dimensions + j], 2);
}
}
for (int i = 0; i < dimensions; i++) {
overallMeans[i] /= dataSize;
minVariances[i] = max(
1e-10,
0.01 * (minVariances[i] / dataSize - pow(overallMeans[i], 2))
);
}
for (int i = 0; i < clusters; i++) {
priorities[i] = (double)counts[i] / dataSize;
if (priorities[i] > 0) {
for (int j = 0; j < dimensions; j++) {
variances[i * dimensions + j] /= counts[i];
variances[i * dimensions + j] = max(
variances[i * dimensions + j], minVariances[j]
);
}
} else {
memcpy(
variances + i * dimensions, minVariances,
dimensions * sizeof(double)
);
fprintf(stderr, "[WARN] Gaussian Distribution %d is nonsense!\n", i);
}
}
double costs[2] = {0, 0};
double *nextPriorities = (double *)calloc(clusters, sizeof(double));
double *nextVariances = (double *)calloc(clusters * dimensions, sizeof(double));
double *nextMeans = (double *)calloc(clusters * dimensions, sizeof(double));
#pragma endregion
#pragma region Fitting
for (int i = 0; i < maxIterations; i++) {
memset(nextPriorities, 0, clusters * sizeof(double));
memset(nextVariances, 0, clusters * dimensions * sizeof(double));
memset(nextMeans, 0, clusters * dimensions * sizeof(double));
costs[0] = costs[1];
costs[1] = 0;
int j = 0;
#pragma omp parallel for num_threads(threads)
for (j = 0; j < dataSize; j++) {
probabilities[j] = 0;
for (int k = 0; k < clusters; k++) {
double probability = 1;
for (int m = 0; m < dimensions; m++) {
probability *= 1 / sqrt(2 * PI * variances[k * dimensions + m]);
const double square = pow(
datasets[j * dimensions + m] - means[k * dimensions + m], 2
);
probability *= exp(-0.5 * square / variances[k * dimensions + m]);
}
probabilities[j] += priorities[k] * probability;
const double sampleProbability =
probability * priorities[k] / probabilities[j];
nextPriorities[k] += sampleProbability;
for (int m = 0; m < dimensions; m++) {
nextMeans[k * dimensions + m] +=
sampleProbability * datasets[j * dimensions + m];
nextVariances[k * dimensions + m] +=
sampleProbability * pow(datasets[j * dimensions + m], 2);
}
}
costs[1] += max(log10(probabilities[j]), -20);
}
costs[1] /= dataSize;
for (int j = 0; j < clusters; j++) {
priorities[j] = nextPriorities[j] / dataSize;
if (priorities[j] > 0) {
for (int k = 0; k < dimensions; k++) {
means[j * dimensions + k] =
nextMeans[j * dimensions + k] / nextPriorities[j];
if (fabs(means[j * dimensions + k]) < DBL_EPSILON) {
means[j * dimensions + k] = 0;
}
variances[j * dimensions + k] = max(
nextVariances[j * dimensions + k] / nextPriorities[j]
- pow(means[j * dimensions + k], 2),
minVariances[k]
);
}
}
}
if (fabs(costs[1] - costs[0]) < epsilon * fabs(costs[1])) {
break;
}
}
#pragma endregion
#pragma region Cleanup
free(counts);
free(overallMeans);
free(minVariances);
free(nextPriorities);
free(nextVariances);
free(nextMeans);
#pragma endregion
}
void saveFile(const char *const fileName) {
FILE *file = fopen(fileName, "wb");
fprintf(file,
"<?xml version=\"1.0\" encoding=\"GBK\"?>\n"
"<model>\n"
"<description>高斯混合模型聚类</description>\n"
"<gaussians>\n"
"<count>%lld</count>\n"
, clusters);
for (int i = 0; i < clusters; i++) {
fprintf(file,
"<gaussian>\n"
"<mean>(%lg",
centers[i * dimensions]);
for (int j = 1; j < dimensions; j++) {
fprintf(file, ", %lg", centers[i * dimensions + j]);
}
fprintf(file,
")</mean>\n"
"<variance>(%lg",
variances[i * dimensions]
);
for (int j = 1; j < dimensions; j++) {
fprintf(file, ", %lg", variances[i * dimensions + j]);
}
fprintf(file,
")</variance>\n"
"<priority>%lg</priority>"
"</gaussian>",
priorities[i]
);
}
fprintf(file,
"</gaussians>\n"
"<dataset>\n"
"<shape>(%lld, %lld)</shape>\n"
"<samples>\n"
, dataSize, dimensions);
for (int i = 0; i < dataSize; i++) {
fprintf(file,
"<sample>\n"
"<data>(%lg"
, datasets[i * dimensions]);
for (int j = 0; j < dimensions; j++) {
fprintf(file, ", %lg", datasets[i * dimensions + j]);
}
fprintf(file,
")</data>\n"
"<label>%d</label>\n"
"<probability>%lf</probability>"
"</sample>\n"
, labels[i], probabilities[i]);
}
fprintf(file,
"</samples>\n"
"</dataset>\n"
"</model>\n"
);
fclose(file);
printf("[Success] Cluster details saved!\n");
}
void cleanUp() {
free(datasets);
free(centers);
free(labels);
free(means);
free(variances);
free(priorities);
free(probabilities);
}

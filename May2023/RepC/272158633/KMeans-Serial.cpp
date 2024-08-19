
#pragma once
#include "KMeans-Serial.hpp"
#include <cassert>
#include <cmath>
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;
constexpr int DOUBLE = sizeof(double);
constexpr int INT = sizeof(int);
KMeans::KMeans(const int dimensions, const int clusters) {
this->dimensions = dimensions;
this->clusters = clusters;
this->means = new double *[this->clusters];
for (int i = 0; i < this->clusters; i++) {
this->means[i] = new double[this->dimensions];
memset(this->means[i], 0, static_cast<size_t>(this->dimensions) * DOUBLE);
}
this->initMode = InitMode::Uniformly;
this->maxInterations = 100;
this->epsilon = 0.001;
}
KMeans::~KMeans() {
for (int i = 0; i < clusters; i++) {
delete[] this->means[i];
}
delete[] this->means;
}
void KMeans::setMean(const int i, const double *point) {
memcpy(this->means[i], point, DOUBLE);
}
void KMeans::setInitMode(enum InitMode mode) {
this->initMode = mode;
}
void KMeans::setMaxIterations(const int iterations) {
this->maxInterations = iterations;
}
void KMeans::setEpsilon(const double epsilon) {
this->epsilon = epsilon;
}
const double *const KMeans::getMean(const int i) {
return this->means[i];
}
const int KMeans::getInitMode() {
return this->initMode;
}
const int KMeans::getMaxIterations() {
return this->maxInterations;
}
const double KMeans::getEpsilon() {
return this->epsilon;
}
const int *KMeans::fit_transform(
const double *const datasets, const int dataSize) {
switch (this->initMode) {
case Randomly: 
{
int interval = dataSize / this->clusters;
srand((unsigned int)time(NULL));
for (int i = 0; i < clusters; i++) {
int selection = (int)((interval - 1) * (double)rand() / RAND_MAX);
selection += interval * i;
for (int j = 0; j < this->dimensions; j++) {
this->means[i][j] = datasets[selection * this->dimensions + j];
}
}
break;
}
case Uniformly: 
{
for (int i = 0; i < this->clusters; i++) {
int selection = i * dataSize / this->clusters;
for (int j = 0; j < this->dimensions; j++) {
this->means[i][j] = datasets[selection * this->dimensions + j];
}
}
break;
}
case Manually: 
break;
}
return predict(datasets, dataSize);
}
const double *const KMeans::loadFile(
ifstream &file, const int dataSize
) {
assert(dataSize >= this->clusters);
double *dataset = new double[this->dimensions * dataSize];
string line;
for (int i = 0; i < dataSize; i++) {
getline(file, line);
istringstream spliter(line);
string cell;
for (int j = 0; j < this->dimensions; j++) {
spliter >> dataset[i * this->dimensions + j];
}
}
return dataset;
}
const int *KMeans::predict(const double *const dataset, const int dataSize) {
int *labels = new int[dataSize];
memset(labels, 0, static_cast<size_t>(dataSize) * INT);
double *costs = new double[2] {0, 0};
int *sampleCounts = new int[this->clusters];
for (int i = 0; i < this->maxInterations; i++) {
memset(sampleCounts, 0, static_cast<size_t>(this->clusters) * INT);
double **nextMeans = new double *[this->clusters];
for (int j = 0; j < this->clusters; j++) {
nextMeans[j] = new double[this->dimensions];
memset(
nextMeans[j], 0,
static_cast<size_t>(this->dimensions) * DOUBLE
);
}
costs[0] = costs[1];
costs[1] = 0;
for (int j = 0; j < dataSize; j++) {
double *sample = new double[this->dimensions];
for (int k = 0; k < this->dimensions; k++) {
sample[k] = dataset[j * this->dimensions + k];
}
costs[1] += getCost(sample, labels[j]);
sampleCounts[labels[j]] += 1;
for (int k = 0; k < this->dimensions; k++) {
nextMeans[labels[j]][k] += sample[k];
}
delete[] sample;
}
costs[1] /= dataSize;
for (int j = 0; j < this->clusters; j++) {
if (sampleCounts[j] > 0) {
for (int k = 0; k < this->dimensions; k++) {
this->means[j][k] = nextMeans[j][k] / sampleCounts[j];
}
}
delete[] nextMeans[j];
}
delete nextMeans;
if (fabs(costs[0] - costs[1]) < this->epsilon * costs[0]) {
break;
}
}
delete[] sampleCounts;
delete[] costs;
return labels;
}
const double KMeans::getCost(const double *sample, int &label) {
label = -1;
double minDistance = INT_MAX;
for (int i = 0; i < this->clusters; i++) {
double temp = getDistance(sample, this->means[i], this->dimensions);
if (temp < minDistance) {
minDistance = temp;
label = i;
}
}
return minDistance;
}
const double KMeans::getDistance(
const double *x, const double *y, const int dimensions) {
double sumSquares = 0;
for (int i = 0; i < dimensions; i++) {
sumSquares += pow(x[i] - y[i], 2);
}
return sqrt(sumSquares);
}
ostream &operator<<(ostream &out, const KMeans *const kmeans) {
out << "模型: K阶中心距聚类" << endl
<< "数据维度: " << kmeans->dimensions << endl
<< "聚类簇数量：" << kmeans->clusters << endl
<< "聚类中心：" << endl;
for (int i = 0; i < kmeans->clusters; i++) {
out << "(" << kmeans->means[i][0];
for (int j = 1; j < kmeans->dimensions; j++) {
out << ", " << kmeans->means[i][j];
}
out << ")" << endl;
}
return out;
}


#pragma once
#include <fstream>
#include "KMeans-Serial.hpp"
class GMM {
private:
int dimensions;
int mixtures;
double *priorities = nullptr;
double **means = nullptr;
double **variances = nullptr;
double *minVariance = nullptr;
int maxIterations;
double epsilon;
KMeans *kmeans = nullptr;
const double getProbability(const double *const sample, const int clusterId);
void initialize(const double *const dataSet, const int dataSize);
public:
GMM(const int dimensions = 1, const int mixtures = 1);
void fit(const double *const dataSet, const int dataSize);
const double getProbability(const double *const sample);
friend ostream &operator<<(ostream &out, const GMM *const model);
};

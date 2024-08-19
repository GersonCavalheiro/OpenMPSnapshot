
#pragma once
#include <iostream>
using namespace std;
enum InitMode {
Randomly, 
Manually, 
Uniformly 
};
class KMeans {
public:
KMeans(const int dimensions = 1, const int clusters = 1);
~KMeans();
void setMean(const int i, const double *point);
void setInitMode(enum InitMode mode);
void setMaxIterations(const int iterations);
void setEpsilon(const double epsilon);
const double *const getMean(const int i);
const int getInitMode();
const int getMaxIterations();
const double getEpsilon();
const int *fit_transform(const double *const datasets, const int dataSize);
const double *const loadFile(ifstream &file, const int size);
const int *predict(const double *const dataset, const int dataSize);
friend ostream &operator<<(ostream &, const KMeans *const);
private:
int dimensions;
int clusters;
double **means;
InitMode initMode;
int maxInterations;
double epsilon;
const double getCost(const double *sample, int &label);
const double getDistance(const double *, const double *, const int);
};

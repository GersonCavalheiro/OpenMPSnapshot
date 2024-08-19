#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
#ifdef _OPENMP
#include <omp.h> 
#endif
#include <cstdio>
#include <iostream>
#include "rapidcsv.h"
#include <cmath>
#include <chrono>

double* getDataset(int* lenght, int* dim);
double* getFarCentroids(double *points, int pointsLength, int dimensions);


const int k = 3;
int main(int argc, char const *argv[]) {

int dataLength; 
int dimensions; 
double *points = getDataset(&dataLength, &dimensions); 
double *centroids = getFarCentroids(points, dataLength, dimensions); 

double distanceFromOld = 0; 
int pointsInCluster[k]; 
double *newCentroids = new double[k*dimensions]; 


auto start = std::chrono::system_clock::now();
do {
for (int x = 0; x < k; x++) {
pointsInCluster[x] = 0;
}
for (int x = 0; x < k*dimensions; x++) {
newCentroids[x] = 0;
}
double dist;
int* clustId = new int[dataLength]; 
double newDist;

#pragma omp parallel for num_threads(8) private(dist, newDist)
for (int i = 0; i < dataLength; i++) {
dist = 100; 
clustId[i] = -1; 
for (int j = 0; j < k; j++) {
newDist = 0; 
for (int x = 0; x < dimensions; x++) {
newDist += fabs(points[i*dimensions + x] - centroids[j*dimensions + x]);
}
if(newDist < dist) {
dist = newDist;
clustId[i] = j;
}
}

}
for(int i = 0; i < dataLength; i++) {
for (int x = 0; x < dimensions; x++) {
newCentroids[clustId[i] * dimensions + x] += points[i * dimensions + x];
}
pointsInCluster[clustId[i]]++;
}

distanceFromOld = 0;
for (int j = 0; j < k; j++) {
for (int x = 0; x < dimensions; x++) {
newCentroids[j*dimensions + x] /= pointsInCluster[j];
distanceFromOld += fabs(newCentroids[j*dimensions + x] - centroids[j*dimensions + x]);
centroids[j*dimensions + x] = newCentroids[j*dimensions + x];
}
}
} while (distanceFromOld > 0.001); 

auto end = std::chrono::system_clock::now();
std::chrono::duration<double> elapsed_seconds = end-start; 

std::ofstream myfile;
myfile.open ("./omp/omp.csv", std::ios::app);
myfile << dataLength;
printf("\n\nomp: %f\n\n\n",elapsed_seconds.count());
myfile << "," << elapsed_seconds.count();
myfile << "\n";
myfile.close();

return 0;
}


double* getDataset(int* lenght, int* dim) {
rapidcsv::Document doc("./dataset.csv", rapidcsv::LabelParams(-1, -1));
const int rows = int(doc.GetRowCount()) - k;
const int dimensions = doc.GetColumnCount() - 1;
*lenght = rows;
*dim = dimensions;
double *points = new double[rows*dimensions];
for(int i = 0; i < rows; i++) {
std::vector<std::string> row = doc.GetRow<std::string>(i);  
double *array = new double[dimensions];
int index = 0;
for(auto element : row) {
if(index != dimensions) {
points[i*dimensions + index] = std::atof(element.c_str());
}
index++;
}
}
return points;
}

double* getFarCentroids(double *points, int pointsLength, int dimensions) {
double reference[dimensions]; 
for (int tmpIdx = 0; tmpIdx < dimensions; tmpIdx++) {
reference[tmpIdx] = points[tmpIdx];
}

double *distances = new double[k-1]; 
double *realPoints = new double[k*dimensions]; 
for (int tmpIdx = 0; tmpIdx < dimensions; tmpIdx++) {
realPoints[(k-1)*dimensions + tmpIdx] = reference[tmpIdx];
}

int maxSize = k - 1; 
for(int i = 0; i < pointsLength; i++){
double dist = 0;
for (int x = 0; x < dimensions; x++) {
dist += fabs(points[i*dimensions + x] - reference[x]);
}
if(dist > distances[maxSize - 1]) { 
int index = 0;
while (dist < distances[index] && index < maxSize) { 
index++;
}
for (int j = maxSize - 1; j > index; j--) {
distances[j] = distances[j - 1];
for (int tmpIdx = 0; tmpIdx < dimensions; tmpIdx++) {
realPoints[j*dimensions + tmpIdx] = realPoints[(j - 1)*dimensions + tmpIdx];
}
}
distances[index] = dist;
for (int tmpIdx = 0; tmpIdx < dimensions; tmpIdx++) {
realPoints[index*dimensions + tmpIdx] = points[i*dimensions + tmpIdx];
}
}
}
return realPoints;
}

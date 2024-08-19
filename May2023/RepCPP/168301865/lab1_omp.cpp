extern "C" {
#include "lab1_omp.h"
}

#include <vector>
#include <tuple>
#include <stdlib.h>
#include <time.h>
#include <limits>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <unistd.h>

std::vector<int>* numOfPointsInPartition;
std::vector<double>*currentCentroidsDouble;
std::vector<int>* partitionEntries;
int* clusterPartIdList;
int* data_pointsG;
int Ng;
int Kg;
int* tidArr;
int cacheLineSizeBlock;

void kmeans_omp(int num_threads, int N, int K, int* data_points, int** data_point_cluster, 
int** centroids, int* num_iterations) {
pthread_t multThreads[num_threads];
tidArr = new int[num_threads];
data_pointsG = data_points;
Ng = N;
Kg = K;
omp_set_dynamic(0);

cacheLineSizeBlock = (int) ceil(double(sysconf(_SC_LEVEL1_DCACHE_LINESIZE))/double(sizeof(double)));

std::vector<int>* centroidSaveDB = new std::vector<int>();
centroidSaveDB->reserve(3*K*201);
currentCentroidsDouble = new std::vector<double>(K*cacheLineSizeBlock);

srand(0);
int numRandomCentInit = 0;
while(numRandomCentInit<K) {
int pickIndex = rand()%N;
int pickedX = *(data_points+(3*pickIndex));
int pickedY = *(data_points+(3*pickIndex)+1);
int pickedZ = *(data_points+(3*pickIndex)+2);

bool matched = false;
for (int i=0; i<numRandomCentInit; i++) {
int iThCurrCentX = (*currentCentroidsDouble)[(cacheLineSizeBlock*i)];
int iThCurrCentY = (*currentCentroidsDouble)[(cacheLineSizeBlock*i)+1];
int iThCurrCentZ = (*currentCentroidsDouble)[(cacheLineSizeBlock*i)+2];
if (pickedX==iThCurrCentX & pickedY==iThCurrCentY & pickedZ==iThCurrCentZ) {
matched = true;
break;
}
}

if (!matched) {
(*currentCentroidsDouble)[cacheLineSizeBlock*numRandomCentInit] = pickedX;
(*currentCentroidsDouble)[cacheLineSizeBlock*numRandomCentInit+1] = pickedY;
(*currentCentroidsDouble)[cacheLineSizeBlock*numRandomCentInit+2] = pickedZ;
numRandomCentInit++;
}
}

for (int i=0; i<K; i++) {
centroidSaveDB->insert(centroidSaveDB->end(), currentCentroidsDouble->begin()+i*cacheLineSizeBlock, 
currentCentroidsDouble->begin()+i*cacheLineSizeBlock+3);
}

*num_iterations = -1;
double error = std::numeric_limits<double>::max();

partitionEntries = new std::vector<int>(N);
numOfPointsInPartition = new std::vector<int>(K);
std::vector<double>* prevCentroidsDouble;
bool zeroThIteration = true;

while (error>1.0 && (*num_iterations<200)) {
if (!zeroThIteration) {
currentCentroidsDouble = new std::vector<double>(cacheLineSizeBlock*K);
for (int i=0; i<K; i++) {
(*numOfPointsInPartition)[i] = 0;
}

#pragma omp parallel num_threads(num_threads)
{
int id = omp_get_thread_num();
int threadNum = omp_get_num_threads();
for (int i=0; i<Ng; i++) {
int partitionId = (*partitionEntries)[i];
if (partitionId%threadNum==id) {
int pickedX = *(data_pointsG+(3*i));
int pickedY = *(data_pointsG+(3*i)+1);
int pickedZ = *(data_pointsG+(3*i)+2);
(*currentCentroidsDouble)[(cacheLineSizeBlock*partitionId)] += pickedX;
(*currentCentroidsDouble)[(cacheLineSizeBlock*partitionId)+1] += pickedY;
(*currentCentroidsDouble)[(cacheLineSizeBlock*partitionId)+2] += pickedZ;
(*numOfPointsInPartition)[partitionId] += 1;
}
}
}

for (int i=0; i<K; i++) {
int iThCurrCentX = (*currentCentroidsDouble)[(cacheLineSizeBlock*i)];
int iThCurrCentY = (*currentCentroidsDouble)[(cacheLineSizeBlock*i)+1];
int iThCurrCentZ = (*currentCentroidsDouble)[(cacheLineSizeBlock*i)+2];
double denominator = double((*numOfPointsInPartition)[i]);
(*currentCentroidsDouble)[(cacheLineSizeBlock*i)] = round(double(iThCurrCentX)/denominator);
(*currentCentroidsDouble)[(cacheLineSizeBlock*i)+1] = round(double(iThCurrCentY)/denominator);
(*currentCentroidsDouble)[(cacheLineSizeBlock*i)+2] = round(double(iThCurrCentZ)/denominator);
}

for (int i=0; i<K; i++) {
centroidSaveDB->insert(centroidSaveDB->end(), currentCentroidsDouble->begin()+i*cacheLineSizeBlock, 
currentCentroidsDouble->begin()+i*cacheLineSizeBlock+3);
}
}

#pragma omp parallel num_threads(num_threads)
{
int id = omp_get_thread_num();
int threadNum = omp_get_num_threads();
int istart = (Ng/threadNum)*(id);
int iend;
if (id==threadNum-1){
iend = Ng;
} else {
iend = (Ng/threadNum)*(id+1);
}
for (int i=istart; i<iend; i++) {
double minDistance = std::numeric_limits<double>::max();
int pickedX = *(data_pointsG+(3*i));
int pickedY = *(data_pointsG+(3*i)+1);
int pickedZ = *(data_pointsG+(3*i)+2);
for (int j=0; j<Kg; j++) {
double jThCurrCentX = (*currentCentroidsDouble)[(cacheLineSizeBlock*j)];
double jThCurrCentY = (*currentCentroidsDouble)[(cacheLineSizeBlock*j)+1];
double jThCurrCentZ = (*currentCentroidsDouble)[(cacheLineSizeBlock*j)+2];
double currCentDistance = pow(pickedX-jThCurrCentX,2)+pow(pickedY-jThCurrCentY,2)+pow(pickedZ-jThCurrCentZ,2);
if (currCentDistance<minDistance) {
minDistance = currCentDistance;
(*partitionEntries)[i] = j;
}
}
}
}

if (!zeroThIteration) {
error = 0.0;
for (int i=0; i<K; i++) {
double iThCurrCentX = (*currentCentroidsDouble)[(cacheLineSizeBlock*i)];
double iThCurrCentY = (*currentCentroidsDouble)[(cacheLineSizeBlock*i)+1];
double iThCurrCentZ = (*currentCentroidsDouble)[(cacheLineSizeBlock*i)+2];
double iThPrevCentX = (*prevCentroidsDouble)[(cacheLineSizeBlock*i)];
double iThPrevCentY = (*prevCentroidsDouble)[(cacheLineSizeBlock*i)+1];
double iThPrevCentZ = (*prevCentroidsDouble)[(cacheLineSizeBlock*i)+2];
error += pow(iThCurrCentX-iThPrevCentX, 2) + pow(iThCurrCentY-iThPrevCentY, 2) + pow(iThCurrCentZ-iThPrevCentZ, 2);
}
prevCentroidsDouble->clear();
delete prevCentroidsDouble;
} else {
zeroThIteration = false;
}
prevCentroidsDouble = currentCentroidsDouble;
*num_iterations += 1;
}
prevCentroidsDouble->clear();
delete prevCentroidsDouble;
delete currentCentroidsDouble;

clusterPartIdList = (int*) malloc(sizeof(int)*N*4);
#pragma opm parallel set_threads(num_threads)
{
int id = omp_get_thread_num();
int threadNum = omp_get_num_threads();
int istart = (Ng/threadNum)*(id);
int iend;
if (id==threadNum-1) {
iend = Ng;
} else {
iend = (Ng/threadNum)*(id+1);
}
for (int i=istart; i<iend; i++) {
int partitionId = (*partitionEntries)[i];
int pickedX = *(data_pointsG+(3*i));
int pickedY = *(data_pointsG+(3*i)+1);
int pickedZ = *(data_pointsG+(3*i)+2);
*(clusterPartIdList+(4*i)) = pickedX;
*(clusterPartIdList+(4*i+1)) = pickedY;
*(clusterPartIdList+(4*i)+2) = pickedZ;
*(clusterPartIdList+(4*i)+3) = partitionId;
}
}


*centroids = centroidSaveDB->data();
*data_point_cluster = clusterPartIdList;
delete[] tidArr;
}
#include <iostream>
#include "filterFinalMatrix.h"
#include "computeB.h"
#include "printMatrix.h"

void filterFinalMatrix(double *&A, double *&B,
int *&nonZeroUserIndexes,
int *&nonZeroItemIndexes,
double *&nonZeroElements,
double *&L,
double *&R,
int &numberOfUsers, int &numberOfItems, int &numberOfFeatures,
int &numberOfNonZeroElements,
int *&BV) {

int i, j, l;

#pragma omp parallel shared(numberOfUsers, numberOfItems, numberOfFeatures, A, B, L, R, numberOfNonZeroElements, nonZeroUserIndexes, nonZeroItemIndexes, BV, std::cout)  default(none) private(i, l, j)
{
computeB(L, R, numberOfUsers, numberOfItems, numberOfFeatures, B);

#pragma omp for private(l) schedule(static)
for (int l = 0; l < numberOfNonZeroElements; l++) {
B[nonZeroUserIndexes[l] * numberOfItems + nonZeroItemIndexes[l]] = 0;
}

#pragma omp for private(i, j) schedule(guided)
for (int i = 0; i < numberOfUsers; i++) {
double max = 0;
int maxPosition;
for (int j = 0; j < numberOfItems; j++) {
if (B[i * numberOfItems + j] > max) {
max = B[i * numberOfItems + j];
maxPosition = j;
} else {
continue;
}
}
BV[i] = maxPosition;
}
};


for (int i = 0; i < numberOfUsers; i++) {
std::cout << BV[i] << std::endl;
}
};
#include "updateLR.h"

void updateLR(double *&A,
double *&prediction, double *&delta,
int *&nonZeroUserIndexes,
int *&nonZeroItemIndexes,
double *&L, double *&R,
double *&StoreL, double *&StoreR,
int &numberOfUsers, int &numberOfItems, int &numberOfFeatures,
int &numberOfNonZeroElements,
double &convergenceCoefficient) {
int i, l, k;

#pragma omp parallel shared(numberOfNonZeroElements, numberOfUsers, numberOfItems, numberOfFeatures, nonZeroUserIndexes, nonZeroItemIndexes, prediction, A, L, R, StoreL, StoreR, convergenceCoefficient, delta) default(none)
{
#pragma omp for private(i, k) schedule(static)
for (int i = 0; i < numberOfFeatures; i++) {
for (int k = 0; k < numberOfUsers; k++) {
StoreL[k * numberOfFeatures + i] = L[k * numberOfFeatures + i];
}
for (int k = 0; k < numberOfItems; k++) {
StoreR[i * numberOfItems + k] = R[i * numberOfItems + k];
}
}

#pragma omp for private(l, k) schedule(static)
for (int l = 0; l < numberOfNonZeroElements; l++) {
prediction[l] = 0;
delta[l] = 0;
for (int k = 0; k < numberOfFeatures; k++) {
prediction[l] += L[nonZeroUserIndexes[l] * numberOfFeatures + k] * R[k * numberOfItems + nonZeroItemIndexes[l]];
}
delta[l] = A[nonZeroUserIndexes[l] * numberOfItems + nonZeroItemIndexes[l]] - prediction[l];
}

#pragma omp for private(l, k) collapse(2) schedule(static)
for (int l = 0; l < numberOfNonZeroElements; l++) {
for (int k = 0; k < numberOfFeatures; k++) {
#pragma omp atomic
L[nonZeroUserIndexes[l] * numberOfFeatures + k] += convergenceCoefficient * (2 * delta[l] * StoreR[k * numberOfItems + nonZeroItemIndexes[l]]);
#pragma omp atomic
R[k * numberOfItems + nonZeroItemIndexes[l]] += convergenceCoefficient * (2 * delta[l] * StoreL[nonZeroUserIndexes[l] * numberOfFeatures + k]);
}
}
};
};
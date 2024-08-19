#include <omp.h>

#include "gaussianElimination.h"

#include "../../../lib/exceptions.h"


namespace numath {
namespace systemsOfEquations {

std::vector<double> simpleGaussianElimination(std::vector<std::vector<double>> augmentedMatrix) {
std::vector<double> results;
try {
__forwardElimination(augmentedMatrix);
results = __backwardSubstitution(augmentedMatrix);
}
catch (DenominatorException &ex) {
throw ex;
}
return results;
}

void __forwardElimination(std::vector<std::vector<double>> &augmentedMatrix) {
const int N = augmentedMatrix.size();
double multDenominator,multiplier;
for (int k = 1; k <= N-1; k++) {
for(int i = k + 1; i <= N; i++) {
multDenominator = augmentedMatrix[k-1][k-1];
if (multDenominator == 0) {
throw DenominatorException();
}
else {
multiplier = augmentedMatrix[i-1][k-1] / multDenominator;
for (int j = k; j <= N + 1; j++) {
augmentedMatrix[i-1][j-1] = augmentedMatrix[i-1][j-1] - (multiplier * augmentedMatrix[k-1][j-1]);
}
}
}
}
}

std::vector<double> __backwardSubstitution(std::vector<std::vector<double>> &augmentedTriangularMatrix) {
const int N = augmentedTriangularMatrix.size();
std::vector<double> results(N, 0.0);
for (int i = N; i > 0; i--) {
results[i-1] = augmentedTriangularMatrix[i-1][N];
for (int j = i+1; j <= N; j++) {
results[i-1] = results[i-1] - augmentedTriangularMatrix[i-1][j-1] * results[j-1];
}
double denominator = augmentedTriangularMatrix[i-1][i-1];        
if (denominator == 0) {
throw DenominatorException();
}
else {
results[i-1] = results[i-1] / denominator;
}
}
return results;
}



}
}
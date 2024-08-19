#include <omp.h>
#include "gaussianEliminationPartialPivot.h"

#include "../../../lib/exceptions.h"

#include <cmath>

namespace numath {
namespace systemsOfEquations {

std::vector<double> gaussianEliminationPartialPivot(std::vector<std::vector<double>> augmentedMatrix) {
std::vector<double> results;
try {
__forwardEliminationPP(augmentedMatrix);
results = __backwardSubstitutionPP(augmentedMatrix);
}
catch (DenominatorException &ex) {
throw ex;
}

return results;
}

void __forwardEliminationPP(std::vector<std::vector<double>> &augmentedMatrix) {
const int N = augmentedMatrix.size();
for (int k = 1; k <= N-1; k++) {
__partialPivot(augmentedMatrix,k,N);
#pragma omp parallel for shared(augmentedMatrix, k) 
for(int i = k + 1; i <= N; i++) {

double multDenominator = augmentedMatrix[k-1][k-1];
if (multDenominator == 0) {
throw DenominatorException();
}
else {
double multiplier = augmentedMatrix[i-1][k-1] / multDenominator;

for (int j = k; j <= N + 1; j++) {
augmentedMatrix[i-1][j-1] = augmentedMatrix[i-1][j-1] - (multiplier * augmentedMatrix[k-1][j-1]);
}
}
}
}
}

void __partialPivot(std::vector<std::vector<double>> &augmentedMatrix,int k, int n){
double maxElement = fabs(augmentedMatrix[k-1][k-1]);
int maxRow = k-1;
#pragma omp parallel for shared(maxElement, maxRow, k, n)
for(int s = k-1; s<n; s++){
double newElement = fabs(augmentedMatrix[s][k-1]);
if(newElement>maxElement){
maxElement=newElement;
maxRow =s;
}
}
if(maxElement==0){
throw SolutionException();
}else{
if(maxRow !=k-1){
for(int i =0;i< abs(augmentedMatrix[0].size());i++){
double aux = augmentedMatrix[k-1][i];
augmentedMatrix[k-1][i] = augmentedMatrix[maxRow][i];
augmentedMatrix[maxRow][i] = aux;
}
}
}

}


std::vector<double> __backwardSubstitutionPP(std::vector<std::vector<double>> &augmentedTriangularMatrix) {
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

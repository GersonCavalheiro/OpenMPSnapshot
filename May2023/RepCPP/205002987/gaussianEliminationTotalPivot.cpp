#include <omp.h>
#include "gaussianEliminationTotalPivot.h"

#include "../../../lib/exceptions.h"

#include <cmath>

namespace numath {
namespace systemsOfEquations {

std::vector<double> gaussianEliminationTotalPivot(std::vector<std::vector<double>> augmentedMatrix) {

std::vector<double> results, r;
try {
std::vector<int> marks =__forwardEliminationTP(augmentedMatrix);
results = __backwardSubstitutionTP(augmentedMatrix);
r = __orderResults(marks,results);
}
catch (DenominatorException &ex) {
throw ex;
}

return r;
}

std::vector<double> __orderResults(std::vector<int> marks, std::vector<double> results){
std::vector<double> r ={11,22,33,44};
int i =0;
for(int a: marks){
r[a-1]=results[i]; 
i++;
}
return r;
}

std::vector<int> __forwardEliminationTP(std::vector<std::vector<double>> &augmentedMatrix) {
const int N = augmentedMatrix.size();
std::vector<int> marks = __fillMarks(N);
for (int k = 1; k <= N-1; k++) {
marks = __totalPivot(augmentedMatrix,marks,k,N);
#pragma omp parallel for
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
return marks;
}

std::vector<int> __totalPivot(std::vector<std::vector<double>> &augmentedMatrix,std::vector<int> &marks,int k, int n){
double max =0;
int maxRow =k-1;
int maxColumn =k-1;
#pragma omp parallel for shared(max, maxRow, maxColumn)
for(int r = k-1; r < n; r++){
for(int s = k-1; s < n; s++){
if(abs(augmentedMatrix[r][s]) > max){
max = abs(augmentedMatrix[r][s]);
maxRow = r;
maxColumn = s;
}
}
}
if(max == 0){
throw SolutionException();
}else{
if(maxRow != k-1){
for(int i = 0; i < abs(augmentedMatrix[0].size()); i++){
double aux = augmentedMatrix[k-1][i];
augmentedMatrix[k-1][i] = augmentedMatrix[maxRow][i];
augmentedMatrix[maxRow][i] = aux;
}

}
if(maxColumn != k-1){
for(int i = 0; i < n; i++){
double aux = augmentedMatrix[i][k-1];
augmentedMatrix[i][k-1] = augmentedMatrix[i][maxColumn];
augmentedMatrix[i][maxColumn] = aux;
}
int aux2 = marks[maxColumn];
marks[maxColumn] = marks[k-1];
marks[k-1] = aux2;

}
}
return marks;
}


std::vector<double> __backwardSubstitutionTP(std::vector<std::vector<double>> &augmentedTriangularMatrix) {
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


std::vector<int> __fillMarks(int n){
size_t s = n;
std::vector<int> marks(s);
#pragma omp parallel for
for(int i = 0; i < n; i++){
marks[i] = i+1;
}
return marks;
}



}
}
#include "jacobi.h"

#include <omp.h>

namespace numath {
namespace systemsOfEquations {

std::vector<double> jacobi(std::vector<double> &variables, 
std::vector<std::vector<double>> &matrix,
std::vector<double> &indepTerms) {

double sum;
std::vector<double> results (indepTerms.size());
const int N = variables.size();

#pragma omp parallel for private(sum)
for (int i = 0; i < N; i++) {

sum = 0;
for (int j = 0; j < N; j++) {
if (j != i) {
sum = sum + (matrix[i][j] * variables[j]);
}
}

results[i] = (indepTerms[i] - sum) / matrix[i][i];
}
return results;
}

}
}
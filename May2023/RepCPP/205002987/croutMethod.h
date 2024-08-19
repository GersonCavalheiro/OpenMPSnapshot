#pragma once
#include <vector>

namespace numath {
namespace systemsOfEquations {


std::vector<double> croutMethod(std::vector<std::vector<double>> A, std::vector<double> b);


void __initializeMatrix(std::vector<std::vector<double>> &L,std::vector<std::vector<double>> &U);


void __LUFactoring(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> &L, std::vector<std::vector<double>> &U,  int N);


std::vector<double> __forwardSubstitutionCM(std::vector<std::vector<double>> &L, std::vector<double> &b);


std::vector<double> __backwardSubstitutionCM(std::vector<std::vector<double>> &U, std::vector<double> &z);


void toStringMatrixCR(std::vector<std::vector<double>> &augmentedMatrix);
void toStringIncMatrixCR(std::vector<std::vector<double>> &matrix, char name);

}
}
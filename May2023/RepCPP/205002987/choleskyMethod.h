#pragma once
#include <vector>

namespace numath {
namespace systemsOfEquations {


std::vector<double> choleskyMethod(std::vector<std::vector<double>> A, std::vector<double> b);


void __initializeMatrixCH(std::vector<std::vector<double>> &L,std::vector<std::vector<double>> &U);


void __LUFactoringCH(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> &L, std::vector<std::vector<double>> &U,  int N);


std::vector<double> __forwardSubstitutionCHM(std::vector<std::vector<double>> &L, std::vector<double> &b);


std::vector<double> __backwardSubstitutionCHM(std::vector<std::vector<double>> &U, std::vector<double> &z);


void toStringMatrixCH(std::vector<std::vector<double>> &matrix);


void toStringIncMatrixCH(std::vector<std::vector<double>> &matrix, char name);

}
}
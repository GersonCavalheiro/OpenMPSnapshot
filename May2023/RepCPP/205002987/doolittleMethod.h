#pragma once
#include <vector>

namespace numath {
namespace systemsOfEquations {


std::vector<double> doolittleMethod(std::vector<std::vector<double>> A, std::vector<double> b);


void __initializeMatrixDM(std::vector<std::vector<double>> &L,std::vector<std::vector<double>> &U);


void __LUFactoringDM(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> &L, std::vector<std::vector<double>> &U,  int N);


std::vector<double> __forwardSubstitutionDM(std::vector<std::vector<double>> &L, std::vector<double> &b);


std::vector<double> __backwardSubstitutionDM(std::vector<std::vector<double>> &U, std::vector<double> &z);



void toStringMatrixD(std::vector<std::vector<double>> &matrix);


void toStringIncMatrixD(std::vector<std::vector<double>> &matrix, char name);

}
}
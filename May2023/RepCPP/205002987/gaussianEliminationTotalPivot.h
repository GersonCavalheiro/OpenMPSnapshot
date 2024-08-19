#pragma once
#include <vector>

namespace numath {
namespace systemsOfEquations {


std::vector<double> gaussianEliminationTotalPivot(std::vector<std::vector<double>> augmentedMatrix);


std::vector<int> __forwardEliminationTP(std::vector<std::vector<double>> &augmentedMatrix);


std::vector<int> __totalPivot(std::vector<std::vector<double>> &augmentedMatrix, std::vector<int> &marks, int k, int n);


std::vector<double> __backwardSubstitutionTP(std::vector<std::vector<double>> &augmentedTriangularMatrix);


std::vector<int> __fillMarks(int n);

void toStringMatrixGT(std::vector<std::vector<double>> &augmentedMatrix);
std::vector<double> __orderResults(std::vector<int> marks, std::vector<double> results);

}
}
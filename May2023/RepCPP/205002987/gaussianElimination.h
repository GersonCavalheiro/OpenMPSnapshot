#pragma once
#include <vector>

namespace numath {
namespace systemsOfEquations {


std::vector<double> simpleGaussianElimination(std::vector<std::vector<double>> augmentedMatrix);


void __forwardElimination(std::vector<std::vector<double>> &augmentedMatrix);


std::vector<double> __backwardSubstitution(std::vector<std::vector<double>> &augmentedTriangularMatrix);


void toStringMatrixGE(std::vector<std::vector<double>> &augmentedMatrix);

}
}
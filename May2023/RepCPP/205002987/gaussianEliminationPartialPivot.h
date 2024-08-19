#pragma once
#include <vector>

namespace numath {
namespace systemsOfEquations {


std::vector<double> gaussianEliminationPartialPivot(std::vector<std::vector<double>> augmentedMatrix);


void __forwardEliminationPP(std::vector<std::vector<double>> &augmentedMatrix);


void __partialPivot(std::vector<std::vector<double>> &augmentedMatrix, int k, int n);


std::vector<double> __backwardSubstitutionPP(std::vector<std::vector<double>> &augmentedTriangularMatrix);


void toStringMatrixGP(std::vector<std::vector<double>> &augmentedMatrix);

}
}
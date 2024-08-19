#pragma once

#include <vector>
namespace numath {
namespace singleVariableEquations {


double falsePosition(double (*func)(double), double xi, double xu, int nIter, double tol, const char *errorType, std::vector<std::vector<double>> &table);

}
}
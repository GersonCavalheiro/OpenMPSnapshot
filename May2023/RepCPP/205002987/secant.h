#pragma once

#include <vector>

namespace numath {
namespace singleVariableEquations {


double secant(double (*func)(double), double x0, double x1, int nIter, double tol, const char *errorType, std::vector<std::vector<double>> &table);

}
}
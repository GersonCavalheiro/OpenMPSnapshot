#pragma once

#include <vector>

namespace numath {
namespace singleVariableEquations {


double newton(double (*func)(double), double (*dFunc)(double), double x0, int nIter, double tol, const char *errorType, std::vector<std::vector<double>> &table);

}
}
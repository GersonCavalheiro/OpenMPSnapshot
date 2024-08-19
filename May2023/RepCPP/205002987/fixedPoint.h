#pragma once

#include <vector>

namespace numath {
namespace singleVariableEquations {


double fixedPoint(double (*func)(double), double (*gFunc)(double), double xa, int nIter, double tol, const char *errorType, std::vector<std::vector<double>> &table);

}
}
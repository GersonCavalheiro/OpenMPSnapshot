#pragma once

namespace numath {
namespace singleVariableEquations {


double bisection(double (*func)(double), double xi, double xu, int nIter, double tol, const char *errorType, std::vector<std::vector<double>> &table);

}
}
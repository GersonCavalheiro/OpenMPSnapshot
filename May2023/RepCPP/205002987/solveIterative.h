#pragma once
#include <vector>

namespace numath {
namespace systemsOfEquations {


std::vector<double> solveIterative(std::vector<std::vector<double>> &A, 
std::vector<double> &b,
std::vector<double> &initialValues, 
double tol, 
int nIter,
std::vector<double> (*method)(std::vector<double>&,
std::vector<std::vector<double>>&,
std::vector<double>&),
double (*errorFunc)(std::vector<double>&,
std::vector<double>&),
double lambda,
std::vector<std::vector<double>> &table);

}
}
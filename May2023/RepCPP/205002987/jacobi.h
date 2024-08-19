#pragma once
#include <vector>

namespace numath {
namespace systemsOfEquations {


std::vector<double> jacobi(std::vector<double> &variables, 
std::vector<std::vector<double>> &matrix,
std::vector<double> &indepTerms);

}
}
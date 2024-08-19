#pragma once

#include <fstream>
#include <vector>
#include "interval.h"

namespace numath {
namespace singleVariableEquations {


Interval incrementalSearch(double (*func)(double), double x0, double delta, int nIter, std::vector<std::vector<double>> &table);

}
}
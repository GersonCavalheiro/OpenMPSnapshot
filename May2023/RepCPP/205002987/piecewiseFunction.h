#pragma once

#include <vector>
#include <string>
#include "point.h"

namespace numath {

struct PiecewiseFunction {
std::vector<std::string> functions;
std::vector<Point> limits;
};

}

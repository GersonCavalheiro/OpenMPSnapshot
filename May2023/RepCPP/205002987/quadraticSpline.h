#pragma once

#include <string>
#include <vector>
#include <utility>
#include "point.h"

namespace numath {
namespace interpolation {



std::pair<std::vector<std::vector<double>>, std::vector<double>> quadraticSpline(std::vector<numath::Point> &points);

}
}
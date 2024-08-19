#pragma once

#include <string>
#include <vector>
#include "point.h"
#include "piecewiseFunction.h"

namespace numath {
namespace interpolation {



PiecewiseFunction linearSpline(std::vector<numath::Point> &points);

}
}
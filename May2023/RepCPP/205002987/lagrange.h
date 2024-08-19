#pragma once

#include <vector>
#include <string>
#include "point.h"

namespace numath{
namespace interpolation {


std::string lagrange(std::vector<numath::Point> &points);

}
}
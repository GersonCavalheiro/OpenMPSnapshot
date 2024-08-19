#pragma once
#include "point.h"
#include <vector>
namespace numath{
namespace differentiation{

double differentiation(std::vector<Point> &points, int direction, double h, int num);
} 
}
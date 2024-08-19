

#pragma once

#include "Common.h"
#include "Distance.h"

void compute_classes(const std::vector<std::array<float, Nv>>& Vec, const std::vector<std::array<float, Nv>>& old_Center, std::array<std::vector<int>, Nc>& Classes);
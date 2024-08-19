

#pragma once

#include "Common.h"

void optimize_center(const std::vector<std::array<float, Nv>>& Vec, std::vector<std::array<float, Nv>>& new_Center, const std::array<std::vector<int>, Nc>& Classes);


#pragma once

#include "Common.h"

long double eucl_diff(const std::array<float, Nv>& src, const std::array<float, Nv>& dst);
long double convergence(const std::vector<std::array<float, Nv>>& curr_Center, const std::vector<std::array<float, Nv>>& prev_Center);
long double normalize_convergence(const long double curr_iter_conv, const long double prev_iter_conv, const int iter_counter);
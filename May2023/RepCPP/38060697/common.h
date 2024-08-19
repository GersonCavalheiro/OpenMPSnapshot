#pragma once

#include <algorithm>
#include <vector>

template <int total_runs = 20, typename Op>
auto benchmark(Op op) {
auto duration = op();  
std::vector<decltype(duration)> measurements;
for (int i = 1; i < total_runs; i++) {
duration = op();
measurements.push_back(duration);
}
return *std::min_element(std::begin(measurements), std::end(measurements));
}

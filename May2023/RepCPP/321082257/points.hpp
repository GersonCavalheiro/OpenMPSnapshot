#pragma once

#include <algorithm>
#include <iomanip>
#include <string>
#include <vector>

using namespace std::string_literals;

using Points = std::vector<Vec3>;

inline std::ostream& operator<<(std::ostream& os, const Points& points)
{
bool first = true;
for (auto v : points) {
if (!first) {
os << ", "s;
} else {
first = false;
}
os << v.str();
}
return os;
}

size_t index_closest(const Points xs, const Vec3 p)
{
size_t smallest_i = 0; 
double smallest_squared_dist = std::numeric_limits<double>::max(); 
for (size_t i = 0; i < xs.size(); ++i) {
if (p.distance_squared(xs[i]) < smallest_squared_dist) {
smallest_squared_dist = p.distance_squared(xs[i]);
smallest_i = i;
}
}
return smallest_i;
}

size_t index_closest_except(
const Points xs, const Vec3 p, const std::vector<size_t> except_indices)
{
int smallest_i = 0; 
double smallest_squared_dist = std::numeric_limits<double>::max(); 
for (size_t i = 0; i < xs.size(); ++i) {
if (std::any_of(std::begin(except_indices), std::end(except_indices),
[&](size_t i2) { return i2 == i; })) {
continue;
}
if (p.distance_squared(xs[i]) < smallest_squared_dist) {
smallest_squared_dist = p.distance_squared(xs[i]);
smallest_i = i;
}
}
return smallest_i;
}

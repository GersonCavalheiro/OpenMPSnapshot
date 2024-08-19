

#pragma once

#include "adt/Point.h" 
#include <algorithm>   
#include <cassert>     
#include <cstdint>     
#include <functional>  
#include <limits>      
#include <memory>      
#include <type_traits> 
#include <vector>      

namespace rawspeed {
class iPoint2D;


template <typename T = uint16_t,
typename = std::enable_if_t<std::is_arithmetic_v<T>>>
class Spline final {
public:
using value_type = T;

struct Segment {
double a;
double b;
double c;
double d;
};

private:
int num_coords;
int num_segments;

std::vector<int> xCp;
std::vector<Segment> segments;

void prepare() {
std::vector<double> h(num_segments);
std::vector<double> alpha(num_segments);
std::vector<double> mu(num_coords);
std::vector<double> z(num_coords);

for (int i = 0; i < num_segments; i++)
h[i] = xCp[i + 1] - xCp[i];

for (int i = 1; i < num_segments; i++) {
Segment& sp = segments[i - 1];
Segment& s = segments[i];
Segment& sn = segments[i + 1];

alpha[i] = (3. / h[i]) * (sn.a - s.a) - (3. / h[i - 1]) * (s.a - sp.a);
}

mu[0] = z[0] = 0;

for (int i = 1; i < num_segments; i++) {
const double l = 2 * (xCp[i + 1] - xCp[i - 1]) - (h[i - 1] * mu[i - 1]);
mu[i] = h[i] / l;
z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l;
}

z.back() = segments.back().c = 0;

for (int i = num_segments - 1; i >= 0; i--) {
Segment& s = segments[i];
Segment& sn = segments[i + 1];

s.c = z[i] - mu[i] * sn.c;
s.b = (sn.a - s.a) / h[i] - h[i] * (sn.c + 2 * s.c) / 3.;
s.d = (sn.c - s.c) / (3. * h[i]);
}

segments.pop_back();

assert(static_cast<typename decltype(segments)::size_type>(num_segments) ==
segments.size());
}

public:
explicit Spline(const std::vector<iPoint2D>& control_points) {
assert(control_points.size() >= 2 &&
"Need at least two points to interpolate between");

assert(control_points.front().x == 0);
assert(control_points.back().x == 65535);

assert(std::adjacent_find(
control_points.cbegin(), control_points.cend(),
[](const iPoint2D& lhs, const iPoint2D& rhs) -> bool {
return std::greater_equal<>()(lhs.x, rhs.x);
}) == control_points.cend() &&
"The X coordinates must all be strictly increasing");

#ifndef NDEBUG
if (!std::is_floating_point<value_type>::value) {
std::for_each(control_points.cbegin(), control_points.cend(),
[](const iPoint2D& p) {
assert(p.y >= std::numeric_limits<value_type>::min());
assert(p.y <= std::numeric_limits<value_type>::max());
});
}
#endif

num_coords = control_points.size();
num_segments = num_coords - 1;

xCp.resize(num_coords);
segments.resize(num_coords);
for (int i = 0; i < num_coords; i++) {
xCp[i] = control_points[i].x;
segments[i].a = control_points[i].y;
}

prepare();
}

[[nodiscard]] std::vector<Segment> getSegments() const { return segments; }

[[nodiscard]] std::vector<value_type> calculateCurve() const {
std::vector<value_type> curve(65536);

for (int i = 0; i < num_segments; i++) {
const Segment& s = segments[i];

for (int x = xCp[i]; x <= xCp[i + 1]; x++) {
double diff = x - xCp[i];
double diff_2 = diff * diff;
double diff_3 = diff * diff * diff;

double interpolated = s.a + s.b * diff + s.c * diff_2 + s.d * diff_3;

if (!std::is_floating_point<value_type>::value) {
interpolated = std::max(
interpolated, double(std::numeric_limits<value_type>::min()));
interpolated = std::min(
interpolated, double(std::numeric_limits<value_type>::max()));
}

curve[x] = interpolated;
}
}

return curve;
}
};

} 

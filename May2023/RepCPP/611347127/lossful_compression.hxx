#pragma once

#include <cstdio> 
#include <cassert> 
#include <vector> 

template <typename real_t, typename real_y_t>
int RamerDouglasPeucker(
std::vector<bool> & active
, real_t const x_list[]
, real_y_t const y_list[]
, int const end
, int const begin=0
, float const epsilon=1e-6
) {
double d2max{0};
int index{-1};
int const p0 = begin, pl = end - 1;
assert(active[p0]); 
assert(active[pl]); 
double const x0 = x_list[p0], y0 = y_list[p0];
double const x_dist = x_list[pl] - x0;
double const y_dist = y_list[pl] - y0;
double const det = x_dist*x_dist + y_dist*y_dist;
if (det <= 0) return -1; 
double const det_inv = 1./det;

for (int i = p0 + 1; i < pl; ++i) { 
if (active[i]) {
double const xi = x_list[i], yi = y_list[i];


double const si = (x_dist*(yi - y0) - y_dist*(xi - x0))*det_inv;

double const d2 = si*si*det;
if (d2 > d2max) {
index = i;
d2max = d2;
} 
} 
} 

if (d2max > epsilon*epsilon) {
assert(index > p0); assert(index < pl);
int const n0 = RamerDouglasPeucker(active, x_list, y_list, index + 1, begin, epsilon);
int const n1 = RamerDouglasPeucker(active, x_list, y_list, end,       index, epsilon);
return n0 + n1 - 1;
} else {
for (int i = p0 + 1; i < pl; ++i) {
active[i] = false; 
} 
return 2;
} 

} 


template <typename real_t, typename real_y_t>
std::vector<bool> RDP_lossful_compression(
real_t const x[]
, real_y_t const y[]
, int const n
, float const epsilon=1e-6
) {
std::vector<bool> mask(n, true);
RamerDouglasPeucker(mask, x, y, n, 0, epsilon);
return mask;
} 


template <typename real_t, typename real_y_t>
void print_compressed(
real_t const x[]
, real_y_t const y[]
, int const n
, float const epsilon=1e-6
, FILE* os=stdout
) {
auto const mask = RDP_lossful_compression(x, y, n, epsilon);
for (int i = 0; i < n; ++i) {
if (mask[i]) std::fprintf(os, "%g %g\n", x[i], y[i]);
} 
std::fprintf(os, "\n");
} 

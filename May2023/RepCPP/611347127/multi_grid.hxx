#pragma once

#include <cstdio> 
#include <cmath> 
#include <cstdint> 
#include <algorithm> 

#include "real_space.hxx" 
#include "data_view.hxx" 
#include "inline_math.hxx" 
#include "status.hxx" 

namespace multi_grid {

status_t analyze_grid_sizes(
real_space::grid_t const & g 
, uint32_t *n_coarse
, int const echo=0
); 



template <typename real_t, typename real_in_t>
status_t restrict_to_any_grid(
real_t out[]
, unsigned const go
, real_in_t const in[]
, unsigned const gi
, size_t const stride=1
, int const bc=0
, int const echo=0 
, bool const use_special_version_for_2x=true
) {
if (go < 1) return go; 
if (gi < 1) return gi; 

view2D<real_t> target(out, stride); 
view2D<real_in_t const> source(in, stride); 

if (use_special_version_for_2x && (2*go == gi)) {
real_t const w8 = 0.5;
for (int io = 0; io < go; ++io) {
set(        target[io], stride, source[2*io + 0], w8);
add_product(target[io], stride, source[2*io + 1], w8);
} 

} else { 
assert(go <= gi); 
if (go == gi) warn("restriction is a copy operation for %d grid points", go);

double const ratio = go/double(gi);
for (int io = 0; io < go; ++io) {
real_t w8s[4] = {0,0,0,0};
int iis[4];
int nw8 = 0;
for (int ii = 0; ii < gi; ++ii) {
double const start = std::max((ii - 0)*ratio, double(io - 0));
double const end   = std::min((ii + 1)*ratio, double(io + 1));
double const w8 = std::max(0.0, end - start);
if (w8 > 0) {
assert(nw8 < 4);
w8s[nw8] = w8;
iis[nw8] = ii;
++nw8;
} 
} 
assert(std::abs((w8s[0] + w8s[1] + w8s[2] + w8s[3]) - 1) < 1e-6);
set(target[io], stride, real_t(0));
for (int iw8 = 0; iw8 < nw8; ++iw8) {
int const ii = iis[iw8];
add_product(target[io], stride, source[ii], w8s[iw8]);
} 
} 
} 

return 0;
} 


template <typename real_t, typename real_in_t>
status_t linear_interpolation(
real_t out[]
, unsigned const go
, real_in_t const in[]
, unsigned const gi
, size_t const stride=1
, int const periodic=0
, int const echo=0 
, bool const use_special_version_for_2x=true
) {
if (go < 1) return go; 
if (gi < 1) return gi; 

view2D<real_t> target(out, stride); 
view2D<real_in_t const> source(in, stride); 
view2D<real_in_t> buffer(2, stride, 0.0); 
if (periodic) {
set(buffer[1], stride, source[0]);      
set(buffer[0], stride, source[gi - 1]); 
} 

if (echo > 2) std::printf("# %s from %d to %d grid points (inner dim %ld)\n", __func__, gi, go, stride);

if ((go == 2*gi) && use_special_version_for_2x) {
real_t const w14 = 0.25, w34 = 0.75;
set(        target[0], stride, buffer[0], w14); 
add_product(target[0], stride, source[0], w34); 
for (int ii = 1; ii < gi; ++ii) {
set(        target[2*ii - 1], stride, source[ii - 1], w34); 
add_product(target[2*ii - 1], stride, source[ii + 0], w14); 
set(        target[2*ii + 0], stride, source[ii - 1], w14); 
add_product(target[2*ii + 0], stride, source[ii + 0], w34); 
} 
set(        target[go - 1], stride, source[gi - 1], w34); 
add_product(target[go - 1], stride, buffer[1],      w14); 

} else { 

double const ratio = gi/double(go);
for (int io = 0; io < go; ++io) {
double const p = (io + 0.5)*ratio + 0.5;
int const ii = int(p); 
real_t const wu = p - ii; 
real_t const wl = 1 - wu; 
assert(std::abs((wl + wu) - 1) < 1e-6);
set(        target[io], stride, (ii >  0) ? source[ii - 1] : buffer[0], wl);
add_product(target[io], stride, (ii < gi) ? source[ii]     : buffer[1], wu);
} 

} 
return 0;
} 


template <typename real_t>
void print_min_max(
real_t const *const begin
, real_t const *const end
, int const echo=0
, char const *title="<array>"
, char const *unit=""
) {
#ifdef DEVEL
if (echo < 1) return;
auto const mm = std::minmax_element(begin, end);
std::printf("# %24s in range [%g, %g] %s\n", title, *mm.first, *mm.second, unit);
#endif
} 


template <typename real_t, typename real_in_t=real_t>
status_t restrict3D(
real_t out[]
, real_space::grid_t const & go
, real_in_t const in[]
, real_space::grid_t const & gi
, int const echo=0 
) {
status_t stat(0);

{ 
int bc[3];
for (char d = 'x'; d  <= 'z'; ++d) {
assert(go(d) <= gi(d));
bc[d-120] = gi.boundary_condition(d);
assert(go.boundary_condition(d) == bc[d-120]);
if (echo > 0) std::printf("# %s in %c-direction from %d to %d grid points, boundary condition %d\n", 
__func__, d, gi(d), go(d), bc[d-120]);
} 
} 

print_min_max(in, in + gi.all(), echo, "input");

size_t const nyx = gi('y')*gi('x');
view2D<real_t> tz(go('z'), nyx); 
stat += restrict_to_any_grid(tz.data(), go('z'), 
in, gi('z'), nyx, gi.boundary_condition('z'), echo);

print_min_max(tz.data(), tz.data() + go('z')*nyx, echo, "z-restricted");

size_t const nx = gi('x');
view2D<real_t> ty(go('z'), go('y')*nx); 
for (int z = 0; z < go('z'); ++z) {
stat += restrict_to_any_grid(ty[z], go('y'),
tz[z], gi('y'), nx, gi.boundary_condition('y'), echo*(z == 0));
} 

print_min_max(ty.data(), ty.data() + go('z')*go('y')*nx, echo, "zy-restricted");

view3D<real_t> tx(out, go('y'), go('x')); 
for (int z = 0; z < go('z'); ++z) {
for (int y = 0; y < go('y'); ++y) {
stat += restrict_to_any_grid(tx(z,y), go('x'),
ty[z] + y*nx, gi('x'), 1, gi.boundary_condition('x'), echo*(z == 0)*(y == 0));
} 
} 

print_min_max(out, out + go.all(), echo, "output");

return stat;
} 


template <typename real_t, typename real_in_t=real_t>
status_t interpolate3D(
real_t out[]
, real_space::grid_t const & go
, real_in_t const in[]
, real_space::grid_t const & gi
, int const echo=0 
) {
status_t stat(0);
for (int d = 0; d < 3; ++d) {
assert(go.boundary_condition(d) == gi.boundary_condition(d));
} 

view3D<real_in_t const> ti(in, gi('y'), gi('x')); 
view3D<real_t>     tx(gi('z'), gi('y'), go('x')); 
for (int z = 0; z < gi('z'); ++z) {
for (int y = 0; y < gi('y'); ++y) {
stat += linear_interpolation(tx(z,y), go('x'),
ti(z,y), gi('x'), 1, gi.boundary_condition('x'), echo*(z == 0)*(y == 0));
} 
} 

size_t const nx = go('x');
view2D<real_t> ty(gi('z'), go('y')*nx); 
for (int z = 0; z < gi('z'); ++z) {
stat += linear_interpolation(ty[z], go('y'),
tx(z,0), gi('y'), nx, gi.boundary_condition('y'), echo*(z == 0));
} 

size_t const nyx = go('y')*go('x');
stat += linear_interpolation(out, go('z'),
ty.data(), gi('z'), nyx, gi.boundary_condition('z'), echo);

return stat;
} 


status_t all_tests(int const echo=0); 

} 

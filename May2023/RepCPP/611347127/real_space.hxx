#pragma once

#include <cstdio> 
#include <cstdint> 
#include <algorithm> 
#include <cmath> 
#include <cassert> 

#include "inline_math.hxx" 
#include "simple_math.hxx" 
#include "constants.hxx" 
#include "bessel_transform.hxx" 
#include "recorded_warnings.hxx" 
#include "boundary_condition.hxx" 

#include "status.hxx" 

namespace real_space {

int constexpr debug = 0;

class grid_t {
private:
uint32_t dims[4]; 
int8_t bc[3]; 
public:
double h[3], inv_h[3]; 
double cell[3][4]; 

grid_t(void) : dims{0,0,0,1}, bc{0,0,0}, h{1,1,1}, inv_h{1,1,1} { set(cell[0], 12, 0.0); } 

grid_t(int const d0, int const d1, int const d2, int const dim_outer=1)
: bc{0,0,0}, h{1,1,1}, inv_h{1,1,1} {
dims[0] = std::max(1, d0); 
dims[1] = std::max(1, d1); 
dims[2] = std::max(1, d2); 
dims[3] = std::max(1, dim_outer);
long const nnumbers = dims[3] * dims[2] * dims[1] * dims[0];
if (nnumbers > 0) {
if (debug) std::printf("# grid with %d x %d x %d * %d = %.6f M numbers\n", 
dims[0], dims[1], dims[2], dims[3], nnumbers*1e-6);
} else {
if (debug) std::printf("# grid invalid: dims={%d, %d, %d,  %d}\n",
dims[0], dims[1], dims[2], dims[3]);
}
set(cell[0], 12, 0.0);
for(int d = 0; d < 3; ++d) cell[d][d] = dims[d]*h[d];
} 

template <typename int_t>
grid_t(int_t const dim[3], int const dim_outer=1) : grid_t(dim[0], dim[1], dim[2], dim_outer) {} 

~grid_t() {
if (debug) std::printf("# release a grid with %d x %d x %d * %d = %.6f M numbers\n",
dims[0], dims[1], dims[2], dims[3], 1e-6*dims[3]*dims[2]*dims[1]*dims[0]);
} 

status_t set_grid_spacing(double const hx, double const hy=-1, double const hz=-1) {
status_t stat(0);
double const h3[3] = {hx, (hy<0)?hx:hy, (hz<0)?hx:hz};
for (int i3 = 0; i3 < 3; ++i3) {
h[i3] = h3[i3]; 
if (h[i3] > 0) {
inv_h[i3] = 1./h[i3]; 
} else {
++stat;
} 
} 
return stat;
} 

status_t set_boundary_conditions(int8_t const bc3[3]) { set(bc, 3, bc3); return 0; }
status_t set_boundary_conditions(int8_t const bcx
, int8_t const bcy=Invalid_Boundary 
, int8_t const bcz=Invalid_Boundary) {
bc[0] = bcx;
bc[1] = (bcy == Invalid_Boundary) ? bcx : bcy;
bc[2] = (bcz == Invalid_Boundary) ? bcx : bcz;
return  (bcx == Invalid_Boundary);
} 

inline int is_Cartesian() const { return 1; } 
inline int is_shifted() const { return 0; } 
inline int volume() const { return simple_math::determinant(
cell[0][0], cell[0][1], cell[0][2],
cell[1][0], cell[1][1], cell[1][2],
cell[2][0], cell[2][1], cell[2][2]); }
inline int operator[] (int const d) const { assert(0 <= d); assert(d < 4); return dims[d]; }
inline int operator() (char const c) const { assert('x' <= (c|32)); assert((c|32) <= 'z'); return dims[(c|32) - 120]; }
inline double dV() const { return is_Cartesian() ? h[0]*h[1]*h[2] : volume()/(dims[0]*dims[1]*dims[2]); } 
inline double grid_spacing(int const d) const { assert(0 >= d); assert(d < 3); return h[d]; } 
inline double const * grid_spacings() const { return h; } 
inline double smallest_grid_spacing() const { return std::min(std::min(h[0], h[1]), h[2]); }
inline size_t all() const { return ((size_t(dims[3]) * dims[2]) * dims[1]) * dims[0]; }
inline int8_t boundary_condition(int  const d) const { assert(0 <= d); assert(d < 3); return bc[d]; }
inline int8_t boundary_condition(char const c) const { return boundary_condition((c|32) - 120); }
inline int8_t const * boundary_conditions() const { return bc; }
inline int number_of_boundary_conditions(int const bc_ref=Periodic_Boundary) const { 
return (bc_ref == bc[0]) + (bc_ref == bc[1]) + (bc_ref == bc[2]); };
}; 



template <typename real_t>
status_t add_function(
real_t values[] 
, grid_t const & g 
, double const r2coeff[] 
, int const ncoeff 
, float const hcoeff 
, double *added=nullptr 
, double const center[3]=nullptr 
, double const factor=1 
, float const r_cut=-1 
) {
status_t stat(0);
double c[3] = {0,0,0}; if (center) set(c, 3, center);
double const r_max = std::sqrt((ncoeff - 1.)/hcoeff); 
double const rcut = (-1 == r_cut) ? r_max : std::min(double(r_cut), r_max);
double const r2cut = rcut*rcut;
assert(hcoeff*r2cut < ncoeff);
assert(g.is_Cartesian());
int imn[3], imx[3];
#ifdef DEBUG
size_t nwindow{1};
#endif 
for (int d = 0; d < 3; ++d) {
imn[d] = std::max(0, int(std::floor((c[d] - rcut)*g.inv_h[d])));
imx[d] = std::min(   int(std::ceil ((c[d] + rcut)*g.inv_h[d])), g[d] - 1);
#ifdef DEBUG
std::printf("# %s window %c = %d elements from %d to %d\n", __func__, 'x'+d, imx[d] + 1 - imn[d], imn[d], imx[d]);
nwindow *= std::max(0, imx[d] + 1 - imn[d]);
#endif 
} 
assert(hcoeff > 0);
double added_charge{0}; 
size_t modified{0}, out_of_range{0};
for (            int iz = imn[2]; iz <= imx[2]; ++iz) {  double const vz = iz*g.h[2] - c[2], vz2 = vz*vz;
for (        int iy = imn[1]; iy <= imx[1]; ++iy) {  double const vy = iy*g.h[1] - c[1], vy2 = vy*vy;
if (vz2 + vy2 < r2cut) {
for (int ix = imn[0]; ix <= imx[0]; ++ix) {  double const vx = ix*g.h[0] - c[0], vx2 = vx*vx;
double const r2 = vz2 + vy2 + vx2;
if (r2 < r2cut) {
int const ir2 = int(hcoeff*r2);
if (ir2 < ncoeff) {
int const izyx = (iz*g('y') + iy)*g('x') + ix;
double const w8 = hcoeff*r2 - ir2; 
int const ir2p1 = ir2 + 1;
auto const value_to_add = (r2coeff[ir2] * (1 - w8)
+ ((ir2p1 < ncoeff) ? r2coeff[ir2p1] : 0)*w8);
values[izyx] += factor*value_to_add;
added_charge += factor*value_to_add;
++modified;
#if 0
#endif 
} else {
++out_of_range;
} 
} 
} 
} 
} 
} 
if (added) *added = added_charge * g.dV(); 
#ifdef DEBUG
std::printf("# %s modified %.3f k inside a window of %.3f k on a grid of %.3f k grid values.\n", 
__func__, modified*1e-3, nwindow*1e-3, g('x')*g('y')*g('z')*1e-3); 
#endif 
if (out_of_range > 0) {
stat += 0 < warn("Found %ld entries out of range of the radial function!\n", out_of_range);
} 
return stat;
} 

template <typename real_t>
status_t Bessel_projection(
double q_coeff[] 
, int const nq 
, float const dq 
, real_t const values[] 
, grid_t const & g 
, double const center[3]=nullptr 
, double const factor=1 
, float const r_cut=10 
) {
double c[3] = {0,0,0}; if (center) set(c, 3, center);
double const rcut = r_cut;
double const r2cut = rcut*rcut; 
int imn[3], imx[3];
for (int d = 0; d < 3; ++d) {
imn[d] = std::max(0, int(std::floor((c[d] - rcut)*g.inv_h[d])));
imx[d] = std::min(   int(std::ceil ((c[d] + rcut)*g.inv_h[d])), g[d] - 1);
} 
set(q_coeff, nq, 0.0); 
for (            int iz = imn[2]; iz <= imx[2]; ++iz) {  double const vz = iz*g.h[2] - c[2], vz2 = vz*vz;
for (        int iy = imn[1]; iy <= imx[1]; ++iy) {  double const vy = iy*g.h[1] - c[1], vy2 = vy*vy;
if (vz2 + vy2 < r2cut) {
for (int ix = imn[0]; ix <= imx[0]; ++ix) {  double const vx = ix*g.h[0] - c[0], vx2 = vx*vx;
double const r2 = vz2 + vy2 + vx2;
if (r2 < r2cut) {
int const ixyz = (iz*g('y') + iy)*g('x') + ix;
double const r = std::sqrt(r2);
double const val = double(values[ixyz]);
for (int iq = 0; iq < nq; ++iq) {
double const q = iq*dq;
double const x = q*r;
double const j0 = bessel_transform::Bessel_j0(x);
q_coeff[iq] += val * j0;
} 
} 
} 
} 
} 
} 
double const sqrt2pi = std::sqrt(2./constants::pi); 
scale(q_coeff, nq, g.dV()*factor*sqrt2pi); 
return 0; 
} 








#ifdef NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline status_t test_create_and_destroy(int const echo=9) {
int const dims[] = {10, 20, 30};
auto gp = new grid_t(dims); 
gp->~grid_t(); 
return 0;
} 

inline status_t test_add_function(int const echo=9) {
if (echo > 0) std::printf("\n# %s\n", __func__);
int const dims[] = {32, 31, 30};
grid_t g(dims);
g.set_grid_spacing(0.333);
double const cnt[] = {g[0]*.42*g.h[0],
g[1]*.51*g.h[1],
g[2]*.60*g.h[2]}; 
int const nr2 = 1 << 11;
float const rcut = 4, inv_hr2 = nr2/(rcut*rcut);
double const hr2 = 1./inv_hr2;
double r2c[nr2], rad_integral{0};
if (echo > 4) std::printf("\n# values on the radial grid\n");
for (int ir2 = 0; ir2 < nr2; ++ir2) { 
double const r2 = ir2*hr2, r = std::sqrt(r2);
r2c[ir2] = std::exp(-r2); 
if (echo > 4) std::printf("%g %g\n", r, r2c[ir2]); 
rad_integral += r2c[ir2] * r;
} 
rad_integral *= 2*constants::pi/inv_hr2;

if (echo > 2) std::printf("\n# add_function()\n\n");
double added{0};
std::vector<double> values(g.all(), 0.0);
add_function(values.data(), g, r2c, nr2, inv_hr2, &added, cnt);
if (echo > 6) std::printf("\n# non-zero values on the Cartesian grid (sum = %g)\n", added);
double xyz_integral{0};
for (        int iz = 0; iz < g('z'); ++iz) {  double const vz = iz*g.h[2] - cnt[2];
for (    int iy = 0; iy < g('y'); ++iy) {  double const vy = iy*g.h[1] - cnt[1];
for (int ix = 0; ix < g('x'); ++ix) {  double const vx = ix*g.h[0] - cnt[0];
auto const ixyz = (iz*g('y') + iy)*g('x') + ix;
auto const val = values[ixyz];
if (0 != val) {
if (echo > 6) std::printf("%g %g\n", std::sqrt(vz*vz + vy*vy + vx*vx), val); 
xyz_integral += val;
} 
} 
} 
} 
xyz_integral *= g.dV(); 
auto const diff = xyz_integral - rad_integral;
if (echo > 1) std::printf("# grid integral = %g  radial integral = %g  difference = %.1e (%.3f %%)\n",
xyz_integral, rad_integral, diff, 100*diff/rad_integral);
return std::abs(diff/rad_integral) > 4e-4;
} 

inline status_t all_tests(int const echo=0) {
status_t stat(0);
stat += test_create_and_destroy(echo);
stat += test_add_function(echo);
return stat;
} 

#endif 

} 

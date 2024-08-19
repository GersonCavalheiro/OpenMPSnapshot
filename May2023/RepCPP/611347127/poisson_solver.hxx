#pragma once

#include <cstdio> 

#include "status.hxx" 
#include "real_space.hxx" 
#include "display_units.h" 

namespace poisson_solver {

status_t solve(
double Ves[] 
, double const rho[] 
, real_space::grid_t const & g 
, char const method='m' 
, int const echo=0 
, double const Bessel_center[]=nullptr
); 

inline char solver_method(char const *const method) { return *method; } 

#ifdef DEVEL

template <typename real_t>
void print_direct_projection(
real_t const array[]
, real_space::grid_t const & g 
, double const factor=1
, double const *center=nullptr
, int const echo=1 
) {
int constexpr X=0, Y=1, Z=2;

double cnt[3] = {.5*(g[X] - 1)*g.h[X], .5*(g[Y] - 1)*g.h[Y], .5*(g[Z] - 1)*g.h[Z]};
if (nullptr != center) set(cnt, 3, center); 

if (echo > 0) std::printf("# projection center (relative to grid point (0,0,0) is"
" %g %g %g spacings\n", cnt[X]*g.inv_h[X], cnt[Y]*g.inv_h[Y], cnt[Z]*g.inv_h[Z]);
for (int iz = 0; iz < g[Z]; ++iz) {
double const z = iz*g.h[Z] - cnt[Z], z2 = z*z;
for (int iy = 0; iy < g[Y]; ++iy) {
double const y = iy*g.h[Y] - cnt[Y], y2 = y*y; 
for (int ix = 0; ix < g[X]; ++ix) {
double const x = ix*g.h[X] - cnt[X], x2 = x*x;
double const r = std::sqrt(x2 + y2 + z2);
int const izyx = (iz*g[Y] + iy)*g[X] + ix;
if (echo > 0) std::printf("%g %g\n", r*Ang, array[izyx]*factor);
} 
} 
} 
if (echo > 0) std::printf("# radii in %s\n\n", _Ang);
} 

#endif 

status_t all_tests(int const echo=0); 

} 

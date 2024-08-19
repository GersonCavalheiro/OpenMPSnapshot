#pragma once

#include <cassert> 
#include <vector> 

#include "status.hxx" 
#include "data_view.hxx" 
#include "sho_tools.hxx" 

namespace sho_potential {

status_t load_local_potential(
std::vector<double> & vtot 
, int dims[3] 
, char const *filename 
, int const echo=0 
); 

status_t normalize_potential_coefficients(
double coeff[] 
, int const numax 
, double const sigma 
, int const echo 
); 

template <typename complex_t, typename phase_t>
status_t potential_matrix(
view2D<complex_t> & Vmat 
, view4D<double> const & t1D 
, double const Vcoeff[] 
, int const numax_m 
, int const numax_i
, int const numax_j
, phase_t const phase=1 
, int const dir01=1 
) {
assert( t1D.stride()  >= sho_tools::n1HO(numax_j) );
assert( t1D.dim1()    >= sho_tools::n1HO(numax_i) );
assert( t1D.dim2()    >= sho_tools::n1HO(numax_m) );
assert( Vmat.stride() >= sho_tools::nSHO(numax_j) );

int mzyx{0}; 
for     (int mu = 0; mu <= numax_m; ++mu) { 
for   (int mz = 0; mz <= mu;      ++mz) {
for (int mx = 0; mx <= mu - mz; ++mx) {
int const my = mu - mz - mx;

auto const phase_Vcoeff_m = phase * Vcoeff[mzyx];

int izyx{0};
for     (int iz = 0; iz <= numax_i;           ++iz) {
for   (int iy = 0; iy <= numax_i - iz;      ++iy) {
for (int ix = 0; ix <= numax_i - iz - iy; ++ix) {

int jzyx{0};
for     (int jz = 0; jz <= numax_j;           ++jz) {  auto const tz   = t1D(dir01*2,mz,iz,jz);
for   (int jy = 0; jy <= numax_j - jz;      ++jy) {  auto const tyz  = t1D(dir01*1,my,iy,jy) * tz;
for (int jx = 0; jx <= numax_j - jz - jy; ++jx) {  auto const txyz = t1D(      0,mx,ix,jx) * tyz;

Vmat(izyx,jzyx) += phase_Vcoeff_m * txyz;

++jzyx;
} 
} 
} 
assert( sho_tools::nSHO(numax_j) == jzyx );

++izyx;
} 
} 
} 
assert( sho_tools::nSHO(numax_i) == izyx );

++mzyx;
} 
} 
} 
assert( sho_tools::nSHO(numax_m) == mzyx );

return 0;
} 

status_t all_tests(int const echo=0); 

} 

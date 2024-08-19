#pragma once

#include <cstdio> 
#include <cstdint> 
#include <complex> 

#include "status.hxx" 

#include "sho_tools.hxx" 
#include "real_space.hxx" 
#include "hermite_polynomial.hxx" 
#include "inline_math.hxx" 
#include "constants.hxx" 
#include "sho_unitary.hxx" 


namespace sho_projection {

template <typename real_t>
inline real_t truncation_radius(real_t const sigma, int const numax=-1) { return 9*sigma; }

template <typename complex_t, int PROJECT0_OR_ADD1> inline
status_t _sho_project_or_add(
complex_t coeff[] 
, int const numax 
, double const center[3] 
, double const sigma
, complex_t values[] 
, real_space::grid_t const &g 
, int const echo=0 
) {
using real_t = decltype(std::real(complex_t(1))); 

auto const rcut = truncation_radius(sigma, numax);
assert(sigma > 0);
double const sigma_inv = 1./sigma;
int off[3], end[3], num[3];
for (int d = 0; d < 3; ++d) {
if (echo > 9) std::printf("# %c-direction: center= %g, rcut= %g", 120+d, center[d]*g.inv_h[d], rcut*g.inv_h[d]);
off[d] = std::ceil((center[d] - rcut)*g.inv_h[d]);
end[d] = std::ceil((center[d] + rcut)*g.inv_h[d]);
if (echo > 9) std::printf(", prelim limits [%d, %d)", off[d], end[d]);
off[d] = std::max(off[d], 0); 
end[d] = std::min(end[d], g[d]); 
if (echo > 9) std::printf(", limits [%d, %d)\n", off[d], end[d]);
num[d] = std::max(0, end[d] - off[d]);
} 
auto const nvolume = (size_t(num[0]) * num[1]) * num[2];
if ((nvolume < 1) && (echo < 7)) return 0; 
if (echo > 2) std::printf("# %s on rectangular sub-domain x:[%d, %d) y:[%d, %d) y:[%d, %d) = %d * %d * %d = %ld points\n",
(0 == PROJECT0_OR_ADD1)?"project":"add", off[0], end[0], off[1], end[1], off[2], end[2],
num[0], num[1], num[2], nvolume);

int const nSHO = sho_tools::nSHO(numax);
if (0 == PROJECT0_OR_ADD1) set(coeff, nSHO, complex_t(0));

if (nvolume < 1) return 0; 


int const M = sho_tools::n1HO(numax);
std::vector<real_t> H1d[3];
for (int dir = 0; dir < 3; ++dir) {
H1d[dir] = std::vector<real_t>(num[dir]*M); 
auto const h1d = H1d[dir].data();

double const grid_spacing = g.h[dir];
if (echo > 55) std::printf("\n# Hermite polynomials for %c-direction:\n", 'x' + dir);
for (int ii = 0; ii < num[dir]; ++ii) {
int const ix = ii + off[dir]; 
real_t const x = (ix*grid_spacing - center[dir])*sigma_inv;
Gauss_Hermite_polynomials(h1d + ii*M, x, numax);
#ifdef DEVEL
if (echo > 55) {
std::printf("%g\t", x);
for (int nu = 0; nu <= numax; ++nu) {
std::printf(" %11.6f", h1d[ii*M + nu]);
} 
std::printf("\n");
} 
#endif 
} 
} 

#ifdef DEVEL
if (1 == PROJECT0_OR_ADD1) {
if (echo > 6) {
std::printf("# addition coefficients ");
for (int iSHO = 0; iSHO < nSHO; ++iSHO) {
std::printf(" %g", std::real(coeff[iSHO]));
} 
std::printf("\n\n");
} 
} 
#endif 

for (        int iz = 0; iz < num[2]; ++iz) {
for (    int iy = 0; iy < num[1]; ++iy) {
for (int ix = 0; ix < num[0]; ++ix) {
int const ixyz = ((iz + off[2])*g('y') + (iy + off[1]))*g('x') + (ix + off[0]);

complex_t val(0);
if (0 == PROJECT0_OR_ADD1) {
val = values[ixyz]; 
} 
if (true) {
int iSHO{0};
for (int nz = 0; nz <= numax; ++nz) {                    auto const H1d_z = H1d[2][iz*M + nz];
for (int ny = 0; ny <= numax - nz; ++ny) {           auto const H1d_y = H1d[1][iy*M + ny];
for (int nx = 0; nx <= numax - nz - ny; ++nx) {  auto const H1d_x = H1d[0][ix*M + nx];
auto const H3d = H1d_z * H1d_y * H1d_x;
if (1 == PROJECT0_OR_ADD1) {
val += coeff[iSHO] * H3d; 
} else {
coeff[iSHO] += val * H3d; 
}
++iSHO; 
} 
} 
} 
assert( nSHO == iSHO );
} 
if (1 == PROJECT0_OR_ADD1) {
values[ixyz] += val; 
} 

} 
} 
} 

if (0 == PROJECT0_OR_ADD1) scale(coeff, nSHO, complex_t(g.dV())); 

#ifdef DEVEL
if (0 == PROJECT0_OR_ADD1) {
if (echo > 6) {
std::printf("# projection coefficients ");
for (int iSHO = 0; iSHO < nSHO; ++iSHO) {
std::printf(" %g", std::real(coeff[iSHO]));
} 
std::printf("\n\n");
} 
} 
#endif 

return 0; 
} 


template <typename complex_t>
status_t sho_project( 
complex_t coeff[] 
, int const numax 
, double const center[3] 
, double const sigma 
, complex_t const values[] 
, real_space::grid_t const &g 
, int const echo=0 
) {
return _sho_project_or_add<complex_t,0>(coeff, numax, center, sigma, (complex_t*)values, g, echo); 
} 

template <typename complex_t>
status_t sho_add( 
complex_t values[] 
, real_space::grid_t const &g 
, complex_t const coeff[] 
, int const numax 
, double const center[3] 
, double const sigma 
, int const echo=0 
) {
return _sho_project_or_add<complex_t,1>((complex_t*)coeff, numax, center, sigma, values, g, echo); 
} 


inline double sho_1D_prefactor(int const nu, double const sigma) {
return std::sqrt( ( 1 << nu ) / ( constants::sqrtpi * sigma * factorial(nu) ) ); 
} 

inline double sho_prefactor(int const nx, int const ny, int const nz, double const sigma) {
return std::sqrt(   double(1 << nx)*double(1 << ny)*double(1 << nz) /
(  factorial(nx) * factorial(ny) * factorial(nz) * pow3(constants::sqrtpi * sigma) ) );
} 

template <typename real_t>
std::vector<real_t> get_sho_prefactors(
int const numax
, double const sigma
) {
int const nSHO = sho_tools::nSHO(numax);
std::vector<real_t> f(nSHO);
int iSHO{0};
for (int nz = 0; nz <= numax; ++nz) {                    auto const fz = sho_1D_prefactor(nz, sigma);
for (int ny = 0; ny <= numax - nz; ++ny) {           auto const fy = sho_1D_prefactor(ny, sigma);
for (int nx = 0; nx <= numax - nz - ny; ++nx) {  auto const fx = sho_1D_prefactor(nx, sigma);
f[iSHO] = fx * fy * fz;
++iSHO;
} 
} 
} 
assert(nSHO == iSHO);
return f;
} 


template <typename real_t> inline
status_t renormalize_coefficients(
real_t out[] 
, real_t const in[] 
, int const numax
, double const sigma
) {
int iSHO{0};
for (int nz = 0; nz <= numax; ++nz) {                    auto const fz = sho_1D_prefactor(nz, sigma);
for (int ny = 0; ny <= numax - nz; ++ny) {           auto const fy = sho_1D_prefactor(ny, sigma);
for (int nx = 0; nx <= numax - nz - ny; ++nx) {  auto const fx = sho_1D_prefactor(nx, sigma);
auto const f = fx * fy * fz;
out[iSHO] = in[iSHO] * f;
++iSHO;
} 
} 
} 
assert( sho_tools::nSHO(numax) == iSHO );
return 0;
} 


inline double radial_L1_prefactor(int const ell, double const sigma) {
return std::sqrt(2) / ( constants::sqrtpi * std::pow(sigma, 2*ell + 3) * factorial<2>(2*ell + 1) );
} 

inline double radial_L2_prefactor(int const ell, double const sigma, int const nrn=0) {
assert( 0 == nrn ); 
double const fm2 = std::pow(sigma, 2*ell + 3) * factorial<2>(2*ell + 1) * constants::sqrtpi * std::pow(0.5, 2 + ell);
return 1/std::sqrt(fm2);
} 

template <typename real_t> inline
status_t renormalize_radial_coeff(
real_t out[] 
, real_t const in[] 
, int const ellmax
, double const sigma
) {
for (int ell = 0; ell <= ellmax; ++ell) { 
auto const pfc = radial_L1_prefactor(ell, sigma) / radial_L2_prefactor(ell, sigma); 
for (int emm = -ell; emm <= ell; ++emm) { 
int const lm = sho_tools::lm_index(ell, emm);
out[lm] = pfc * in[lm]; 
} 
} 
return 0;
} 

inline
status_t renormalize_electrostatics(
double vlm[] 
, double const vzyx[] 
, int const ellmax
, double const sigma
, sho_unitary::Unitary_SHO_Transform const & u
, int const echo=0
) {
status_t stat(0);

int const nSHO = sho_tools::nSHO(ellmax);
std::vector<double> vzyx_nrm(nSHO, 0.0); 

stat += renormalize_coefficients(vzyx_nrm.data(), vzyx, ellmax, sigma);

std::vector<double> vnlm(nSHO, 0.0); 
stat += u.transform_vector(vnlm.data(), sho_tools::order_nlm,
vzyx_nrm.data(), sho_tools::order_zyx, ellmax, 0);

stat += renormalize_radial_coeff(vlm, vnlm.data(), ellmax, sigma);

return stat;
} 

inline
status_t denormalize_electrostatics(
double qzyx[] 
, double const qlm[] 
, int const ellmax
, double const sigma
, sho_unitary::Unitary_SHO_Transform const & u
, int const echo=0
) {
status_t stat(0);

int const nSHO = sho_tools::nSHO(ellmax);
std::vector<double> qnlm(nSHO, 0.0); 

stat += renormalize_radial_coeff(qnlm.data(), qlm, ellmax, sigma);

std::vector<double> qzyx_nrm(nSHO, 0.0); 
stat += u.transform_vector(qzyx_nrm.data(), sho_tools::order_zyx,
qnlm.data(), sho_tools::order_nlm, ellmax, 0);

stat += renormalize_coefficients(qzyx, qzyx_nrm.data(), ellmax, sigma);

return stat;
} 

status_t all_tests(int const echo=0); 

} 

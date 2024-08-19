#pragma once

#include <cmath> 

#ifndef NO_UNIT_TESTS
#include <cstdio> 
#include <algorithm> 
#include <cmath> 
#include <cassert> 

#include "constants.hxx" 
#endif 
#include "status.hxx" 


template <typename real_t>
inline void Gauss_Hermite_polynomials(real_t H[], real_t const x, int const numax, real_t const rcut=9) {
real_t const H0 = (x*x < rcut*rcut) ? std::exp(-0.5*x*x) : 0; 

H[0] = H0; 

real_t Hnup1, Hnu{H0}, Hnum1{0};
for (int nu = 0; nu < numax; ++nu) {
real_t const nuhalf = 0.5 * real_t(nu); 
Hnup1 = x * Hnu - nuhalf * Hnum1; 
H[nu + 1] = Hnup1; 
Hnum1 = Hnu; Hnu = Hnup1; 
} 

} 

namespace hermite_polynomial {

#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

template <typename real_t>
inline status_t test_Gauss_Hermite_polynomials(int const echo=4, double const threshold=7e-15) {
int constexpr nx = 1 << 13;
int constexpr numax = 7, M = 1 + numax;
double const hg = 1.25/(1 << 9);
if (echo > 3) std::printf("# %s: numax = %d\n", __func__, numax);
real_t hh[M][M]; 
double prefactor[M];
for (int i = 0; i < M; ++i) {
for (int j = 0; j < M; ++j) {
hh[i][j] = 0; 
} 
prefactor[i] = 1;
} 

double max_dev[] = {0, 0}; 
for (int twice = 0; twice < 2; ++twice) {

for (int ix = 0; ix < nx; ++ix) {
real_t const x = (ix - .5*(nx - 1))*hg;
real_t hp[M];
Gauss_Hermite_polynomials(hp, x, numax);
for (int i = 0; i < M; ++i) {
for (int j = 0; j < M; ++j) {
hh[i][j] += hp[i]*hp[j]; 
} 
} 
if (echo > 6) {
std::printf("%g  ", x);
for (int nu = 0; nu <= numax; ++nu) {
std::printf(" %g", hp[nu]*prefactor[nu]);
} 
std::printf("\n");
} 
} 
if (echo > 5) std::printf("\n\n");

if (0 == twice) { 

for (int ij = 0; ij < M*M; ++ij) { hh[0][ij] *= hg; } 

double const pi_factor = 1./std::sqrt(constants::pi);
for (int nu = 0; nu <= numax; ++nu) {
double const diag = hh[nu][nu];
if (echo > 4) std::printf("# nu = %d norm^2 = %g --> %g\n", nu, diag, diag*pi_factor*(1 << nu));
assert(diag > 0);
auto const scal = 1./std::sqrt(diag);
for (int nup = 0; nup <= numax; ++nup) {
hh[nu][nup] *= scal;
hh[nup][nu] *= scal;
int const d = (nu == nup); 
max_dev[d] = std::max(max_dev[d], std::abs(double(hh[nu][nup]) - d));
} 
assert(std::abs(hh[nu][nu] - 1) < 1e-7);
prefactor[nu] = scal;
} 

if (echo > 5) {
std::printf("# %s  %d x %d overlap matrix, deviation from unity\n", __func__, M, M);
for (int nu = 0; nu <= numax; ++nu) {
std::printf("# nu = %d ", nu);
for (int nup = 0; nup <= numax; ++nup) {
std::printf("%9.1e", hh[nu][nup] - (nu == nup));
} 
std::printf("\n");
} 
std::printf("\n\n");
} 

if (echo > 1) std::printf("# %s<%s> largest deviation from unit matrix is %.1e and %.1e on the"
" diagonal\n", __func__, (sizeof(real_t) == 8)?"double":"float", max_dev[0], max_dev[1]);

} 
} 

return (max_dev[0] + max_dev[1] > threshold);
} 

inline status_t all_tests(int const echo=0) {
if (echo > 0) std::printf("\n# %s: %s\n\n", __FILE__, __func__);
status_t stat(0);
stat += test_Gauss_Hermite_polynomials<float>(echo, 3.5e-6);
stat += test_Gauss_Hermite_polynomials<double>(echo);
return stat;
} 

#endif 

} 

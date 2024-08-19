#pragma once

#ifndef NO_UNIT_TESTS
#include <cstdio> 
#include <algorithm> 
#endif
#include <cmath> 
#include <cassert> 

#include "constants.hxx" 
#include "status.hxx" 
#include "quantum_numbers.h" 

namespace sho_radial {

template <typename real_t>
void radial_eigenstates(
real_t poly[] 
, int const nrn 
, int const ell 
, real_t const factor=1 
) {


poly[0] = factor;
for (int k = 0; k < nrn; ++k) {
poly[k + 1] = (poly[k]*(k - nrn)*2)/real_t((k + 1)*(2*ell + 2*k + 3));
} 

} 

template <typename real_t>
real_t exponential_integral_k(int const k) {
if (0 == k) return real_t(0.5*constants::sqrtpi);
if (1 == k) return real_t(0.5); 
assert(k > 0);
return real_t(0.5)*(k - 1)*exponential_integral_k<real_t>(k - 2); 
} 

template <typename real_t>
real_t radial_normalization(real_t const coeff[], int const nrn, int const ell) {

auto const prod = new real_t[2*nrn + 1]; 
for (int p = 0; p < 2*nrn + 1; ++p) {
prod[p] = 0;
} 
for (int k = 0; k <= nrn; ++k) {
for (int p = 0; p <= nrn; ++p) {
prod[k + p] += coeff[k]*coeff[p]; 
} 
} 

real_t exp_int_k = exponential_integral_k<real_t>(2*ell + 2);
real_t norm{0};
for (int p = 0; p <= 2*nrn; ++p) { 
norm += prod[p]*exp_int_k;
exp_int_k *= (p + ell + 1.5); 
} 
delete[] prod;
return 1./std::sqrt(norm); 
} 

template <typename real_t>
real_t radial_normalization(int const nrn, int const ell) {
auto const coeff = new real_t[nrn + 1]; 
radial_eigenstates(coeff, nrn, ell); 
auto const result = radial_normalization(coeff, nrn, ell);
delete[] coeff;
return result;
} 

template <typename real_t>
real_t expand_poly(real_t const coeff[], int const ncoeff, double const x) {
real_t value{0};
double xpow{1};
for (int i = 0; i < ncoeff; ++i) {
value += coeff[i] * xpow;
xpow *= x;
} 
return value;
} 








#ifdef NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

template <typename real_t>
real_t numerical_norm(real_t const c0[], int const nrn0,
real_t const c1[], int const nrn1, int const ell) {
double constexpr dr = 1./(1 << 12), rmax = 12.;
int const nr = rmax / dr;
real_t norm{0};
for (int ir = 0; ir < nr; ++ir) {
double const r = (ir - .5)*dr;
double const r2 = r*r;
double const Gauss = std::exp(-r2); 
double const f0 = expand_poly(c0, 1 + nrn0, r2);
double const f1 = expand_poly(c1, 1 + nrn1, r2);
double const r2pow = std::pow(r2, 1 + ell); 
norm += Gauss * r2pow * f0 * f1;
} 
norm *= dr; 
return norm;
} 

inline status_t test_orthonormality(int const echo=1) {
int constexpr numax = 9;
int constexpr n = (numax*(numax + 4) + 4)/4; 
if (echo > 1) std::printf("# %s  numax= %d has %d different radial SHO states\n", __func__, numax, n);

double c[n][8]; 

ell_QN_t ell_list[n];
enn_QN_t nrn_list[n];
double   fac_list[n];

int i = 0;
for (int ene = 0; ene <= numax; ++ene) { 
for (int nrn = ene/2; nrn >= 0; --nrn) { 
int const ell = ene - 2*nrn;

ell_list[i] = ell;
nrn_list[i] = nrn;
radial_eigenstates(c[i], nrn, ell);
double const fac = radial_normalization(c[i], nrn, ell);
assert(std::abs(fac - radial_normalization<double>(nrn, ell)) < 1e-12); 
fac_list[i] = fac;
if (echo > 2) std::printf("# %s %3d state  nrn= %d  ell= %d  factor= %g\n", __func__, i, nrn, ell, fac);
radial_eigenstates(c[i], nrn, ell, fac_list[i]); 

++i; 
} 
} 
assert(n == i); 

double dev[] = {0, 0};
for (int i = 0; i < n; ++i) {
int const ell = ell_list[i];
for (int j = i; j >= 0; --j) { 
if (ell == ell_list[j]) {
auto const delta_ij = numerical_norm(c[i], nrn_list[i], c[j], nrn_list[j], ell);
int const i_equals_j = (i == j);
if (false) {
if (i != j) {
auto const delta_ji = numerical_norm(c[j], nrn_list[j], c[i], nrn_list[i], ell);
assert(std::abs(delta_ji - delta_ij) < 1e-15);
} 
} 
if (echo > 14 - i_equals_j) std::printf("%g ", delta_ij - i_equals_j);
dev[i_equals_j] = std::max(dev[i_equals_j], std::abs(delta_ij - i_equals_j));
} 
} 
if (echo > 13) std::printf("\n");
} 
if (echo > 0) std::printf("# normalization of radial SHO eigenfunctions differs by %g from unity\n", dev[1]); 
if (echo > 0) std::printf("# orthogonality of radial SHO eigenfunctions differs by %g from zero\n",  dev[0]); 

if (echo > 7) { 
std::printf("\n## radial SHO functions: ln= ");
for (int i = 0; i < n; ++i) {
std::printf(" %d%d", ell_list[i],nrn_list[i]);
} 
for (int ir = 0; ir < 666; ++ir) {
double const r = ir*0.01, r2 = r*r; 
double const Gauss = std::exp(-0.5*r2);
std::printf("\n%.2f", r);
for (int i = 0; i < n; ++i) {
std::printf(" %g", expand_poly(c[i], 1 + nrn_list[i], r2)*Gauss);
} 
} 
std::printf("\n\n");
} 

return (dev[0] + dev[1] > 5e-11);
} 

inline status_t all_tests(int const echo=0) {
status_t stat(0);
stat += test_orthonormality(echo);
return stat;
} 

#endif 

} 

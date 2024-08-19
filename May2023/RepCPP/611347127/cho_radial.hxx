#pragma once

#include <cmath> 
#include <cassert> 
#include <cstdint> 
#include <vector> 
#ifndef NO_UNIT_TESTS
#include <cstdio> 
#include <algorithm> 
#endif

#define HAS_BMP_EXPORT
#ifdef  HAS_BMP_EXPORT
#include "bitmap.hxx" 

template <typename real_t>
inline void Hermite_polynomials(real_t H[], real_t const x, int const numax, real_t const rcut=9) {
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

#endif 


namespace cho_radial {



template <typename int_t>
inline int_t constexpr nCHO(int_t const numax) { return ((numax + 1)*(numax + 2))/2; }

template <typename int_t>
inline int_t constexpr nCHO_radial(int_t const numax) { return (numax*(numax + 4) + 4)/4; }

template <typename real_t>
void radial_eigenstates(
real_t poly[] 
, int const nrn 
, int const ell 
, real_t const factor=1 
) {


poly[0] = factor;
for (int k = 0; k < nrn; ++k) {
poly[k + 1] = (poly[k]*(k - nrn))/((k + 1.)*(k + 1 + ell));
} 

} 

double exponential_integral_k(int const k) {
if (0 == k) return 0.5;
assert(k > 0);
return k*exponential_integral_k(k - 1); 
} 

template <typename real_t>
real_t inner_product(real_t const coeff0[], int const nrn0, real_t const coeff1[], int const nrn1, int const ell) {
std::vector<double> prod(nrn0 + nrn1 + 1, 0.0); 
for (int p0 = 0; p0 <= nrn0; ++p0) {
for (int p1 = 0; p1 <= nrn1; ++p1) {
prod[p0 + p1] += double(coeff0[p0])*coeff1[p1]; 
} 
} 

double dot{0};
double exp_int_k = exponential_integral_k(ell);
for (int p = 0; p <= nrn0 + nrn1; ++p) { 
assert(exp_int_k == exponential_integral_k(ell + p)); 
dot += prod[p]*exp_int_k;
exp_int_k *= (ell + p + 1); 
} 
return dot; 
} 

template <typename real_t>
real_t radial_normalization(real_t const coeff[], int const nrn, int const ell) {
auto const norm = inner_product(coeff, nrn, coeff, nrn, ell);
assert(norm > 0);
return 1./std::sqrt(norm); 
} 

template <typename real_t=double>
real_t radial_normalization(int const nrn, int const ell) {
std::vector<real_t> coeff(nrn + 1);  
radial_eigenstates(coeff.data(), nrn, ell); 
auto const result = radial_normalization(coeff.data(), nrn, ell);
return result;
} 

template <typename real_t>
real_t expand_poly(real_t const coeff[], int const ncoeff, double const x) {
real_t value{0};
double xpow{1};
for (int p = 0; p < ncoeff; ++p) {
value += coeff[p] * xpow;
xpow *= x;
} 
return value;
} 

#ifdef NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

char const *const ellchar = "spdfghijklmno";

template <typename real_t>
real_t numerical_inner_product(real_t const c0[], int const nrn0, 
real_t const c1[], int const nrn1, int const ell) {
double constexpr dr = 1./(1 << 12), rmax = 12.;
int const nr = rmax / dr;
real_t dot{0};
for (int ir = 0; ir < nr; ++ir) {
double const r = (ir - .5)*dr;
double const r2 = r*r;
double const Gauss = std::exp(-r2); 
double const f0 = expand_poly(c0, 1 + nrn0, r2);
double const f1 = expand_poly(c1, 1 + nrn1, r2);
double const r2pow = r*std::pow(r2, ell); 
dot += Gauss * r2pow * f0 * f1;
} 
dot *= dr; 
return dot;
} 

template <int numax=9>
inline status_t test_orthonormality(int const echo=1) {
int constexpr n = nCHO_radial(numax);
if (echo > 1) std::printf("# %s  numax= %d has %d different radial CHO states\n", __func__, numax, n);

double c[n][8]; 

int8_t   ell_list[n];
uint8_t  nrn_list[n];
double   fac_list[n];

int i{0};
for (int ell = numax; ell >= 0; --ell) { 
for (int nrn = 0; nrn <= (numax - ell)/2; ++nrn) {

ell_list[i] = ell;
nrn_list[i] = nrn;
radial_eigenstates(c[i], nrn, ell);
double const fac = radial_normalization(c[i], nrn, ell);
assert(std::abs(fac - radial_normalization(nrn, ell)) < 1e-12); 
fac_list[i] = fac;
if (echo > 4) std::printf("# %s %3d state  nrn= %d  ell= %d  factor= %g\n", __func__, i, nrn, ell, fac);
radial_eigenstates(c[i], nrn, ell, fac); 

++i; 
} 
} 
assert(n == i); 

double deviation[][2] = {{0, 0}, {0, 0}};
for (int method = 0; method < 2; ++method) { 
double *const dev = deviation[method];
for (int i = 0; i < n; ++i) {
int const ell = ell_list[i];
for (int j = i; j >= 0; --j) { 
if (ell == ell_list[j]) {
auto const delta_ij = method ? inner_product(c[i], nrn_list[i], c[j], nrn_list[j], ell):
numerical_inner_product(c[i], nrn_list[i], c[j], nrn_list[j], ell);
int const i_equals_j = (i == j);
if (true) {
if (i != j) {
auto const delta_ji = method ? inner_product(c[j], nrn_list[j], c[i], nrn_list[i], ell):
numerical_inner_product(c[j], nrn_list[j], c[i], nrn_list[i], ell);
auto const asymmetry = delta_ji - delta_ij;
if (std::abs(asymmetry) > 1e-12) {
if (echo > 0) std::printf("# asymmetry of radial CHO eigenfunctions by %.1e between %c%d and %c%d method=%d\n", 
asymmetry, ellchar[ell_list[i]], nrn_list[i], ellchar[ell_list[j]], nrn_list[j], method);
assert(std::abs(asymmetry) < 1e-9);
} 
} 
} 
if (echo > 5 - i_equals_j) std::printf("%g ", delta_ij - i_equals_j);
dev[i_equals_j] = std::max(dev[i_equals_j], std::abs(delta_ij - i_equals_j));
} 
} 
if (echo > 4) std::printf("\n");
} 
for (int diag = 0; diag < 2*(echo > 3); ++diag) {
std::printf("# normalization of radial CHO eigenfunctions differs by %g from %s (method=%s)\n",
dev[diag], diag?"unity":"zero ", method?"analytical":"numerical");
} 
} 
return (deviation[1][0] + deviation[1][1] > 5e-11);
} 


template <int numax=7>
inline status_t test_Gram_Schmidt(int const echo=1) {

double dev{0};
double c[1 + numax][1 + numax];
for (int ell = 0; ell <= numax; ++ell) {
if (echo > 3) std::printf("# %s ell=%d\n", __func__, ell);

for (int i = 0; i < (1 + numax)*(1 + numax); ++i) {
c[0][i] = 0; 
} 
double fac[1 + numax]; 
for (int nrn = 0; nrn <= (numax - ell)/2; ++nrn) {
c[nrn][nrn] = (nrn % 2)?-1:1; 
for (int jrn = 0; jrn < nrn; ++jrn) {
auto const a = inner_product(c[nrn], nrn, c[jrn], jrn, ell);
auto const d = inner_product(c[jrn], jrn, c[jrn], jrn, ell);
assert(d > 0);
auto const a_over_d = a/d;
for (int p = 0; p <= jrn; ++p) {
c[nrn][p] -= a_over_d*c[jrn][p]; 
} 
auto const check = inner_product(c[nrn], nrn, c[jrn], jrn, ell);
dev = std::max(dev, std::abs(check));
if (echo > 11) std::printf("# overlap between %c%d and %c%d is %g\n",
ellchar[ell], nrn, ellchar[ell], jrn, check);
} 

auto const f = radial_normalization(c[nrn], nrn, ell);
fac[nrn] = f; 

if (echo > 5) {
std::printf("# %c%d-poly", ellchar[ell], nrn);
for (int p = nrn; p >= 0; --p) {
std::printf(" + %g*r^%d", f*c[nrn][p], ell+2*p);
} 
std::printf("\n");
} 

if (1) { 
double coeff[1 + numax];
radial_eigenstates(coeff, nrn, ell);
auto const fc = radial_normalization(coeff, nrn, ell);
if (echo > 5) std::printf("# %c%d-poly", ellchar[ell], nrn);
double diff{0};
for (int p = nrn; p >= 0; --p) {
if (echo > 5) std::printf(" + %g*r^%d", fc*coeff[p], ell+2*p);
diff = std::max(diff, std::abs(f*c[nrn][p] - fc*coeff[p]));
} 
if (echo > 5) std::printf("  largest difference %.1e\n", diff);
} 

} 

if (echo > 8) { 
bool const with_exp = true,
with_ell = true;
std::printf("\n# Radial CHO functions for ell=%d\n", ell);
for (int ir = 0; ir <= 1000; ++ir) {
auto const r = ir*0.01, r2 = r*r;
auto const Gauss = with_exp ? std::exp(-0.5*r2) : 1;
auto const r_ell = with_ell ? std::pow(r, ell)  : 1;
std::printf("%g", r);
for (int nrn = 0; nrn <= (numax - ell)/2; ++nrn) {
std::printf("\t%g", fac[nrn]*r_ell*expand_poly(c[nrn], 1 + nrn, r2)*Gauss);
} 
std::printf("\n");
} 
std::printf("\n");
} 

} 

if (echo > 1) std::printf("# %s largest deviation is %.1e\n", __func__, dev);
return (dev > 1e-12);
} 

#ifdef HAS_BMP_EXPORT
template <typename real_t=double> 
inline status_t test_radial_and_Cartesian_image(int const echo=0) {
status_t stat(0);
int constexpr Resolution = 1024;
int constexpr Lcut = 4;
double const sigma_inv = 0.01, center = 0.5*(Resolution - 1)*sigma_inv;

double Hermite[Resolution][Lcut];
{
double Hermite_norm[Lcut];
double constexpr sqrtpi = 1.77245385090551602729816748334115;
double H_norm2{sqrtpi};
for (int n = 0; n < Lcut; ++n) {
Hermite_norm[n] = 1./std::sqrt(H_norm2);
H_norm2 *= 0.5*(n + 1);
} 

std::vector<double> Hermite_norm_check(Lcut, 0.0);
for (int ix = 0; ix < Resolution; ++ix) {
double const x = ix*sigma_inv - center;
Hermite_polynomials(Hermite[ix], x, Lcut - 1);
for (int n = 0; n < Lcut; ++n) {
Hermite[ix][n] *= Hermite_norm[n];
Hermite_norm_check[n] += Hermite[ix][n]*Hermite[ix][n];
} 
} 
if (echo > 3) {
std::printf("# Hermite norm check ");
for (int n = 0; n < Lcut; ++n) {
std::printf(" %g", Hermite_norm_check[n]*sigma_inv);
} 
std::printf("\n");
} 
}

auto const data = new real_t[2][Resolution][Resolution][4];
for (int nu = 0; nu < Lcut; ++nu) {

for (int nx = 0; nx <= nu; ++nx) { 
int const ny = nu - nx;        

int const m = 2*nx - nu;
int const ell = std::abs(m);
int const nrn = (nu - ell)/2;
double Radial_coeff_ell_nrn[Lcut];
{
radial_eigenstates(Radial_coeff_ell_nrn, nrn, ell);
auto const f = radial_normalization(Radial_coeff_ell_nrn, nrn, ell);
radial_eigenstates(Radial_coeff_ell_nrn, nrn, ell, f);
}

for (int i = 0; i < 2*Resolution*Resolution*4; ++i) data[0][0][0][i] = 0;
char basename[2][96];
std::snprintf(basename[0], 95, "cartesian_%d_%d", nx, ny);
std::snprintf(basename[1], 95, "radial_%d_%d", m, nrn);
for (int iy = 0; iy < Resolution; ++iy) {
double const y = iy*sigma_inv - center;
for (int ix = 0; ix < Resolution; ++ix) {
double const x = ix*sigma_inv - center;
double const Gaussian  = Hermite[ix][0]  * Hermite[iy][0];
double const Cartesian = Hermite[ix][nx] * Hermite[iy][ny];
double const r2 = x*x + y*y;
double const radial_function = expand_poly(Radial_coeff_ell_nrn, 1 + nrn, r2) * std::pow(r2, 0.5*ell);
double const theta = std::atan2(y, x);
double const Radial = Gaussian * radial_function * ((m < 0)?std::sin(ell*theta):std::cos(ell*theta));
data[1][iy][ix][ Radial    > 0     ] = std::abs(Radial);
data[1][iy][ix][(Radial    > 0) + 1] = std::abs(Radial);
data[0][iy][ix][ Cartesian > 0     ] = std::abs(Cartesian);
data[0][iy][ix][(Cartesian > 0) + 1] = std::abs(Cartesian);
} 
} 

for (int cr = 0; cr < 2; ++cr) { 
real_t maxi{0};
for (int i = 0; i < Resolution*Resolution*4; ++i) {
maxi = std::max(maxi, data[cr][0][0][i]);
} 
real_t const maxi_inv = 1./maxi;
for (int i = 0; i < Resolution*Resolution*4; ++i) {
data[cr][0][0][i] *= maxi_inv; 
data[cr][0][0][i] = 1 - data[cr][0][0][i]; 
} 

stat += bitmap::write_bmp_file(basename[cr], data[cr][0][0], Resolution, Resolution);
} 

} 
} 

double constexpr pi = 3.14159265358979323846; 
{
auto const maxangle = -90; 
int const nx = 3, ny = 0; 
int const mx = 1, my = 2; 
int const nframes = -1;
for (int iframe = 0; iframe <= nframes; ++iframe) {
char basename[96]; std::snprintf(basename, 95, "frame%5.5d", iframe);
auto const angle = (iframe*maxangle*pi)/(180.*nframes);
auto const fn = std::cos(angle),
fm = std::sin(angle);

for (int iy = 0; iy < Resolution; ++iy) {
for (int ix = 0; ix < Resolution; ++ix) {
for (int rgb = 0; rgb < 4; ++rgb) data[0][iy][ix][rgb] = 0;
auto const Cartesian = fn * Hermite[ix][nx] * Hermite[iy][ny]
+ fm * Hermite[ix][mx] * Hermite[iy][my];
data[0][iy][ix][ Cartesian > 0     ] = std::abs(Cartesian);
data[0][iy][ix][(Cartesian > 0) + 1] = std::abs(Cartesian);
} 
} 

real_t maxi{0};
for (int i = 0; i < Resolution*Resolution*4; ++i) {
maxi = std::max(maxi, data[0][0][0][i]);
} 
real_t const maxi_inv = 1./maxi;
for (int i = 0; i < Resolution*Resolution*4; ++i) {
data[0][0][0][i] *= maxi_inv; 
data[0][0][0][i] = 1 - data[0][0][0][i]; 
} 

stat += bitmap::write_bmp_file(basename, data[0][0][0], Resolution, Resolution);
} 
} 

delete[] data;
return stat;
} 
#endif 


inline status_t all_tests(int const echo=0) {
status_t stat(0);
stat += test_orthonormality(echo);
stat += test_Gram_Schmidt(echo);
#ifdef HAS_BMP_EXPORT
stat += bitmap::test_image(echo);
stat += test_radial_and_Cartesian_image(echo);
#endif 
return stat;
} 

#endif 

} 

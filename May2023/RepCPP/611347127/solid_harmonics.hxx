#pragma once

#include <cstdio> 
#include <cmath> 
#include <cassert> 
#include <vector> 

#include "constants.hxx" 
#include "status.hxx" 

namespace solid_harmonics {

double constexpr pi = constants::pi;
double constexpr Y00inv = 3.5449077018110318, 
Y00 = .28209479177387817; 

template <typename real_t>
void rlXlm_implementation(
real_t xlm[]
, int const ellmax
, real_t const cth, real_t const sth
, real_t const cph, real_t const sph
, real_t const r2=1
, bool const cos_trick=true
) {

static std::vector<real_t> xnorm;
static int ellmaxd = -1; 

if (ellmax > ellmaxd) {
#ifdef DEBUG
std::printf("# %s resize table of normalization constants from %d to %d\n", __func__, (1 + ellmaxd)*(1 + ellmaxd), (1 + ellmax)*(1 + ellmax));
#endif 
xnorm.resize((1 + ellmax)*(1 + ellmax));

{ 
double const fpi = 4.0*pi;
for (int l = 0; l <= ellmax; ++l) {
int const lm0 = l*l + l;
double const a = std::sqrt((2*l + 1.)/fpi);
double cd{1}, sgn{-1};
xnorm[lm0] = a;
for (int m = 1; m <= l; ++m) {
cd /= ((l + 1. - m)*(l + m));
auto const xn = a*std::sqrt(2*cd); 
xnorm[lm0 + m] = xn;
xnorm[lm0 - m] = xn * sgn; 
sgn = -sgn; 
} 
} 
} 
ellmaxd = ellmax; 
} else if (ellmax < 0) {
ellmaxd = -1; 
xnorm.resize(0); 
}
if (ellmax < 0) return;

int const S = 1 + ellmax; 
std::vector<real_t> p(S*S); 

real_t fac{1};
for (int m = 0; m < ellmax; ++m) {
fac *= (1 - 2*m);
p[m     + S*m] = fac;
p[m + 1 + S*m] = (m + 1 + m)*cth*fac;
for (int l = m + 2; l <= ellmax; ++l) {
p[l + S*m] = ((2*l - 1)*cth*p[l - 1 + S*m] - (l + m - 1)*r2*p[l - 2 + S*m])/(l - m);
} 
fac *= sth;
} 
p[ellmax + S*ellmax] = (1 - 2*ellmax)*fac;

std::vector<real_t> cs(S, real_t(1));
std::vector<real_t> sn(S, real_t(0));
if (cos_trick) {
if (ellmax > 0) {
cs[1] = cph; sn[1] = sph;
auto const cph2 = 2*cph;
for (int m = 2; m <= ellmax; ++m) {
sn[m] = cph2*sn[m - 1] - sn[m - 2];
cs[m] = cph2*cs[m - 1] - cs[m - 2];
} 
} 
} else {
for (int m = 1; m <= ellmax; ++m) {
sn[m] = cph*sn[m - 1] + cs[m - 1]*sph;
cs[m] = cph*cs[m - 1] - sn[m - 1]*sph;
} 
} 

for (int m = 0; m <= ellmax; ++m) {
for (int l = m; l <= ellmax; ++l) {
int const lm0 = l*l + l;
real_t const Plm = p[l + S*m];
xlm[lm0 + m] = xnorm[lm0 + m]*Plm*sn[m]; 
xlm[lm0 - m] = xnorm[lm0 + m]*Plm*cs[m]; 
} 
} 

} 

template <typename real_t>
void Xlm(real_t xlm[], int const ellmax, double const theta, double const phi) {
rlXlm_implementation(xlm, ellmax, std::cos(theta), std::sin(theta), std::cos(phi), std::sin(phi));
} 

template <typename real_t, typename vector_real_t>
void Xlm(real_t xlm[], int const ellmax, vector_real_t const v[3]) {
real_t constexpr small = 1e-12;
auto const x = v[0], y = v[1], z = v[2];
auto const xy2 = x*x + y*y;
auto const r   = std::sqrt(xy2 + z*z);
auto const rxy = std::sqrt(xy2);

real_t cth{1}, sth{0};
if (r > small) {
cth = z*(1/r);
sth = rxy*(1/r);
}
real_t cph{1}, sph{0};
if (rxy > small) {
cph = x*(1/rxy);
sph = y*(1/rxy);
}
rlXlm_implementation(xlm, ellmax, cth, sth, cph, sph);
} 

template <typename real_t, typename vector_real_t>
void rlXlm(real_t xlm[], int const ellmax, vector_real_t const x, vector_real_t const y, vector_real_t const z) {
real_t const r2 = x*x + y*y + z*z;
rlXlm_implementation(xlm, ellmax, z, real_t(1), x, y, r2, false);
} 

template <typename real_t, typename vector_real_t>
void rlXlm(real_t xlm[], int const ellmax, vector_real_t const v[3]) {
rlXlm(xlm, ellmax, v[0], v[1], v[2]);
} 

template <typename real_t=double>
void cleanup(int const echo=0) {
if (echo > 5) std::printf("# %s %s<%s>: free internal memory\n",
__FILE__, __func__, (sizeof(real_t) == 4)?"float":"double");
real_t z{0};
rlXlm_implementation(&z, -1, z, z, z, z);
} 


inline int lm_index(int const ell, int const emm) { return ell*ell + ell + emm; }

inline int find_ell(int const lm) { int lp1{0}; while (lp1*lp1 <= lm) ++lp1; return lp1 - 1; } 
inline int find_emm(int const lm, int const ell) { return lm - lm_index(ell, 0); }
inline int find_emm(int const lm) { return find_emm(lm, find_ell(lm)); }

#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline status_t test_indices(int const echo=0) {
status_t stat(0);
for (int lm = -3; lm < 64; ++lm) {
int const ell = find_ell(lm),
emm = find_emm(lm, ell);
if (echo > 4) std::printf("# %s    lm=%d -> ell=%d emm=%d\n", __FILE__, lm, ell, emm);
stat += (lm_index(ell, emm) != lm);
} 
return stat;
} 

inline status_t test_Y00_inverse(int const echo=0) {
return ( Y00 * Y00inv != 1.0 ); 
} 

template <typename real_t>
inline status_t test_memory_cleanup(int const echo=0, int const ellmax=9) {
status_t stat(0);
real_t const vec[] = {1, 2, 3};
for (int ell = 0; ell <= ellmax; ++ell) {
std::vector<real_t> xlm((1 + ell)*(1 + ell));
rlXlm(xlm.data(), ell, vec);
cleanup<real_t>(echo); 
} 
return stat;
} 

inline status_t all_tests(int const echo=0) {
if (echo > 0) std::printf("\n# %s (for tests of the orthogonality run --test angular_grid)\n", __FILE__);
status_t stat(0);
stat += test_Y00_inverse(echo);
stat += test_indices(echo);
stat += test_memory_cleanup<float>(echo);
stat += test_memory_cleanup<double>(echo);
return stat;
} 

#endif 

} 

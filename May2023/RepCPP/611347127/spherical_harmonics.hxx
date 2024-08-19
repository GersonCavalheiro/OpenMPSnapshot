#pragma once

#include <cstdio> 
#include <complex> 
#include <cmath> 
#include <vector> 

#include "constants.hxx" 
#include "inline_math.hxx" 

namespace spherical_harmonics {

template <typename real_t>
inline void Ylm(
std::complex<real_t> ylm[]
, int const ellmax
, double const v[3]
) {

real_t constexpr small = 1e-12;


static std::vector<real_t> ynorm;
static int ellmaxd = -1; 

if (ellmax > ellmaxd) {
#ifdef DEBUG
std::printf("# %s: resize table of normalization constants from %d to %d\n",
__func__, pow2(1 + ellmaxd), pow2(1 + ellmax));
#endif 
ynorm.resize(pow2(1 + ellmax));

{ 
double const fpi = 4*constants::pi; 
for (int l = 0; l <= ellmax; ++l) {
int const lm0 = l*l + l;
double const a = std::sqrt((2*l + 1.)/fpi);
double cd{1}, sgn{-1};
ynorm[lm0] = a;
for (int m = 1; m <= l; ++m) {
cd /= ((l + 1. - m)*(l + m));
auto const yn = a*std::sqrt(cd);
ynorm[lm0 + m] = yn;
ynorm[lm0 - m] = yn * sgn;
sgn = -sgn; 
} 
} 
} 
ellmaxd = ellmax; 
} else if (ellmax < 0) {
ellmaxd = -1; 
ynorm.resize(0); 
}
if (ellmax < 0) return;

auto const x = v[0], y = v[1], z = v[2];
auto const xy2 = x*x + y*y;
auto const r = std::sqrt(xy2 + z*z);
auto const rxy = std::sqrt(xy2);

real_t cth{1}, sth{0};
if (r > small) {
cth = z/r;
sth = rxy/r;
}
real_t cph{1}, sph{0};
if (rxy > small) {
cph = x/rxy;
sph = y/rxy;
}

int const S = (1 + ellmax); 
std::vector<real_t> p((1 + ellmax)*S);

real_t fac{1};
for (int m = 0; m < ellmax; ++m) {
fac *= (1 - 2*m);
p[m     + S*m] = fac;
p[m + 1 + S*m] = (m + 1 + m)*cth*fac;
for (int l = m + 2; l <= ellmax; ++l) {
p[l + S*m] = ((2*l - 1)*cth*p[l - 1 + S*m] - (l + m - 1)*p[l - 2 + S*m])/real_t(l - m);
} 
fac *= sth;
} 
p[ellmax + S*ellmax] = (1 - 2*ellmax)*fac;

std::vector<real_t> c(1 + ellmax, real_t(1)),
s(1 + ellmax, real_t(0));
if (ellmax > 0) {
c[1] = cph; s[1] = sph;
auto const cph2 = 2*cph;
for (int m = 2; m <= ellmax; ++m) {
s[m] = cph2*s[m - 1] - s[m - 2];
c[m] = cph2*c[m - 1] - c[m - 2];
} 
} 

for (int m = 0; m <= ellmax; ++m) {
for (int l = m; l <= ellmax; ++l) {
int const lm0 = l*l + l;
auto const ylms = p[l + S*m]*std::complex<real_t>(c[m], s[m]);
ylm[lm0 + m] = ynorm[lm0 + m]*ylms;
ylm[lm0 - m] = ynorm[lm0 - m]*std::conj(ylms);
} 
} 

return;
} 

template <typename real_t=double>
void cleanup(int const echo=0) {
if (echo > 5) std::printf("# %s %s<%s>: free internal memory\n",
__FILE__, __func__, (sizeof(real_t) == 4)?"float":"double");
std::complex<real_t> z{0};
double v[3];
Ylm(&z, -1, v);
} 









#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

template <typename real_t>
inline status_t test_memory_cleanup(int const echo=0, int const ellmax=9) {
status_t stat(0);
double const vec[] = {1, 2, 3};
for (int ell = 0; ell <= ellmax; ++ell) {
std::vector<std::complex<real_t>> ylm(pow2(1 + ell));
Ylm(ylm.data(), ell, vec);
cleanup<real_t>(echo);
} 
return stat;
} 

inline status_t all_tests(int const echo=0) {
if (echo > 0) std::printf("\n# %s %s\n", __FILE__, __func__);
status_t stat(0);
stat += test_memory_cleanup<float>(echo);
stat += test_memory_cleanup<double>(echo);
return stat;
} 

#endif 

} 

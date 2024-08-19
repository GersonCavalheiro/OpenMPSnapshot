#pragma once

#include <cstdint> 
#include <cassert> 
#include <cstdio> 
#include <cmath> 
#include <vector> 
#include <complex> 

#include "status.hxx" 
#include "sho_tools.hxx" 
#include "green_parallel.hxx" 

#include "control.hxx" 

namespace green_projection {

inline float Hermite_polynomials_1D( 
double (*const __restrict__ H1D)[3][4] 
, float  (*const __restrict__ xi_squared)[4] 
, int    const lmax
, double const xyza[3+1] 
, float  const xyzc[3]   
, double const hxyz[3+1] 
) {
double const R2_projection = pow2(double(hxyz[3])); 

if (lmax < 0) return R2_projection; 

double const sigma_inverse = xyza[3]*xyza[3]; 
for (int idir = 0; idir < 3; ++idir) {
double const grid_spacing  = hxyz[idir]; 
double const cube_position = xyzc[idir]; 
double const atom_position = xyza[idir]; 
for (int i4 = 0; i4 < 4; ++i4) {

double const xi = sigma_inverse*(grid_spacing*(cube_position*4 + i4 + 0.5) - atom_position); 
double const xi2 = xi*xi; 
xi_squared[idir][i4] = xi2; 
double const H0 = (xi2 < R2_projection) ? std::exp(-0.5*xi2) : 0; 

H1D[0][idir][i4] = H0; 

double Hnp1, Hn{H0}, Hnm1{0};
for (int nu = 0; nu < lmax; ++nu) {
double const nuhalf = 0.5 * nu; 
Hnp1 = xi * Hn - nuhalf * Hnm1; 
H1D[nu + 1][idir][i4] = Hnp1; 
Hnm1 = Hn; Hn = Hnp1; 
} 


} 
} 

return R2_projection;
} 

template <int R1C2=2, int Noco=1>
std::vector<std::vector<double>> SHOprj( 
double   const (*const __restrict__ pGreen)[R1C2][Noco][Noco*64] 
, double   const (*const __restrict__ AtomImagePos)[3+1] 
, int8_t   const (*const __restrict__ AtomImageLmax) 
, uint32_t const (*const __restrict__ AtomImageStarts) 
, uint32_t const (*const __restrict__ AtomImageIndex) 
, double   const (*const __restrict__ AtomImagePhase)[4] 
, uint32_t const nAtomImages
, int8_t   const (*const __restrict__ AtomLmax)
, uint32_t const nAtoms
, float    const (*const __restrict__ colCubePos)[3+1] 
, double   const (*const __restrict__ hGrid) 
, uint32_t const nrhs
, int      const echo=0 
) {
if (echo > 0) std::printf("# %s for %d atoms\n", __func__, nAtoms);
std::vector<std::vector<double>> pGp(nAtoms); 
for (int iatom = 0; iatom < nAtoms; ++iatom) {
int const lmax = AtomLmax[iatom];
int const nSHO = sho_tools::nSHO(lmax);
pGp[iatom].resize(nSHO*nSHO*Noco*Noco, 0.0); 
} 

typedef std::complex<double> complex_t;
complex_t constexpr zero = complex_t(0, 0);

for (int iai = 0; iai < nAtomImages; ++iai) {

int const lmax = AtomLmax[iai];
int constexpr Lmax = 7;
if (lmax > Lmax) std::printf("# %s Error: lmax= %d but max. Lmax= %d, iai=%d\n", __func__, lmax, Lmax, iai);
assert(lmax <= Lmax);

auto const a0 = AtomImageStarts[iai];
int const nSHO = sho_tools::nSHO(lmax);
int const n2HO = sho_tools::n2HO(lmax);

std::vector<complex_t> piGp(nSHO*nSHO*Noco*Noco, zero);

double H1D[1 + Lmax][3][4]; 
float     xi_squared[3][4]; 

for (int spin = 0; spin < Noco; ++spin) {
for (int spjn = 0; spjn < Noco; ++spjn) {
for (int jsho = 0; jsho < nSHO; ++jsho) {

std::vector<complex_t> czyx(nSHO, zero);

for (uint32_t irhs = 0; irhs < nrhs; ++irhs) {

auto const R2_proj = Hermite_polynomials_1D(H1D, xi_squared, lmax, AtomImagePos[iai], colCubePos[irhs], hGrid);

for (int z4 = 0; z4 < 4; ++z4) { 
auto const d2z = xi_squared[2][z4] - R2_proj;
if (d2z < 0) {
std::vector<complex_t> byx(n2HO, zero);
for (int y4 = 0; y4 < 4; ++y4) { 
auto const d2yz = xi_squared[1][y4] + d2z;
if (d2yz < 0) {
std::vector<complex_t> ax(lmax + 1, zero);
for (int x4 = 0; x4 < 4; ++x4) { 
auto const d2xyz = xi_squared[0][x4] + d2yz;
if (d2xyz < 0) {
int const xyz = (z4*4 + y4)*4 + x4;
int constexpr RealPart = 0, ImagPart= R1C2 - 1;
auto const pG = complex_t(pGreen[(a0 + jsho)*nrhs + irhs][RealPart][spin][spjn*64 + xyz],  
pGreen[(a0 + jsho)*nrhs + irhs][ImagPart][spin][spjn*64 + xyz]); 
for (int ix = 0; ix <= lmax; ++ix) { 
ax[ix] += pG * H1D[ix][0][x4];
} 
} 
} 
for (int iy = 0; iy <= lmax; ++iy) { 
int const iyx0 = (iy*(2*lmax + 3 - iy)) >> 1;
auto const Hy = H1D[iy][1][y4]; 
for (int ix = 0; ix <= lmax - iy; ++ix) { 
byx[iyx0 + ix] += ax[ix] * Hy;
} 
} 
} 
} 
for (int isho = 0, iz = 0; iz <= lmax; ++iz) { 
auto const Hz = H1D[iz][2][z4]; 
for (int iy = 0; iy <= lmax - iz; ++iy) { 
int const iyx0 = (iy*(2*lmax + 3 - iy)) >> 1;
for (int ix = 0; ix <= lmax - iz - iy; ++ix) { 
czyx[isho] += byx[iyx0 + ix] * Hz; 
++isho;
} 
} 
} 
} 
} 

} 

for (int isho = 0; isho < nSHO; ++isho) {
piGp[((isho*nSHO + jsho)*Noco + spin)*Noco + spjn] = czyx[isho];
} 

} 
} 
} 

auto const iatom = AtomImageIndex[iai];
assert(nSHO*nSHO*Noco*Noco == pGp[iatom].size() && "Inconsistency between AtomLmax[:] and AtomImageLmax[AtomImageIndex[:]]");
auto const phase = AtomImagePhase[iai];
auto const ph = complex_t(phase[0], phase[1]); 
for (int ij = 0; ij < nSHO*nSHO*Noco*Noco; ++ij) {
complex_t const c = piGp[ij] * ph; 
pGp[iatom][ij] += c.imag(); 
} 

} 

return pGp;
} 

#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline status_t all_tests(int const echo=0) {
status_t stat(0);
return stat;
} 

#endif 

} 

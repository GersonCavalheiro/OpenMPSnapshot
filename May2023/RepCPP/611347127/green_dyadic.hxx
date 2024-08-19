#pragma once

#include <cstdio> 
#include <cstdint> 
#include <cassert> 
#include <cmath> 
#include <vector> 

#include "status.hxx" 
#include "green_memory.hxx" 
#include "green_sparse.hxx" 
#include "inline_math.hxx" 
#include "sho_tools.hxx" 
#include "constants.hxx" 
#include "green_parallel.hxx" 
#include "green_projection.hxx" 

#ifndef NO_UNIT_TESTS
#include "control.hxx" 
#endif 

#ifndef HAS_NO_CUDA
#include <cuda/std/complex> 
#define std__complex cuda::std::complex
#else
#include <complex> 
#define std__complex std::complex
#endif 

namespace green_dyadic {


template <typename real_t>
float __host__ __device__
Hermite_polynomials_1D(
real_t (*const __restrict__ H1D)[3][4] 
, float  (*const __restrict__ xi_squared)[4] 
, int    const ivec 
, int    const lmax
, double const xyza[3+1] 
, float  const xyzc[3]   
, double const hxyz[3+1] 
)
{

double const R2_projection = pow2(double(hxyz[3])); 

if (lmax < 0) return R2_projection; 

int constexpr l2b = 2;
int constexpr n4 = 1 << l2b; 
assert(4 == n4);
if (ivec < 3*n4) { 


int const idir = ivec >> l2b; 
assert(idir >= 0); assert(idir < 3);
int const i4   = ivec - n4*idir; 
assert(i4 >= 0); assert(i4 < n4);

double const grid_spacing  = hxyz[idir]; 
double const cube_position = xyzc[idir]; 
double const atom_position = xyza[idir]; 
double const sigma_inverse = xyza[3]*xyza[3]; 

double const xi = sigma_inverse*(grid_spacing*(cube_position*n4 + i4 + 0.5) - atom_position); 
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


return R2_projection; 
} 


int constexpr Lmax_default=7; 

template <typename real_t, int R1C2=2, int Noco=1, int Lmax=Lmax_default>
void __global__ SHOprj( 
#ifdef HAS_NO_CUDA
dim3 const & gridDim, dim3 const & blockDim,
#endif 
real_t         (*const __restrict__ Cpr)[R1C2][Noco   ][Noco*64] 
, real_t   const (*const __restrict__ Psi)[R1C2][Noco*64][Noco*64] 
, green_sparse::sparse_t<> const (*const __restrict__ sparse)
, double   const (*const __restrict__ AtomPos)[3+1] 
, int8_t   const (*const __restrict__ AtomLmax) 
, uint32_t const (*const __restrict__ AtomStarts) 
, uint32_t const (*const __restrict__ irow_of_inzb) 
, float    const (*const __restrict__ CubePos)[3+1] 
, double   const (*const __restrict__ hGrid) 
)
{
assert(1       ==  gridDim.z);
assert(Noco*64 == blockDim.x);
assert(Noco    == blockDim.y);
assert(R1C2    == blockDim.z);

int const nrhs   = gridDim.y;

__shared__ double hgrid[3+1]; 

#ifndef HAS_NO_CUDA
if (threadIdx.x < 4) hgrid[threadIdx.x] = hGrid[threadIdx.x]; 
int const irhs  = blockIdx.y;  
int const iatom = blockIdx.x;  
#else  
set(hgrid, 4, hGrid);
int const natoms = gridDim.x;
for (int irhs  = 0; irhs < nrhs; ++irhs)
for (int iatom = 0; iatom < natoms; ++iatom)
#endif 
{ 

auto const bsr_of_iatom = sparse[irhs].rowStart();

#ifndef HAS_NO_CUDA
if (bsr_of_iatom[iatom] >= bsr_of_iatom[iatom + 1]) return; 
#endif 

auto const inzb_of_bsr = sparse[irhs].colIndex();

int const lmax = AtomLmax[iatom];
if (lmax > Lmax) std::printf("# %s Error: lmax= %d but max. Lmax= %d, iatom=%d\n", __func__, lmax, Lmax, iatom);
assert(lmax <= Lmax);

__shared__ real_t H1D[Lmax + 1][3][4]; 
__shared__ float     xi_squared[3][4]; 
__shared__ double xyza[3+1]; 
__shared__ float  xyzc[3];   

#ifndef HAS_NO_CUDA
if (threadIdx.x < 4) xyza[threadIdx.x] = AtomPos[iatom][threadIdx.x]; 
int const reim = threadIdx.z; 
int const spin = threadIdx.y; 
int const j    = threadIdx.x; 
#else 
set(xyza, 4, AtomPos[iatom]);
for (int reim = 0; reim < R1C2; ++reim)
for (int spin = 0; spin < Noco; ++spin)
for (int j = 0; j < Noco*64; ++j)
#endif 
{ 

real_t czyx[sho_tools::nSHO(Lmax)]; 
for (int sho = 0; sho < sho_tools::nSHO(Lmax); ++sho) {
czyx[sho] = 0; 
} 

__syncthreads(); 

for (auto bsr = bsr_of_iatom[iatom]; bsr < bsr_of_iatom[iatom + 1]; ++bsr) {

__syncthreads();

auto const inzb = inzb_of_bsr[bsr]; 
auto const irow = irow_of_inzb[inzb]; 

__shared__ float R2_proj;
#ifndef HAS_NO_CUDA
if (threadIdx.x < 3) xyzc[threadIdx.x] = CubePos[irow][threadIdx.x]; 
__syncthreads();


if (0 == threadIdx.y && 0 == threadIdx.z) 
R2_proj = Hermite_polynomials_1D(H1D, xi_squared, j, lmax, xyza, xyzc, hgrid);

__syncthreads();
#else  
set(xyzc, 3, CubePos[irow]);

for (int jj = 0; jj < 12; ++jj) {
R2_proj = Hermite_polynomials_1D(H1D, xi_squared, jj, lmax, xyza, xyzc, hgrid);
} 
#endif 


for (int z = 0; z < 4; ++z) { 
auto const d2z = xi_squared[2][z] - R2_proj;
if (d2z < 0) {

real_t byx[sho_tools::n2HO(Lmax)];
for (int iyx = 0; iyx < sho_tools::n2HO(lmax); ++iyx) { 
byx[iyx] = 0; 
} 

for (int y = 0; y < 4; ++y) { 
auto const d2yz = xi_squared[1][y] + d2z;
if (d2yz < 0) {

real_t ax[Lmax + 1];
for (int ix = 0; ix <= lmax; ++ix) { 
ax[ix] = 0; 
} 

for (int x = 0; x < 4; ++x) { 
auto const d2xyz = xi_squared[0][x] + d2yz;
if (d2xyz < 0) {
int const xyz = (z*4 + y)*4 + x;
real_t const ps = Psi[inzb][reim][spin*64 + xyz][j]; 
for (int ix = 0; ix <= lmax; ++ix) { 
auto const Hx = H1D[ix][0][x]; 
ax[ix] += ps * Hx; 
} 
} 
} 

for (int iy = 0; iy <= lmax; ++iy) { 
int const iyx0 = (iy*(2*lmax + 3 - iy)) >> 1;
auto const Hy = H1D[iy][1][y]; 
for (int ix = 0; ix <= lmax - iy; ++ix) { 
byx[iyx0 + ix] += ax[ix] * Hy; 
} 
} 

} 
} 

for (int sho = 0, iz = 0; iz <= lmax; ++iz) { 
auto const Hz = H1D[iz][2][z]; 
for (int iy = 0; iy <= lmax - iz; ++iy) { 
int const iyx0 = (iy*(2*lmax + 3 - iy)) >> 1;
for (int ix = 0; ix <= lmax - iz - iy; ++ix) { 
czyx[sho] += byx[iyx0 + ix] * Hz; 
++sho;
} 
} 
} 

} 
} 

} 

{ 
auto const a0 = AtomStarts[iatom];
for (int sho = 0; sho < sho_tools::nSHO(lmax); ++sho) {
Cpr[(a0 + sho)*nrhs + irhs][reim][spin][j] = czyx[sho]; 
} 
} 

}} 

} 

template <typename real_t, int R1C2=2, int Noco=1>
void __host__ SHOprj_driver(
real_t         (*const __restrict__ Cpr)[R1C2][Noco   ][Noco*64] 
, real_t   const (*const __restrict__ Psi)[R1C2][Noco*64][Noco*64] 
, double   const (*const __restrict__ AtomPos)[3+1] 
, int8_t   const (*const __restrict__ AtomLmax) 
, uint32_t const (*const __restrict__ AtomStarts) 
, uint32_t const natoms 
, green_sparse::sparse_t<> const (*const __restrict__ sparse)
, uint32_t const (*const __restrict__ RowIndexCube) 
, float    const (*const __restrict__ CubePos)[3+1] 
, double   const (*const __restrict__ hGrid) 
, int      const nrhs 
, int const echo=0
) {
if (natoms*nrhs < 1) return;
dim3 const gridDim(natoms, nrhs, 1), blockDim(Noco*64, Noco, R1C2);
if (echo > 3) std::printf("# %s<%s,R1C2=%d,Noco=%d> <<< {natoms=%d, nrhs=%d, 1}, {%d, Noco=%d, R1C2=%d} >>>\n",
__func__, real_t_name<real_t>(), R1C2, Noco, natoms, nrhs, Noco*64, Noco, R1C2);
SHOprj<real_t,R1C2,Noco> 
#ifndef HAS_NO_CUDA
<<< gridDim, blockDim >>> (
#else  
( gridDim, blockDim,
#endif 
Cpr, Psi, sparse, AtomPos, AtomLmax, AtomStarts, RowIndexCube, CubePos, hGrid);
} 



template <typename real_t, int R1C2=2, int Noco=1, int Lmax=Lmax_default>
void __global__ SHOadd( 
#ifdef HAS_NO_CUDA
dim3 const & gridDim, dim3 const & blockDim,
#endif 
real_t         (*const __restrict__ Psi)[R1C2][Noco*64][Noco*64] 
, real_t   const (*const __restrict__ Cad)[R1C2][Noco   ][Noco*64] 
, uint32_t const (*const __restrict__ bsr_of_inzb)
, uint32_t const (*const __restrict__ iatom_of_bsr)
, uint32_t const (*const __restrict__ irow_of_inzb) 
, uint16_t const (*const __restrict__ irhs_of_inzb) 
, double   const (*const __restrict__ AtomPos)[3+1] 
, int8_t   const (*const __restrict__ AtomLmax) 
, uint32_t const (*const __restrict__ AtomStarts) 
, float    const (*const __restrict__ CubePos)[3+1] 
, double   const (*const __restrict__ hGrid) 
, int      const nrhs 
)
{
assert(1       ==  gridDim.y);
assert(1       ==  gridDim.z);
assert(Noco*64 == blockDim.x);
assert(Noco    == blockDim.y);
assert(R1C2    == blockDim.z);

__shared__ double hgrid[3+1]; 

#ifndef HAS_NO_CUDA
if (threadIdx.x < 4) hgrid[threadIdx.x] = hGrid[threadIdx.x]; 
int const inzb = blockIdx.x;  
#else  
set(hgrid, 4, hGrid);
for (int inzb = 0; inzb < gridDim.x; ++inzb)
#endif 
{ 

#ifndef HAS_NO_CUDA
if (bsr_of_inzb[inzb] >= bsr_of_inzb[inzb + 1]) return; 
#endif 

auto const irow = irow_of_inzb[inzb]; 
auto const irhs = irhs_of_inzb[inzb]; 

__shared__ real_t H1D[1 + Lmax][3][4]; 
__shared__ float     xi_squared[3][4]; 
__shared__ double xyza[3+1]; 
__shared__ float  xyzc[3+1]; 

#ifndef HAS_NO_CUDA
if (threadIdx.x < 4) xyzc[threadIdx.x] = CubePos[irow][threadIdx.x]; 
int const reim = threadIdx.z; 
int const spin = threadIdx.y; 
int const j    = threadIdx.x; 
#else  
set(xyzc, 4, CubePos[irow]);
for (int reim = 0; reim < R1C2; ++reim)
for (int spin = 0; spin < Noco; ++spin)
for (int j = 0; j < Noco*64; ++j)
#endif 
{ 

real_t czyx[4][4][4]; 
for (int z = 0; z < 4; ++z) {
for (int y = 0; y < 4; ++y) {
for (int x = 0; x < 4; ++x) {
czyx[z][y][x] = 0; 
} 
} 
} 


int64_t mask_all{0}; 

for (auto bsr = bsr_of_inzb[inzb]; bsr < bsr_of_inzb[inzb + 1]; ++bsr) {

__syncthreads();
auto const iatom = iatom_of_bsr[bsr];

int const lmax = AtomLmax[iatom];
auto const a0  = AtomStarts[iatom];
assert(lmax <= Lmax);

__shared__ float R2_proj;
#ifndef HAS_NO_CUDA
if (threadIdx.x < 4) xyza[threadIdx.x] = AtomPos[iatom][threadIdx.x]; 

__syncthreads();

if (0 == threadIdx.y && 0 == threadIdx.z) 
R2_proj = Hermite_polynomials_1D(H1D, xi_squared, j, lmax, xyza, xyzc, hgrid);

#else  
set(xyza, 4, AtomPos[iatom]);

for (int jj = 0; jj < 12; ++jj) {
R2_proj = Hermite_polynomials_1D(H1D, xi_squared, jj, lmax, xyza, xyzc, hgrid);
} 
#endif 

int sho{0};
for (int iz = 0; iz <= lmax; ++iz) { 

real_t byx[4][4]; 
for (int y = 0; y < 4; ++y) {
for (int x = 0; x < 4; ++x) {
byx[y][x] = 0; 
} 
} 

for (int iy = 0; iy <= lmax - iz; ++iy) { 

real_t ax[4] = {0, 0, 0, 0}; 

for (int ix = 0; ix <= lmax - iz - iy; ++ix) { 
real_t const ca = Cad[(a0 + sho)*nrhs + irhs][reim][spin][j]; 
++sho;
for (int x = 0; x < 4; ++x) {
real_t const Hx = H1D[ix][0][x]; 
ax[x] += Hx * ca; 
} 
} 

for (int y = 0; y < 4; ++y) {
real_t const Hy = H1D[iy][1][y]; 
for (int x = 0; x < 4; ++x) {
byx[y][x] += Hy * ax[x]; 
} 
} 

} 

int64_t mask_atom{0}; 
for (int z = 0; z < 4; ++z) {
real_t const Hz = H1D[iz][2][z]; 
auto const d2z = xi_squared[2][z] - R2_proj;
if (d2z < 0) {
for (int y = 0; y < 4; ++y) {
auto const d2yz = xi_squared[1][y] + d2z;
if (d2yz < 0) {
for (int x = 0; x < 4; ++x) {
auto const d2xyz = xi_squared[0][x] + d2yz;
if (d2xyz < 0) {
czyx[z][y][x] += Hz * byx[y][x]; 
int const xyz = (z*4 + y)*4 + x;
mask_atom |= (int64_t(1) << xyz);
} 
} 
} 
} 
} 
} 
mask_all |= mask_atom; 

} 

} 

if (mask_all) {
for (int z = 0; z < 4; ++z) {
for (int y = 0; y < 4; ++y) {
for (int x = 0; x < 4; ++x) {
int const xyz = (z*4 + y)*4 + x;
if ((mask_all >> xyz) & 0x1) { 
Psi[inzb][reim][spin*64 + xyz][j] += czyx[z][y][x]; 
} 
} 
} 
} 
} 

}} 

} 


template <typename real_t, int R1C2=2, int Noco=1>
void __host__ SHOadd_driver(
real_t         (*const __restrict__ Psi)[R1C2][Noco*64][Noco*64] 
, real_t   const (*const __restrict__ Cad)[R1C2][Noco   ][Noco*64] 
, double   const (*const __restrict__ AtomPos)[3+1] 
, int8_t   const (*const __restrict__ AtomLmax) 
, uint32_t const (*const __restrict__ AtomStarts) 
, uint32_t const (*const __restrict__ RowStartCubes) 
, uint32_t const (*const __restrict__ ColIndexAtoms) 
, uint32_t const (*const __restrict__ RowIndexCubes) 
, uint16_t const (*const __restrict__ ColIndexCubes) 
, float    const (*const __restrict__ CubePos)[3+1] 
, double   const (*const __restrict__ hGrid) 
, int      const nnzb 
, int      const nrhs 
, int const echo=0
) {
if (nnzb < 1) return;
dim3 const gridDim(nnzb, 1, 1), blockDim(Noco*64, Noco, R1C2);
if (echo > 3) std::printf("# %s<%s,R1C2=%d,Noco=%d> <<< {nnzb=%d, 1, 1}, {%d, Noco=%d, R1C2=%d} >>>\n",
__func__, real_t_name<real_t>(), R1C2, Noco, nnzb, Noco*64, Noco, R1C2);
SHOadd<real_t,R1C2,Noco> 
#ifndef HAS_NO_CUDA
<<< gridDim, blockDim >>> (
#else  
( gridDim, blockDim,
#endif 
Psi, Cad, RowStartCubes, ColIndexAtoms, RowIndexCubes, ColIndexCubes, AtomPos, AtomLmax, AtomStarts, CubePos, hGrid, nrhs);
} 









template <typename real_t, int R1C2=2, int Noco=1, int n64=64>
void __global__ SHOsum( 
#ifdef HAS_NO_CUDA
dim3 const & gridDim, dim3 const & blockDim,
#endif 
double         (*const __restrict__ aac)[R1C2][Noco][Noco*n64] 
, real_t         (*const __restrict__ aic)[R1C2][Noco][Noco*n64] 
, int8_t   const (*const __restrict__ AtomLmax) 
, uint32_t const (*const __restrict__ AtomStarts) 
, uint32_t const (*const __restrict__ AtomImageStarts) 
, double   const (*const __restrict__ AtomImagePhase)[4] 
, uint32_t const (*const __restrict__ bsr_of_iatom) 
, uint32_t const (*const __restrict__ iai_of_bsr)   
, bool     const collect=true 
)
{
assert((1 == Noco && (1 == R1C2 || 2 == R1C2)) || (2 == Noco && 2 == R1C2));
auto const nrhs  = gridDim.y;
assert(1        ==  gridDim.z);
assert(Noco*n64 == blockDim.x);
assert(Noco     == blockDim.y);
assert(1        == blockDim.z);

float const scale_imaginary = collect ? 1 : -1; 

#ifndef HAS_NO_CUDA
auto const irhs  = blockIdx.y;
auto const iatom = blockIdx.x;
#else  
auto const natoms = gridDim.x;
for (int irhs = 0; irhs < nrhs; ++irhs)
for (int iatom = 0; iatom < natoms; ++iatom)
#endif 
{ 

int const a0 = AtomStarts[iatom];
int const lmax = AtomLmax[iatom];
int const nSHO = sho_tools::nSHO(lmax);

#ifndef HAS_NO_CUDA
auto const spin = threadIdx.y;
auto const ivec = threadIdx.x;
#else  
for (int spin = 0; spin < blockDim.y; ++spin)
for (int ivec = 0; ivec < blockDim.x; ++ivec)
#endif 
{ 

int constexpr Re = 0, Im = R1C2 - 1; 
if (collect) {
for (int ai = 0; ai < nSHO; ++ai) {
aac[(a0 + ai)*nrhs + irhs][Re][spin][ivec] = 0.0;
aac[(a0 + ai)*nrhs + irhs][Im][spin][ivec] = 0.0; 
} 
} 

for (auto bsr = bsr_of_iatom[iatom]; bsr < bsr_of_iatom[iatom + 1]; ++bsr) {

auto const iai = iai_of_bsr[bsr]; 

auto const phase = AtomImagePhase[iai];
auto const ph = std__complex<double>(phase[0], phase[1]*scale_imaginary);
auto const i0 = AtomImageStarts[iai];

for (int ai = 0; ai < nSHO; ++ai) {
if (collect) {
if (1 == R1C2) {
aac[(a0 + ai)*nrhs + irhs][Re][0][ivec] += phase[0] *
aic[(i0 + ai)*nrhs + irhs][Re][0][ivec]; 
} else {
auto const c = ph * std__complex<double>(
aic[(i0 + ai)*nrhs + irhs][Re][spin][ivec],
aic[(i0 + ai)*nrhs + irhs][Im][spin][ivec]);
aac[(a0 + ai)*nrhs + irhs][Re][spin][ivec] += c.real();
aac[(a0 + ai)*nrhs + irhs][Im][spin][ivec] += c.imag(); 
} 
} else {
if (1 == R1C2) {
aic[(i0 + ai)*nrhs + irhs][Re][0][ivec] = phase[0] *
aac[(a0 + ai)*nrhs + irhs][Re][0][ivec]; 
} else {
auto const c = ph * std__complex<double>(
aac[(a0 + ai)*nrhs + irhs][Re][spin][ivec],
aac[(a0 + ai)*nrhs + irhs][Im][spin][ivec]);
aic[(i0 + ai)*nrhs + irhs][Re][spin][ivec] = c.real();
aic[(i0 + ai)*nrhs + irhs][Im][spin][ivec] = c.imag(); 
} 
} 
} 

} 
}} 

} 

template <typename real_t, int R1C2=2, int Noco=1, int n64=64>
void __host__ SHOsum_driver(
double         (*const __restrict__ aac)[R1C2][Noco][Noco*n64] 
, real_t         (*const __restrict__ aic)[R1C2][Noco][Noco*n64] 
, int8_t   const (*const __restrict__ AtomLmax) 
, uint32_t const (*const __restrict__ AtomStarts) 
, uint32_t const (*const __restrict__ AtomImageStarts) 
, double   const (*const __restrict__ AtomImagePhase)[4] 
, green_sparse::sparse_t<> const & sparse_SHOsum
, uint32_t const nAtoms 
, int      const nrhs
, bool     const collect=true 
, int      const echo=0
)
{
assert((1 == Noco && (1 == R1C2 || 2 == R1C2)) || (2 == Noco && 2 == R1C2));

dim3 const gridDim(nAtoms, nrhs, 1), blockDim(Noco*n64, Noco, 1);
if (echo > 3) std::printf("# %s<%s,R1C2=%d,Noco=%d> <<< {nAtoms=%d, nrhs=%d, 1}, {%d, Noco=%d, 1} >>>\n",
__func__, real_t_name<real_t>(), R1C2, Noco,  nAtoms, nrhs,  Noco*n64, Noco);
SHOsum<real_t,R1C2,Noco,n64> 
#ifndef HAS_NO_CUDA
<<< gridDim, blockDim >>> (
#else  
( gridDim, blockDim,
#endif 
aac, aic, AtomLmax, AtomStarts, AtomImageStarts, AtomImagePhase,
sparse_SHOsum.rowStart(), sparse_SHOsum.colIndex(), collect);
} 





template <typename real_t, int R1C2=2, int Noco=1, int n64=64>
void __global__ SHOmul( 
#ifdef HAS_NO_CUDA
dim3 const & gridDim, dim3 const & blockDim,
#endif 
real_t         (*const __restrict__ aac)[R1C2][Noco][Noco*n64] 
, real_t   const (*const __restrict__ apc)[R1C2][Noco][Noco*n64] 
, double   const (*const *const __restrict__ AtomMatrices) 
, int8_t   const (*const __restrict__ AtomLmax) 
, uint32_t const (*const __restrict__ AtomStarts) 
)
{
assert((1 == Noco && (1 == R1C2 || 2 == R1C2)) || (2 == Noco && 2 == R1C2));
int const nrhs   = gridDim.y;
assert(1       ==  gridDim.z);
assert(Noco*n64== blockDim.x);
assert(Noco    == blockDim.y);
assert(1       == blockDim.z);

#ifndef HAS_NO_CUDA
int const irhs  = blockIdx.y;
int const iatom = blockIdx.x;
#else  
int const natoms = gridDim.x;
for (int irhs = 0; irhs < nrhs; ++irhs)
for (int iatom = 0; iatom < natoms; ++iatom)
#endif 
{ 

int const a0 = AtomStarts[iatom];
int const lmax = AtomLmax[iatom];
int const nSHO = sho_tools::nSHO(lmax);
auto const AtomMat = AtomMatrices[iatom];

#ifndef HAS_NO_CUDA
int const spin = threadIdx.y;
int const ivec = threadIdx.x;
#else  
for (int spin = 0; spin < blockDim.y; ++spin)
for (int ivec = 0; ivec < blockDim.x; ++ivec)
#endif 
{ 

for (int ai = 0; ai < nSHO; ++ai) {
int constexpr Real = 0;
if (1 == R1C2) { 
double cad{0};
for (int aj = 0; aj < nSHO; ++aj) {
double const cpr = apc[(a0 + aj)*nrhs + irhs][Real][0][ivec]; 
double const am = AtomMat[ai*nSHO + aj];                      
cad += am * cpr; 
} 
aac[(a0 + ai)*nrhs + irhs][Real][0][ivec] = cad;                  

} else {
assert(2 == R1C2); 
int constexpr Imag = R1C2 - 1; 
std__complex<double> cad(0.0, 0.0);
for (int spjn = 0; spjn < Noco; ++spjn) {
for (int aj = 0; aj < nSHO; ++aj) {
std__complex<double> const cpr(     
apc[(a0 + aj)*nrhs + irhs][Real][spjn][ivec],
apc[(a0 + aj)*nrhs + irhs][Imag][spjn][ivec]);
std__complex<double> const am(      
AtomMat[(((spin*Noco + spjn)*R1C2 + Real)*nSHO + ai)*nSHO + aj],
AtomMat[(((spin*Noco + spjn)*R1C2 + Imag)*nSHO + ai)*nSHO + aj]);
cad += am * cpr; 
} 
} 
aac[(a0 + ai)*nrhs + irhs][Real][spin][ivec] = cad.real(); 
aac[(a0 + ai)*nrhs + irhs][Imag][spin][ivec] = cad.imag(); 

} 
} 

}} 

} 

template <typename real_t, int R1C2=2, int Noco=1, int n64=64>
void __host__ SHOmul_driver(
real_t         (*const __restrict__ aac)[R1C2][Noco][Noco*n64] 
, real_t   const (*const __restrict__ apc)[R1C2][Noco][Noco*n64] 
, double   const (*const *const __restrict__ AtomMatrices)
, int8_t   const (*const __restrict__ AtomLmax)
, uint32_t const (*const __restrict__ AtomStarts)
, int      const nAtoms 
, int      const nrhs 
, int const echo=0 
) {
assert((1 == Noco && (1 == R1C2 || 2 == R1C2)) || (2 == Noco && 2 == R1C2));
if (nAtoms*nrhs < 1) return;

dim3 const gridDim(nAtoms, nrhs, 1), blockDim(Noco*n64, Noco, 1);
if (echo > 3) std::printf("# %s<%s,R1C2=%d,Noco=%d> <<< {nAtoms=%d, nrhs=%d, 1}, {%d, Noco=%d, 1} >>>\n",
__func__, real_t_name<real_t>(), R1C2, Noco,  nAtoms, nrhs,  Noco*n64, Noco);
SHOmul<real_t,R1C2,Noco,n64> 
#ifndef HAS_NO_CUDA
<<< gridDim, blockDim >>> (
#else 
(    gridDim, blockDim,
#endif 
aac, apc, AtomMatrices, AtomLmax, AtomStarts);
} 





class dyadic_plan_t {
public: 

uint32_t* AtomStarts          = nullptr; 
int8_t*   AtomLmax            = nullptr; 
double**  AtomMatrices        = nullptr; 
uint32_t nAtoms               = 0;

uint32_t* AtomImageIndex      = nullptr; 
uint32_t* AtomImageStarts     = nullptr; 
double  (*AtomImagePos)[3+1]  = nullptr; 
int8_t*   AtomImageLmax       = nullptr; 
double  (*AtomImagePhase)[4]  = nullptr; 
int8_t  (*AtomImageShift)[4]  = nullptr; 
uint32_t nAtomImages          = 0;

double* grid_spacing          = nullptr; 

int32_t nrhs                  = 0;
green_sparse::sparse_t<>* sparse_SHOprj = nullptr; 
green_sparse::sparse_t<>  sparse_SHOadd;
green_sparse::sparse_t<>  sparse_SHOsum;

std::vector<int32_t> global_atom_index;

size_t flop_count_SHOgen = 0,
flop_count_SHOsum = 0,
flop_count_SHOmul = 0,
flop_count_SHOadd = 0;

public: 

dyadic_plan_t(int const echo=0) {
if (echo > 0) std::printf("# construct %s\n", __func__);
} 

~dyadic_plan_t() {
#ifdef DEBUG
std::printf("# destruct %s\n", __func__);
#endif 
free_memory(AtomImageIndex);
free_memory(AtomImageStarts);
free_memory(AtomStarts);
if (AtomMatrices) for (uint32_t ia = 0; ia < nAtoms; ++ia) free_memory(AtomMatrices[ia]);
free_memory(AtomMatrices);
free_memory(grid_spacing);
free_memory(AtomImagePos);
free_memory(AtomImageLmax);
free_memory(AtomImagePhase);
free_memory(AtomImageShift);
free_memory(AtomLmax);
if (sparse_SHOprj) for (int32_t irhs = 0; irhs < nrhs; ++irhs) sparse_SHOprj[irhs].~sparse_t<>();
free_memory(sparse_SHOprj);
} 

status_t consistency_check() const {
status_t stat(0);
assert(nAtomImages >= nAtoms);
auto const rowStart = sparse_SHOsum.rowStart();
auto const colIndex = sparse_SHOsum.colIndex();
stat += (sparse_SHOsum.nRows() != nAtoms);
if (!rowStart) return stat;
for (uint32_t ia = 0; ia < nAtoms; ++ia) {
for (auto bsr = rowStart[ia]; bsr < rowStart[ia + 1]; ++bsr) {
auto const iai = colIndex[bsr];
stat += (AtomImageLmax[iai] != AtomLmax[ia]);
} 
} 
return stat; 
} 



inline int flop_count_SHOprj_SHOadd(int const L) {
return 2*(4*4*4 * (L+1) + 4*4 * (((L+1)*(L+2))/2) + 4 * (((L+1)*(L+2)*(L+3))/6));
} 

inline int flop_count_Hermite_Gauss(int const L) {
int constexpr flop_exp = 0; 
return 7 + flop_exp + 4*L;
} 

void update_flop_counts(int const echo=0) {

flop_count_SHOgen = 0;
flop_count_SHOadd = 0;
{
auto const iai_of_bsr = sparse_SHOadd.colIndex();
auto const nnz        = sparse_SHOadd.nNonzeros();
for (size_t bsr = 0; bsr < nnz; ++bsr) {
int const lmax = AtomImageLmax[iai_of_bsr[bsr]];
flop_count_SHOadd += 64 * flop_count_SHOprj_SHOadd(lmax); 
flop_count_SHOgen += 12 * flop_count_Hermite_Gauss(lmax); 
} 
}

flop_count_SHOmul = 0;
for (int ia = 0; ia < nAtoms; ++ia) {
int const lmax = AtomImageLmax[ia];
flop_count_SHOmul += pow2(sho_tools::nSHO(lmax));
} 
flop_count_SHOmul *= 2*nrhs*64; 

flop_count_SHOsum = 0;
if (nAtomImages > nAtoms) {
for (int iai = 0; iai < nAtomImages; ++iai) {
int const lmax = AtomImageLmax[iai];
flop_count_SHOsum += sho_tools::nSHO(lmax);
} 
flop_count_SHOsum *= 2*nrhs*64; 
} else {  }

} 

size_t get_flop_count(int const R1C2, int const Noco, int const echo=0) const {
size_t nops{0};
nops +=   flop_count_SHOmul*pow2(R1C2)*pow3(Noco); 
nops += 2*flop_count_SHOsum*pow2(R1C2)*pow2(Noco); 
nops += 2*flop_count_SHOgen; 
nops += 2*flop_count_SHOadd*R1C2*pow2(Noco); 
return nops;
} 

}; 

template <typename real_t, int R1C2=2, int Noco=1>
size_t __host__ multiply(
real_t         (*const __restrict__ Ppsi)[R1C2][Noco*64][Noco*64] 
, real_t         (*const __restrict__  Cpr)[R1C2][Noco]   [Noco*64] 
, real_t   const (*const __restrict__  psi)[R1C2][Noco*64][Noco*64] 
, dyadic_plan_t const & p
, uint32_t const (*const __restrict__ RowIndexCubes) 
, uint16_t const (*const __restrict__ ColIndexCubes) 
, float    const (*const __restrict__ CubePos)[3+1] 
, uint32_t const nnzb 
, int const echo=0 
, double (*const __restrict__  Cpr_export)[R1C2][Noco][Noco*64]=nullptr 
) {
assert((1 == Noco && (1 == R1C2 || 2 == R1C2)) || (2 == Noco && 2 == R1C2));
if (p.nAtomImages*p.nrhs < 1) return 0; 

assert(p.AtomImageStarts);
size_t const natomcoeffs = p.AtomImageStarts[p.nAtomImages];
if (echo > 6) std::printf("# %s<%s,R1C2=%d,Noco=%d> nAtoms=%d nAtomImages=%d nrhs=%d ncoeffs=%ld\n",
__func__, real_t_name<real_t>(), R1C2, Noco, p.nAtoms, p.nAtomImages, p.nrhs, natomcoeffs);

SHOprj_driver<real_t,R1C2,Noco>(Cpr, psi, p.AtomImagePos, p.AtomImageLmax, p.AtomImageStarts, p.nAtomImages,
p.sparse_SHOprj, RowIndexCubes, CubePos, p.grid_spacing, p.nrhs, echo);

real_t (*Cad)[R1C2][Noco][Noco*64]{nullptr};

size_t const ncoeffs = p.AtomStarts[p.nAtoms];
if (p.nAtomImages > p.nAtoms) {

auto cprj = Cpr_export ? Cpr_export : get_memory<double[R1C2][Noco][Noco*64]>(ncoeffs*p.nrhs, echo, "cprj");

SHOsum_driver<real_t,R1C2,Noco>(cprj, Cpr, p.AtomLmax, p.AtomStarts, p.AtomImageStarts, p.AtomImagePhase,
p.sparse_SHOsum, p.nAtoms, p.nrhs, true, echo); 
if (Cpr_export) { return 0; }

auto cadd = get_memory<double[R1C2][Noco][Noco*64]>(ncoeffs*p.nrhs, echo, "cadd");

SHOmul_driver<double,R1C2,Noco>(cadd, cprj, p.AtomMatrices, p.AtomLmax, p.AtomStarts, p.nAtoms, p.nrhs, echo);

free_memory(cprj);

Cad = get_memory<real_t[R1C2][Noco][Noco*64]>(natomcoeffs*p.nrhs, echo, "Cad"); 

SHOsum_driver<real_t,R1C2,Noco>(cadd, Cad, p.AtomLmax, p.AtomStarts, p.AtomImageStarts, p.AtomImagePhase,
p.sparse_SHOsum, p.nAtoms, p.nrhs, false, echo); 

free_memory(cadd);

} else {
assert(p.nAtomImages == p.nAtoms); 
assert(natomcoeffs == ncoeffs);
if (Cpr_export) { set(Cpr_export[0][0][0], ncoeffs*p.nrhs*R1C2*Noco*Noco*64, Cpr[0][0][0]); return 0; }

Cad = get_memory<real_t[R1C2][Noco][Noco*64]>(natomcoeffs*p.nrhs, echo, "Cad"); 

SHOmul_driver<real_t,R1C2,Noco>(Cad, Cpr, p.AtomMatrices, p.AtomLmax, p.AtomStarts, p.nAtoms, p.nrhs, echo);

} 

SHOadd_driver<real_t,R1C2,Noco>(Ppsi, Cad, p.AtomImagePos, p.AtomImageLmax, p.AtomImageStarts,
p.sparse_SHOadd.rowStart(), p.sparse_SHOadd.colIndex(),
RowIndexCubes, ColIndexCubes, CubePos, p.grid_spacing, nnzb, p.nrhs, echo);
free_memory(Cad);

return p.get_flop_count(R1C2, Noco, echo);
} 


template <typename real_t, int R1C2=2, int Noco=1>
std::vector<std::vector<double>> get_projection_coefficients(
real_t   const (*const __restrict__ Green)[R1C2][Noco*64][Noco*64] 
, dyadic_plan_t const & p
, uint32_t const (*const __restrict__ RowIndexCubes) 
, float    const (*const __restrict__ rowCubePos)[3+1] 
, float    const (*const __restrict__ colCubePos)[3+1] 
, int const echo=0 
) {
if (echo > 0) std::printf("# %s\n", __func__);
size_t const ncoeffs = p.AtomStarts[p.nAtoms];
auto Cpr_export = get_memory<double[R1C2][Noco][Noco*64]>(ncoeffs*p.nrhs, echo, "Cpr_export");
size_t const natomcoeffs = p.AtomImageStarts[p.nAtomImages];
auto Cpr = get_memory<real_t[R1C2][Noco][Noco*64]>(natomcoeffs*p.nrhs, echo, "Cpr");
auto const nflop = multiply<real_t,R1C2,Noco>(nullptr, Cpr, Green, p, RowIndexCubes, nullptr, rowCubePos, 0, echo, Cpr_export);
free_memory(Cpr);
assert(0 == nflop);
if (echo > 3) std::printf("# %s from projected Green function\n", __func__);
assert(p.nAtomImages == p.nAtoms && "reduction over Bloch phases not yet implemented!");
auto const result = green_projection::SHOprj(Cpr_export, p.AtomImagePos, p.AtomImageLmax, p.AtomImageStarts, p.AtomImageIndex,
p.AtomImagePhase, p.nAtomImages, p.AtomLmax, p.nAtoms, colCubePos, p.grid_spacing, p.nrhs, echo);
free_memory(Cpr_export);
return result;
} 


inline std::vector<double> __host__ sho_normalization(int const lmax, double const sigma=1) {
int const n1ho = sho_tools::n1HO(lmax);
std::vector<double> v1(n1ho, constants::sqrtpi*sigma);
{
double fac{1};
for (int nu = 0; nu <= lmax; ++nu) {
v1[nu] *= fac;
fac *= 0.5*(nu + 1); 
} 
}

std::vector<double> vec(sho_tools::nSHO(lmax), 1.);
{
int sho{0};
for (int iz = 0; iz <= lmax; ++iz) {
for (int iy = 0; iy <= lmax - iz; ++iy) {
for (int ix = 0; ix <= lmax - iz - iy; ++ix) {
vec[sho] = v1[ix] * v1[iy] * v1[iz];
++sho;
}}} 
assert(vec.size() == sho);
}
return vec;
} 


#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 


inline status_t test_Hermite_polynomials_1D(int const echo=0, double const sigma=16) {
int constexpr Lmax = 7;
double H1D[Lmax + 1][3][4]; 
float  h1D[Lmax + 1][3][4]; 
float     xi_squared[3][4]; 
float  const xyzc[] = {0, 0, 0}; 
double const hxyz[] = {1, 1, 1,   6.2832}; 
double       xyza[] = {0, 0, 0,   1./std::sqrt(sigma)}; 
if (echo > 10) std::printf("\n## %s xi, H1D[0:%d]:", __func__, Lmax);
double maxdev{0};
for (int it = 0; it < 9; ++it) {
for (int i3 = 0; i3 < 3; ++i3) {
xyza[i3] = -4*(it*3 + i3);
for (int i4 = 0; i4 < 4; ++i4) {
int const i = i3*4 + i4; 
Hermite_polynomials_1D(h1D, xi_squared, i, Lmax, xyza, xyzc, hxyz); 
Hermite_polynomials_1D(H1D, xi_squared, i, Lmax, xyza, xyzc, hxyz); 
if (echo > 10) std::printf("\n%g  ", std::sqrt(xi_squared[i3][i4]));
for (int l = 0; l <= Lmax; ++l) {
if (echo > 10) std::printf(" %g", H1D[l][i3][i4]);
auto const dev = h1D[l][i3][i4] - H1D[l][i3][i4];
if (0 != H1D[l][i3][i4]) {
auto const reldev = std::abs(dev/H1D[l][i3][i4]);
maxdev = std::max(maxdev, reldev);
}
} 
} 
} 
} 
if (echo > 3) std::printf("\n# %s largest relative deviation between float and double is %.1e\n", __func__, maxdev);
if (echo > 10) std::printf("\n\n\n");
return 0;
} 


template <typename real_t, int R1C2=2, int Noco=1>
size_t multiply( 
real_t         (*const __restrict__ Ppsi)[R1C2][Noco*64][Noco*64] 
, real_t         (*const __restrict__  Cpr)[R1C2][Noco]   [Noco*64] 
, real_t   const (*const __restrict__  psi)[R1C2][Noco*64][Noco*64] 
, double   const (*const __restrict__ AtomPos)[3+1] 
, int8_t   const (*const __restrict__ AtomLmax) 
, uint32_t const (*const __restrict__ AtomStarts) 
, uint32_t const natoms
, green_sparse::sparse_t<> const (*const __restrict__ sparse_SHOprj)
, double   const (*const          *const __restrict__ AtomMatrices)
, green_sparse::sparse_t<> const &                    sparse_SHOadd
, uint32_t const (*const __restrict__ RowIndexCubes) 
, uint16_t const (*const __restrict__ ColIndexCubes) 
, float    const (*const __restrict__ CubePos)[3+1] 
, double   const (*const __restrict__ hGrid) 
, uint32_t const nnzb 
, uint32_t const nrhs 
, int const echo=0 
) {
assert((1 == Noco && (1 == R1C2 || 2 == R1C2)) || (2 == Noco && 2 == R1C2));
if (natoms*nrhs < 1) return 0; 

auto const natomcoeffs = AtomStarts[natoms];
if (echo > 6) std::printf("# %s<%s> R1C2=%d Noco=%d natoms=%d nrhs=%d ncoeffs=%d\n", __func__, real_t_name<real_t>(), R1C2, Noco, natoms, nrhs, natomcoeffs);

SHOprj_driver<real_t,R1C2,Noco>(Cpr, psi, AtomPos, AtomLmax, AtomStarts, natoms, sparse_SHOprj, RowIndexCubes, CubePos, hGrid, nrhs, echo);


auto Cad = get_memory<real_t[R1C2][Noco][Noco*64]>(natomcoeffs*nrhs, echo, "Cad"); 

SHOmul_driver<real_t,R1C2,Noco,64>(Cad, Cpr, AtomMatrices, AtomLmax, AtomStarts, natoms, nrhs, echo);

SHOadd_driver<real_t,R1C2,Noco>(Ppsi, Cad, AtomPos, AtomLmax, AtomStarts, sparse_SHOadd.rowStart(), sparse_SHOadd.colIndex(), RowIndexCubes, ColIndexCubes, CubePos, hGrid, nnzb, nrhs, echo);

free_memory(Cad);

return 0; 
} 



template <typename real_t, int R1C2=2, int Noco=1>
inline status_t test_SHOprj_and_SHOadd(int const echo=0, int8_t const lmax=6) {
auto const sigma = control::get("green_dyadic.test.sigma", 1.);
auto const hg    = control::get("green_dyadic.test.grid.spacing", 0.25);
auto const rc    = control::get("green_dyadic.test.rc", 7.);
int  const nb    = control::get("green_dyadic.test.nb", 14.);
int  const natoms = 1, nrhs = 1, nnzb = pow3(nb);
int  const nsho = sho_tools::nSHO(lmax);
auto psi  = get_memory<real_t[R1C2][Noco*64][Noco*64]>(nnzb, echo, "psi");
auto Vpsi = get_memory<real_t[R1C2][Noco*64][Noco*64]>(nnzb, echo, "Vpsi");
set(psi[0][0][0], nnzb*R1C2*pow2(Noco*64ull), real_t(0)); 
auto apc = get_memory<real_t[R1C2][Noco   ][Noco*64]>(natoms*nsho*nrhs, echo, "apc");
set(apc[0][0][0], natoms*nsho*nrhs*R1C2*pow2(Noco)*64, real_t(0)); 

auto sparse_SHOprj = get_memory<green_sparse::sparse_t<>>(nrhs, echo, "sparse_SHOprj");
{
std::vector<uint32_t> iota(nnzb); for (int inzb = 0; inzb < nnzb; ++inzb) iota[inzb] = inzb;
std::vector<std::vector<uint32_t>> SHO_prj(natoms, iota);
sparse_SHOprj[0] = green_sparse::sparse_t<>(SHO_prj, false, __func__, echo - 9);
}
green_sparse::sparse_t<> sparse_SHOadd;
{
std::vector<std::vector<uint32_t>> SHO_add(nnzb, std::vector<uint32_t>(1, 0));
sparse_SHOadd    = green_sparse::sparse_t<>(SHO_add, false, __func__, echo - 9);
}

auto ColIndexCubes = get_memory<uint16_t>(nnzb, echo, "ColIndexCubes");     set(ColIndexCubes, nnzb, uint16_t(0));
auto RowIndexCubes = get_memory<uint32_t>(nnzb, echo, "RowIndexCubes");     for (int inzb = 0; inzb < nnzb; ++inzb) RowIndexCubes[inzb] = inzb;
auto hGrid         = get_memory<double>(3+1, echo, "hGrid");                set(hGrid, 3, hg); hGrid[3] = rc;
auto AtomPos       = get_memory<double[3+1]>(natoms, echo, "AtomPos");      set(AtomPos[0], 3, hGrid, 0.5*4*nb);  AtomPos[0][3] = 1./std::sqrt(sigma);
auto AtomLmax      = get_memory<int8_t>(natoms, echo, "AtomLmax");          set(AtomLmax, natoms, lmax);
auto AtomStarts    = get_memory<uint32_t>(natoms + 1, echo, "AtomStarts");  for(int ia = 0; ia <= natoms; ++ia) AtomStarts[ia] = ia*nsho;
auto CubePos       = get_memory<float[3+1]>(nnzb, echo, "CubePos");
for (int iz = 0; iz < nb; ++iz) {
for (int iy = 0; iy < nb; ++iy) {
for (int ix = 0; ix < nb; ++ix) {
int const xyz0[] = {ix, iy, iz, 0};
set(CubePos[(iz*nb + iy)*nb + ix], 4, xyz0);
}}} 

if (echo > 11) std::printf("# here %s:%d\n", __func__, __LINE__);
{
auto const dVol = hGrid[0]*hGrid[1]*hGrid[2];
auto const sho_norm = sho_normalization(lmax, sigma);
for (int isho = 0; isho < std::min(nsho, 64); ++isho) {
apc[isho*nrhs][0][0][isho] = dVol/sho_norm[isho]; 
} 
}

SHOadd_driver<real_t,R1C2,Noco>(psi, apc, AtomPos, AtomLmax, AtomStarts, sparse_SHOadd.rowStart(), sparse_SHOadd.colIndex(), RowIndexCubes, ColIndexCubes, CubePos, hGrid, nnzb, nrhs, echo);
if (0) {
cudaDeviceSynchronize();
size_t nz{0};
for (int i = 0; i < nnzb*64; ++i) {
auto const value = psi[i >> 6][0][i & 63][0];
nz += (0 == value);
if (echo > 19) std::printf(" %g", value);
} 
if (echo > 3) std::printf("\n# %ld non-zeros of %d\n", nz, nnzb*64);
} 

SHOprj_driver<real_t,R1C2,Noco>(apc, psi, AtomPos, AtomLmax, AtomStarts, natoms, sparse_SHOprj, RowIndexCubes, CubePos, hGrid, nrhs, echo);

float maxdev[2] = {0, 0}; 
float maxdev_nu[8][2] = {{0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}};
{ 
cudaDeviceSynchronize();
std::vector<int8_t> nu_of_sho(nsho, -1);
{
int sho{0};
for (int iz = 0; iz <= lmax; ++iz) {
for (int iy = 0; iy <= lmax - iz; ++iy) {
for (int ix = 0; ix <= lmax - iz - iy; ++ix) {
nu_of_sho[sho] = ix + iy + iz;
++sho;
} 
} 
} 
assert(nu_of_sho.size() == sho);
}
auto const msho = std::min(nsho, 64);
if (echo > 5) std::printf("# %d of %d projection coefficients ", msho, nsho);
for (int isho = 0; isho < msho; ++isho) {
if (echo > 9) std::printf("\n# projection coefficients[%2d]: ", isho);
for (int jsho = 0; jsho < msho; ++jsho) {
double const value = apc[isho*nrhs][0][0][jsho];
int const diag = (isho == jsho); 
if (echo > 9) std::printf(" %.1e", value);
float const absdev = std::abs(value - diag);
maxdev[diag] = std::max(maxdev[diag], absdev);
auto const nu = std::max(nu_of_sho[isho], nu_of_sho[jsho]);
assert(nu >= 0);
if (nu < 8) maxdev_nu[nu][diag] = std::max(maxdev_nu[nu][diag], absdev);
} 
if (echo > 9) std::printf(" diagonal=");
if (echo > 5) std::printf(" %.3f", apc[isho*nrhs][0][0][isho]);
} 
if (echo > 5) std::printf("\n");
} 
if (echo > 2) std::printf("# %s<%s> orthogonality error %.2e, normalization error %.2e\n", __func__, real_t_name<real_t>(), maxdev[0], maxdev[1]);
if (echo > 5) {
for (int nu = 0; nu < std::min(8, lmax + 1); ++nu) {
std::printf("# %s  nu=%d  %.2e  %.2e\n", real_t_name<real_t>(), nu, maxdev_nu[nu][0], maxdev_nu[nu][1]);
} 
} 

if (1) {
auto AtomMatrices = get_memory<double*>(natoms, echo, "AtomMatrices");
for (int ia = 0; ia < natoms; ++ia) {
if (echo > 9) std::printf("# %s atom image #%d has lmax= %d and %d coefficients starting at %d\n", __func__, ia, AtomLmax[ia], nsho, AtomStarts[ia]);
AtomMatrices[ia] = get_memory<double>(pow2(Noco)*2*pow2(nsho), echo, "AtomMatrix[ia]");
set(AtomMatrices[ia], pow2(Noco)*2*pow2(nsho), 0.0);
} 
multiply<real_t,R1C2,Noco>(Vpsi, apc, psi, AtomPos, AtomLmax, AtomStarts, natoms, sparse_SHOprj, AtomMatrices, sparse_SHOadd, RowIndexCubes, ColIndexCubes, CubePos, hGrid, nnzb, nrhs, echo);
for (int ia = 0; ia < natoms; ++ia) {
free_memory(AtomMatrices[ia]);
} 
free_memory(AtomMatrices);
} 

sparse_SHOprj[0].~sparse_t<>();
free_memory(sparse_SHOprj);
free_memory(ColIndexCubes);
free_memory(RowIndexCubes);
free_memory(CubePos);
free_memory(AtomStarts);
free_memory(AtomLmax);
free_memory(AtomPos);
free_memory(apc);
free_memory(Vpsi);
free_memory(psi);
return 0;
} 

inline status_t test_SHOprj_and_SHOadd(int const echo=0) {
status_t stat(0);
int const more = control::get("green_dyadic.test.more", 0.);
stat +=   test_SHOprj_and_SHOadd<float ,1,1>(echo); 
if (more) test_SHOprj_and_SHOadd<float ,2,1>(echo); 
if (more) test_SHOprj_and_SHOadd<float ,2,2>(echo); 
stat +=   test_SHOprj_and_SHOadd<double,1,1>(echo); 
if (more) test_SHOprj_and_SHOadd<double,2,1>(echo); 
if (more) test_SHOprj_and_SHOadd<double,2,2>(echo); 
return stat;
} 

inline status_t all_tests(int const echo=0) {
status_t stat(0);
stat += test_Hermite_polynomials_1D(echo);
stat += test_SHOprj_and_SHOadd(echo);
return stat;
} 

#endif 

} 


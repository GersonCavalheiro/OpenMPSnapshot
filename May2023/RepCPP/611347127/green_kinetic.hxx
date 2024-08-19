#pragma once

#include <cstdio> 
#include <cstdint> 
#include <cassert> 

#include "status.hxx" 
#include "green_memory.hxx" 
#include "green_sparse.hxx" 
#include "data_view.hxx" 

template <typename uint_t, typename int_t> inline
size_t index3D(uint_t const n[3], int_t const i[3]) {
return size_t(i[2]*n[1] + i[1])*n[0] + i[0];
} 


namespace green_kinetic {

int constexpr nhalo = 4; 

int32_t constexpr BLOCK_EXISTS = 1;
int32_t constexpr BLOCK_IS_ZERO = 0;
int32_t constexpr BLOCK_NEEDS_PHASE = -1;


template <typename real_t, int R1C2=2, int Noco=1> 
void __global__ Laplace8th( 
#ifdef    HAS_NO_CUDA
dim3 const & gridDim, dim3 const & blockDim,
#endif 
real_t        (*const __restrict__ Tpsi)[R1C2][Noco*64][Noco*64] 
, real_t  const (*const __restrict__  psi)[R1C2][Noco*64][Noco*64] 
, int32_t const (*const *const __restrict__ index_list) 
, double const prefactor
, int const Stride 
, double const phase[2][2]
) {
double const norm = prefactor/5040.; 
real_t const  c0 =     -14350*norm,
c1 =       8064*norm,
c2 =      -1008*norm,
c3 =        128*norm,
c4 =         -9*norm; 


assert(16 == gridDim.y);
assert(1  == gridDim.z);
assert(Noco*64 == blockDim.x);
assert(Noco    == blockDim.y);
assert(R1C2    == blockDim.z);

#ifdef    HAS_NO_CUDA
dim3 blockIdx(0,0,0);
for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y)
for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x)
#endif 
{ 

auto const *const list = index_list[blockIdx.x]; 

int const i16 = blockIdx.y;
int const i64 = (16==Stride)? i16 : ( (4==Stride)? (16*(i16 >> 2) + (i16 & 0x3)) : (4*i16) );

#ifdef    HAS_NO_CUDA
dim3 threadIdx(0,0,0);
for (threadIdx.z = 0; threadIdx.z < blockDim.z; ++threadIdx.z)
for (threadIdx.y = 0; threadIdx.y < blockDim.y; ++threadIdx.y)
for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x)
#endif 
{ 

#define INDICES(i4) [threadIdx.z][threadIdx.y*64 + i64 + Stride*i4][threadIdx.x]

real_t w0{0}, w1{0}, w2{0}, w3{0}; 

int ilist{nhalo - 1}; 

int ii = list[ilist++]; 
if (BLOCK_IS_ZERO == ii) {
} else { 
if (ii > 0) std::printf("# Error: ii= %d ilist= %i\n", ii, ilist);
assert(ii <= BLOCK_NEEDS_PHASE && "list[3] must be either 0 (isolated BC) or negative (periodic BC)");
int const jj = BLOCK_NEEDS_PHASE*(ii + BLOCK_EXISTS); 
assert(jj >= 0); 
assert(phase && "a phase must be given for complex BCs");
real_t const ph_Re = phase[0][0]; 
w0 = ph_Re * psi[jj]INDICES(0);
w1 = ph_Re * psi[jj]INDICES(1);
w2 = ph_Re * psi[jj]INDICES(2);
w3 = ph_Re * psi[jj]INDICES(3);
if (2 == R1C2) {
real_t const ph_Im = phase[0][1] * (1 - 2*threadIdx.z); 
#define INDICES_Im(i4) [R1C2 - 1 - threadIdx.z][threadIdx.y*64 + i64 + Stride*i4][threadIdx.x]
w0 -= ph_Im * psi[jj]INDICES_Im(0);
w1 -= ph_Im * psi[jj]INDICES_Im(1);
w2 -= ph_Im * psi[jj]INDICES_Im(2);
w3 -= ph_Im * psi[jj]INDICES_Im(3);
} 
} 

real_t w4, w5, w6, w7, wn; 

ii = list[ilist++] - BLOCK_EXISTS; assert(ii >= 0); 
w4 = psi[ii]INDICES(0);
w5 = psi[ii]INDICES(1);
w6 = psi[ii]INDICES(2);
w7 = psi[ii]INDICES(3);

assert(5 == ilist);
while (ii >= 0) {
int const i0 = ii; 
ii = list[ilist++] - BLOCK_EXISTS; 
bool const load = (ii >= 0);

#define FD9POINT(i4,  M4, M3, M2, M1, W0, P1, P2, P3, P4) \
P4 = load ? psi[ii]INDICES(i4) : 0; \
Tpsi[i0]INDICES(i4) += c0*W0 + c1*M1 + c1*P1 + c2*M2 + c2*P2 + c3*M3 + c3*P3 + c4*M4 + c4*P4; \
M4 = P4;

if (0 == (ilist & 0x1)) { 
FD9POINT(0,  w0, w1, w2, w3, w4, w5, w6, w7, wn)
FD9POINT(1,  w1, w2, w3, w4, w5, w6, w7, w0, wn)
FD9POINT(2,  w2, w3, w4, w5, w6, w7, w0, w1, wn)
FD9POINT(3,  w3, w4, w5, w6, w7, w0, w1, w2, wn)
} else {                
FD9POINT(0,  w4, w5, w6, w7, w0, w1, w2, w3, wn)
FD9POINT(1,  w5, w6, w7, w0, w1, w2, w3, w4, wn)
FD9POINT(2,  w6, w7, w0, w1, w2, w3, w4, w5, wn)
FD9POINT(3,  w7, w0, w1, w2, w3, w4, w5, w6, wn)
} 
#undef  FD9POINT
} 

ii = list[ilist - 1]; 
if (BLOCK_IS_ZERO == ii) {
} else {
assert(ii < 0 && "last list item must be either 0 (isolated BC) or negative (periodic BC)");
int const jj = BLOCK_NEEDS_PHASE*(ii + BLOCK_EXISTS); 
assert(jj >= 0); 
int const i0 = list[ilist - 2] - BLOCK_EXISTS; 
assert(i0 >= 0); 
assert(phase && "no (right) phase given"); 
real_t const ph_Re = phase[1][0]; 
w0 = ph_Re * psi[jj]INDICES(0);
w1 = ph_Re * psi[jj]INDICES(1);
w2 = ph_Re * psi[jj]INDICES(2);
w3 = ph_Re * psi[jj]INDICES(3);
if (2 == R1C2) {
real_t const ph_Im = phase[1][1] * (1 - 2*threadIdx.z); 
w0 -= ph_Im * psi[jj]INDICES_Im(0);
w1 -= ph_Im * psi[jj]INDICES_Im(1);
w2 -= ph_Im * psi[jj]INDICES_Im(2);
w3 -= ph_Im * psi[jj]INDICES_Im(3);
#undef  INDICES_Im
} 
Tpsi[i0]INDICES(0) += c4*w0;
Tpsi[i0]INDICES(1) += c3*w0 + c4*w1;
Tpsi[i0]INDICES(2) += c2*w0 + c3*w1 + c4*w2;
Tpsi[i0]INDICES(3) += c1*w0 + c2*w1 + c3*w2 + c4*w3;
} 

#undef  INDICES
}} 


} 



template <typename real_t, int R1C2=2, int Noco=1>
void __global__ Laplace16th( 
#ifdef    HAS_NO_CUDA
dim3 const & gridDim, dim3 const & blockDim,
#endif 
real_t        (*const __restrict__ Tpsi)[R1C2][Noco*64][Noco*64] 
, real_t  const (*const __restrict__  psi)[R1C2][Noco*64][Noco*64] 
, int32_t const (*const *const __restrict__ index_list) 
, double const prefactor
, int const Stride 
, double const phase[2][2]=nullptr 
) {
assert(nullptr == phase);          
double const norm = prefactor/302702400.;
real_t const  c0 = -924708642*norm,
c1 =  538137600*norm,
c2 =  -94174080*norm,
c3 =   22830080*norm,
c4 =   -5350800*norm,
c5 =    1053696*norm,
c6 =    -156800*norm,
c7 =      15360*norm,
c8 =       -735*norm;

assert(16 == gridDim.y);
assert(1  == gridDim.z);
assert(Noco*64 == blockDim.x);
assert(Noco    == blockDim.y);
assert(R1C2    == blockDim.z);

#ifdef    HAS_NO_CUDA
dim3 blockIdx(0,0,0);
for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y)
for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x)
#endif 
{ 

auto const *const list = index_list[blockIdx.x]; 

int const i16 = blockIdx.y; 
int const i64 = (16==Stride)? i16 : ( (4==Stride)? (16*(i16 >> 2) + (i16 & 0x3)) : (4*i16) );

#ifdef    HAS_NO_CUDA
dim3 threadIdx(0,0,0);
for (threadIdx.z = 0; threadIdx.z < blockDim.z; ++threadIdx.z)
for (threadIdx.y = 0; threadIdx.y < blockDim.y; ++threadIdx.y)
for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x)
#endif 
{ 

#define INDICES(i4) [threadIdx.z][threadIdx.y*64 + i64 + Stride*i4][threadIdx.x]

real_t w0{0}, w1{0}, w2{0}, w3{0}, w4{0}, w5{0}, w6{0}, w7{0}, 
w8, w9, wa, wb, wc, wd, we, wf, wn; 

int ilist{0}; 
for (int i4 = 0; i4 < nhalo; ++i4) {
assert(0 == list[ilist++] && "Laplace16th can only deal with isolated boundary conditions");
} 

int i0 = list[ilist++] - BLOCK_EXISTS; 
assert(i0 >= 0); 
w8 = psi[i0]INDICES(0); 
w9 = psi[i0]INDICES(1); 
wa = psi[i0]INDICES(2); 
wb = psi[i0]INDICES(3); 

int i1 = list[ilist++] - BLOCK_EXISTS; 
if (i1 >= 0) {
wc = psi[i1]INDICES(0); 
wd = psi[i1]INDICES(1); 
we = psi[i1]INDICES(2); 
wf = psi[i1]INDICES(3); 
} else {
wc = 0; wd = 0; we = 0; wf = 0; 
} 

int i2 = list[ilist++] - BLOCK_EXISTS; 

assert(ilist == 7);

while (i0 >= 0) {
bool const load = (i2 >= 0); 

#define FD17POINT(i4,  M8, M7, M6, M5, M4, M3, M2, M1, W0, P1, P2, P3, P4, P5, P6, P7, P8) \
P8 = load ? psi[i2]INDICES(i4) : 0; \
Tpsi[i0]INDICES(i4) += c0*W0 + c1*M1 + c1*P1 + c2*M2 + c2*P2 + c3*M3 + c3*P3 + c4*M4 + c4*P4 \
+ c5*M5 + c5*P5 + c6*M6 + c6*P6 + c7*M7 + c7*P7 + c8*M8 + c8*P8; \
M8 = P8;

int const mod4 = ilist & 0x3; 
if        (0x3 == mod4) { 
FD17POINT(0,  w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, wa, wb, wc, wd, we, wf, wn)
FD17POINT(1,  w1, w2, w3, w4, w5, w6, w7, w8, w9, wa, wb, wc, wd, we, wf, w0, wn)
FD17POINT(2,  w2, w3, w4, w5, w6, w7, w8, w9, wa, wb, wc, wd, we, wf, w0, w1, wn)
FD17POINT(3,  w3, w4, w5, w6, w7, w8, w9, wa, wb, wc, wd, we, wf, w0, w1, w2, wn)
} else if (0x0 == mod4) { 
FD17POINT(0,  w4, w5, w6, w7, w8, w9, wa, wb, wc, wd, we, wf, w0, w1, w2, w3, wn)
FD17POINT(1,  w5, w6, w7, w8, w9, wa, wb, wc, wd, we, wf, w0, w1, w2, w3, w4, wn)
FD17POINT(2,  w6, w7, w8, w9, wa, wb, wc, wd, we, wf, w0, w1, w2, w3, w4, w5, wn)
FD17POINT(3,  w7, w8, w9, wa, wb, wc, wd, we, wf, w0, w1, w2, w3, w4, w5, w6, wn)
} else if (0x1 == mod4) { 
FD17POINT(0,  w8, w9, wa, wb, wc, wd, we, wf, w0, w1, w2, w3, w4, w5, w6, w7, wn)
FD17POINT(1,  w9, wa, wb, wc, wd, we, wf, w0, w1, w2, w3, w4, w5, w6, w7, w8, wn)
FD17POINT(2,  wa, wb, wc, wd, we, wf, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, wn)
FD17POINT(3,  wb, wc, wd, we, wf, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, wa, wn)
} else                   { 
FD17POINT(0,  wc, wd, we, wf, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, wa, wb, wn)
FD17POINT(1,  wd, we, wf, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, wa, wb, wc, wn)
FD17POINT(2,  we, wf, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, wa, wb, wc, wd, wn)
FD17POINT(3,  wf, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, wa, wb, wc, wd, we, wn)
} 
#undef  FD17POINT
i0 = i1; i1 = i2; i2 = list[ilist++] - BLOCK_EXISTS; 
} 

#undef  INDICES
}} 

} 






template <typename real_t, int R1C2=2, int Noco=1>
int Laplace_driver(
real_t        (*const __restrict__ Tpsi)[R1C2][Noco*64][Noco*64] 
, real_t  const (*const __restrict__  psi)[R1C2][Noco*64][Noco*64] 
, int32_t const (*const *const __restrict__ index_list) 
, double const prefactor
, uint32_t const num
, int const Stride 
, double const phase[2][2]=nullptr
, int const FD_range=4 
) {
if (num < 1 || FD_range < 1) return 0;
assert(1 == Stride || 4 == Stride || 16 == Stride);
auto const kernel_ptr = (8 == FD_range) ? Laplace16th<real_t,R1C2,Noco> : Laplace8th<real_t,R1C2,Noco>;
if (8 == FD_range) phase = nullptr; 
dim3 const gridDim(num, 16, 1), blockDim(Noco*64, Noco, R1C2);
kernel_ptr 
#ifdef    HAS_NO_CUDA
(    gridDim, blockDim,
#else  
<<< gridDim, blockDim >>> (
#endif 
Tpsi, psi, index_list, prefactor, Stride, phase);
return (8 == FD_range) ? 17 : 9; 
} 


void __host__ set_phase(
double phase[3][2][2]
, double const phase_angles[3]=nullptr
, int const echo=0
); 



status_t finite_difference_plan(
green_sparse::sparse_t<int32_t> & sparse 
, int16_t & FD_range 
, int const dd 
, bool const boundary_is_periodic
, uint32_t const num_target_coords[3]
, uint32_t const RowStart[]
, uint16_t const ColIndex[]
, view3D<int32_t> const & iRow_of_coords 
, std::vector<bool> const sparsity_pattern[] 
, unsigned const nrhs=1 
, int const echo=0 
); 




class kinetic_plan_t {
public:

kinetic_plan_t() {} 
#if 0
kinetic_plan_t(
green_sparse::sparse_t<int32_t> & sparse 
, int16_t & FD_range 
, int const dd 
, bool const boundary_is_periodic
, uint32_t const num_target_coords[3]
, uint32_t const RowStart[]
, uint16_t const ColIndex[]
, view3D<int32_t> const & iRow_of_coords 
, std::vector<bool> const sparsity_pattern[] 
, unsigned const nrhs=1 
, double const grid_spacing=1 
, int const echo=0 
) 
: derivative_direction(dd)
{
auto const stat = finite_difference_plan(sparse, FD_range, 
dd, boundary_is_periodic, num_target_coords, RowStart, ColIndex,
iRow_of_coords, sparsity_pattern, nrhs, echo);
if (0 == stat) {
prefactor = -0.5/(grid_spacing*grid_spacing);
lists = get_memory<int32_t const *>(sparse.nRows(), echo, "lists[dd]");
} else {
if (echo > 2) std::printf("# failed to set up finite_difference_plan for %c-direction, status= %i\n", 'x' + dd, int(stat));
}
} 
#endif 

void set(
int const dd
, double const grid_spacing=1
, size_t const nnzbX=1
, int const echo=0 
) {
derivative_direction = dd;
nnzb = nnzbX;
prefactor = -0.5/(grid_spacing*grid_spacing);
char lists_[8] = "list[?]"; lists_[5] = 'x' + dd;
lists = get_memory<int32_t const *>(sparse.nRows(), echo, lists_);
auto const rowStart = sparse.rowStart();
auto const colIndex = sparse.colIndex();
for (uint32_t il = 0; il < sparse.nRows(); ++il) {
lists[il] = &colIndex[rowStart[il]];
} 
} 

~kinetic_plan_t() {
if (lists) {
std::printf("# free list for dd=%i\n", derivative_direction);
free_memory(lists);
}
} 

public:
green_sparse::sparse_t<int32_t> sparse;
double prefactor = 0; 
size_t nnzb; 
int16_t FD_range = 8; 
int32_t const ** lists = nullptr; 
int derivative_direction = -1; 

public:

template <typename real_t, int R1C2=2, int Noco=1>
size_t multiply(
real_t         (*const __restrict__ Tpsi)[R1C2][Noco*64][Noco*64] 
, real_t   const (*const __restrict__  psi)[R1C2][Noco*64][Noco*64] 
, double   const phase[2][2]=nullptr 
, int      const echo=0
) const { 
int  const stride = 1 << (2*derivative_direction); 
auto const nFD = Laplace_driver<real_t,R1C2,Noco>(Tpsi, psi, lists, prefactor, sparse.nRows(), stride, phase, FD_range);
size_t const nops = nnzb*nFD*R1C2*pow2(Noco*64ul)*2ul;
if (echo > 7) {
char const fF = (8 == sizeof(real_t)) ? 'F' : 'f'; 
std::printf("# green_kinetic::%s dd=\'%c\', nFD= %d, number= %d, %.3f M%clop\n",
__func__, 'x' + derivative_direction, nFD, sparse.nRows(), nops*1e-6, fF);
} 
return nops; 
} 

}; 



#if 0
template <typename real_t, int R1C2=2, int Noco=1>
size_t multiply( 
real_t         (*const __restrict__ Tpsi)[R1C2][Noco*64][Noco*64] 
, real_t   const (*const __restrict__  psi)[R1C2][Noco*64][Noco*64] 
, green_sparse::sparse_t<int32_t> const kinetic_plan[3]
, double const hgrid[3] 
, double const phase[3][2][2] 
, int16_t const FD_range[3] 
, size_t const nnzb=1 
, int const echo=0
) {
int const nFD[] = {FD_range[0], FD_range[1], FD_range[2]};
uint32_t num[3];
int32_t const ** lists[3];
for (int dd = 0; dd < 3; ++dd) {
num[dd] = kinetic_plan[dd].nRows();
lists[dd] = get_memory<int32_t const *>(num[dd], echo, "num[dd]");
auto const rowStart = kinetic_plan[dd].rowStart();
auto const colIndex = kinetic_plan[dd].colIndex();
for (uint32_t il = 0; il < num[dd]; ++il) {
lists[dd][il] = &colIndex[rowStart[il]];
} 
} 
if (echo > 13) std::printf("# green_kinetic::%s FD_range= %d %d %d, numbers= %d %d %d\n",
__func__, FD_range[0], FD_range[1], FD_range[2], num[0], num[1], num[2]);

auto const nops = multiply<real_t,R1C2,Noco>(Tpsi, psi, num, lists[0], lists[1], lists[2], hgrid, nFD, phase, nnzb, echo);

for (int dd = 0; dd < 3; ++dd) {
free_memory(lists[dd]);
} 
return nops;
} 
#endif

status_t all_tests(int const echo=0); 


} 

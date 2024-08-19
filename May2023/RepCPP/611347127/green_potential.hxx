#pragma once

#include <cstdio> 
#include <cstdint> 
#include <cassert> 
#include <complex> 

#include "status.hxx" 
#include "green_memory.hxx" 
#include "inline_math.hxx" 
#include "green_parallel.hxx" 
#include "global_coordinates.hxx" 
#include "recorded_warnings.hxx" 
#include "print_tools.hxx" 

namespace green_potential {

template <typename real_t, int R1C2=2, int Noco=1>
void __global__ Potential( 
#ifdef    HAS_NO_CUDA
dim3 const & gridDim, dim3 const & blockDim,
#endif 
real_t        (*const __restrict__ Vpsi)[R1C2][Noco*64][Noco*64] 
, real_t  const (*const __restrict__  psi)[R1C2][Noco*64][Noco*64] 
, double  const (*const *const __restrict__ Vloc)[64] 
, int32_t const (*const __restrict__ iloc_of_inzb) 
, int16_t const (*const __restrict__ shift)[3+1] 
, double  const (*const __restrict__ hxyz) 
, int     const nnzb 
, float   const Vconf 
, float   const rcut2 
, real_t  const E_real 
, real_t  const E_imag 
) {
assert((1 == Noco && (1 == R1C2 || 2 == R1C2)) || (2 == Noco && 2 == R1C2));

assert(64      ==  gridDim.x);
assert(1       ==  gridDim.z);
assert(Noco*64 == blockDim.x);
assert(Noco    == blockDim.y);
assert(R1C2    == blockDim.z);

bool const imaginary = ((2 == R1C2) && (0 != E_imag));

#ifndef HAS_NO_CUDA
int const inz0 = blockIdx.y;  
int const i64  = blockIdx.x;  
int const reim = threadIdx.z; 
int const spin = threadIdx.y; 
int const j64  = threadIdx.x; 
#else  
assert(1 == gridDim.y && "CPU kernel Potential needs increment 1 for grid stride loop");
int constexpr inz0 = 0;
for (int i64 = 0; i64 < 64; ++i64)
for (int reim = 0; reim < R1C2; ++reim)
for (int spin = 0; spin < Noco; ++spin)
for (int j64 = 0; j64 < Noco*64; ++j64)
#endif 
{ 

#ifdef  CONFINEMENT_POTENTIAL
int const x = ( i64       & 0x3) - ( j64       & 0x3);
int const y = ((i64 >> 2) & 0x3) - ((j64 >> 2) & 0x3);
int const z = ((i64 >> 4) & 0x3) - ((j64 >> 4) & 0x3);
#endif 

assert(R1C2 >= Noco); 

auto const V_imag = E_imag * real_t(1 - 2*reim);

for (int inzb = inz0; inzb < nnzb; inzb += gridDim.y) { 

real_t Vconfine = 0; 
#ifdef  CONFINEMENT_POTENTIAL
if (rcut2 >= 0.f) {
int constexpr n4 = 4;
auto const s = shift[inzb]; 
auto const d2 = pow2((int(s[0])*n4 + x)*real_t(hxyz[0]))
+ pow2((int(s[1])*n4 + y)*real_t(hxyz[1]))
+ pow2((int(s[2])*n4 + z)*real_t(hxyz[2]));
auto const d2out = real_t(d2 - rcut2);
Vconfine = (d2out > 0) ? Vconf*pow2(d2out) : 0; 
} 
#endif 

auto const iloc = iloc_of_inzb[inzb]; 
real_t const Vloc_diag = (iloc < 0) ? 0 : Vloc[spin][iloc][i64];

real_t const Vtot = Vloc_diag + Vconfine - E_real; 

auto vpsi = Vtot * psi[inzb][reim][spin*64 + i64][j64]; 

if (imaginary) {
vpsi += V_imag * psi[inzb][1 - reim][spin*64 + i64][j64];
} 

if (2 == Noco && iloc >= 0) { 




















real_t const cs = (1 - 2*(reim ^ spin)); 

vpsi += Vloc[2][iloc][i64] * psi[inzb][    reim][(1 - spin)*64 + i64][j64];    
vpsi += Vloc[3][iloc][i64] * psi[inzb][1 - reim][(1 - spin)*64 + i64][j64]*cs; 
} 

Vpsi[inzb][reim][spin*64 + i64][j64] = vpsi; 

} 

} 

} 


template <typename real_t, int R1C2=2, int Noco=1>
size_t multiply(
real_t         (*const __restrict__ Vpsi)[R1C2][Noco*64][Noco*64] 
, real_t   const (*const __restrict__  psi)[R1C2][Noco*64][Noco*64] 
, double   const (*const *const __restrict__ Vloc)[64] 
, int32_t  const (*const __restrict__ vloc_index) 
, int16_t  const (*const __restrict__ shift)[3+1] 
, double   const (*const __restrict__ hxyz) 
, uint32_t const nnzb 
, std::complex<double> const E_param=0 
, float    const Vconf=0  
, float    const rcut2=-1 
, int const echo=0
) {

if (echo > 11) {
std::printf("# %s<%s,R1C2=%d,Noco=%d> Vpsi=%p, psi=%p, Vloc=%p, vloc_index=%p, shift=%p, hxyz=%p, nnzb=%d, Vconf=%g, rcut2=%.f, E=(%g, %g)\n",
__func__, (4 == sizeof(real_t))?"float":"double", R1C2, Noco, (void*)Vpsi, (void*)psi,
(void*)Vloc, (void*)vloc_index, (void*)shift, (void*)hxyz, nnzb, Vconf, rcut2, E_param.real(), E_param.imag());
} 

Potential<real_t,R1C2,Noco>
#ifndef   HAS_NO_CUDA
<<< dim3(64, 7, 1), dim3(Noco*64, Noco, R1C2) >>> ( 
#else  
( dim3(64, 1, 1), dim3(Noco*64, Noco, R1C2),
#endif 
Vpsi, psi, Vloc, vloc_index, shift, hxyz, nnzb, Vconf, rcut2, E_param.real(), E_param.imag());

return (  1ul
+  2ul*(0 != E_param.imag())
+  4ul*(2 == Noco)
#ifdef CONFINEMENT_POTENTIAL
+ 10ul*(rcut2 >= 0.f)
#endif 
)*nnzb*pow2(64ul*Noco)*R1C2; 
} 



#ifdef    NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else  

template <typename real_t, int R1C2=2, int Noco=1>
inline status_t test_multiply(
double   const (*const *const __restrict__ Vloc)[64] 
, int32_t  const (*const __restrict__ vloc_index) 
, int16_t  const (*const __restrict__ shift)[3+1] 
, double   const (*const __restrict__ hxyz) 
, uint32_t const nnzb=1
, int const echo=0
) {
auto psi   = get_memory<real_t[R1C2][Noco*64][Noco*64]>(nnzb, echo, "psi");
auto Vpsi  = get_memory<real_t[R1C2][Noco*64][Noco*64]>(nnzb, echo, "Vpsi");

if (echo > 5) std::printf("# %s<real_t=%s,R1C2=%d,Noco=%d>(Vpsi=%p, psi=%p, ...)\n",
__func__, (8 == sizeof(real_t)) ? "double" : "float", R1C2, Noco, (void*)Vpsi, (void*)psi);
multiply<real_t,R1C2,Noco>(Vpsi, psi, Vloc, vloc_index, shift, hxyz, nnzb);

free_memory(Vpsi);
free_memory(psi);
return 0;
} 

inline status_t test_multiply(int const echo=0, int const Noco=2) {
status_t stat(0);
uint32_t const nnzb = 1;
auto Vloc = get_memory<double(*)[64]>(Noco*Noco, echo, "Vloc");
for (int mag = 0; mag < Noco*Noco; ++mag) Vloc[mag] = get_memory<double[64]>(1, echo, "Vloc[mag]");
auto vloc_index = get_memory<int32_t>(1, echo, "vloc_index"); vloc_index[0] = 0;
auto shift = get_memory<int16_t[3+1]>(1, echo, "shift");      set(shift[0], 3+1, int16_t(0));
auto hxyz = get_memory<double>(3+1, echo, "hxyz");            set(hxyz, 3+1, 1.);

stat += test_multiply<float ,1,1>(Vloc, vloc_index, shift, hxyz, nnzb, echo);
stat += test_multiply<float ,2,1>(Vloc, vloc_index, shift, hxyz, nnzb, echo);
stat += test_multiply<float ,2,2>(Vloc, vloc_index, shift, hxyz, nnzb, echo);
stat += test_multiply<double,1,1>(Vloc, vloc_index, shift, hxyz, nnzb, echo);
stat += test_multiply<double,2,1>(Vloc, vloc_index, shift, hxyz, nnzb, echo);
stat += test_multiply<double,2,2>(Vloc, vloc_index, shift, hxyz, nnzb, echo);

free_memory(hxyz);
free_memory(shift);
free_memory(vloc_index);
for (int mag = 0; mag < Noco*Noco; ++mag) free_memory(Vloc[mag]);
free_memory(Vloc);
return stat;
} 

inline status_t all_tests(int const echo=0) {
status_t stat(0);
stat += test_multiply(echo);
return stat;
} 

#endif 

} 

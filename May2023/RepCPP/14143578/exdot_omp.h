


#pragma once
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>

#include "accumulate.h"
#include "ExSUM.FPE.hpp"
#include <omp.h>

namespace dg
{
namespace exblas{
namespace cpu{



inline static void ReductionStep(int step, int64_t * acc1, int64_t * acc2,
int volatile * ready)
{
#ifndef _WITHOUT_VCL
_mm_prefetch((char const*)ready, _MM_HINT_T0);
while(*ready < step) {
_mm_pause();
}
#endif
int imin = IMIN, imax = IMAX;
Normalize( acc1, imin, imax);
imin = IMIN, imax = IMAX;
Normalize( acc2, imin, imax);
for(int i = IMIN; i <= IMAX; ++i) {
acc1[i] += acc2[i];
}
}


inline static void Reduction(unsigned int tid, unsigned int tnum, std::vector<int32_t>& ready,
std::vector<int64_t>& acc, int const linesize)
{
for(unsigned int s = 1; (unsigned)(1 << (s-1)) < tnum; ++s)
{
int32_t volatile * c = &ready[tid * linesize];
++*c; 
#ifdef _WITHOUT_VCL
#pragma omp barrier 
#endif
if(tid % (1 << s) == 0) { 
unsigned int tid2 = tid | (1 << (s-1)); 
if(tid2 < tnum) {
ReductionStep(s, &acc[tid*BIN_COUNT], &acc[tid2*BIN_COUNT],
&ready[tid2 * linesize]);
}
}
}
}

template<typename CACHE, typename PointerOrValue1, typename PointerOrValue2>
void ExDOTFPE(int N, PointerOrValue1 a, PointerOrValue2 b, int64_t* h_superacc, bool* err) {
int const linesize = 16;    
int maxthreads = omp_get_max_threads();
std::vector<int64_t> acc(maxthreads*BIN_COUNT,0);
std::vector<int32_t> ready(maxthreads * linesize);
std::vector<bool> error( maxthreads, false);

#pragma omp parallel
{
unsigned int tid = omp_get_thread_num();
unsigned int tnum = omp_get_num_threads();

CACHE cache(&acc[tid*BIN_COUNT]);
*(int32_t volatile *)(&ready[tid * linesize]) = 0;  

#ifndef _WITHOUT_VCL
int l = ((tid * int64_t(N)) / tnum) & ~7ul; 
int r = ((((tid+1) * int64_t(N)) / tnum) & ~7ul) - 1;

for(int i = l; i < r; i+=8) {
#ifndef _MSC_VER
asm ("# myloop");
#endif
vcl::Vec8d x  = make_vcl_vec8d(a,i)*make_vcl_vec8d(b,i);
vcl::Vec8db finite = vcl::is_finite( x);
if( !vcl::horizontal_and( finite) ) error[tid] = true;

cache.Accumulate(x);
}
if( tid+1==tnum && r != N-1) {
r+=1;
vcl::Vec8d x  = make_vcl_vec8d(a,r,N-r)*make_vcl_vec8d(b,r,N-r);

vcl::Vec8db finite = vcl::is_finite( x);
if( !vcl::horizontal_and( finite) ) error[tid] = true;
cache.Accumulate(x);
}
#else
int l = ((tid * int64_t(N)) / tnum);
int r = ((((tid+1) * int64_t(N)) / tnum) ) - 1;
for(int i = l; i <= r; i++) {
double x = get_element(a,i)*get_element(b,i);
cache.Accumulate(x);
}
#endif
cache.Flush();
int imin=IMIN, imax=IMAX;
Normalize(&acc[tid*BIN_COUNT], imin, imax);

Reduction(tid, tnum, ready, acc, linesize);
}
for( int i=IMIN; i<=IMAX; i++)
h_superacc[i] = acc[i];
for ( int i=0; i<maxthreads; i++)
if( error[i] == true) *err = true;
}

template<typename CACHE, typename PointerOrValue1, typename PointerOrValue2, typename PointerOrValue3>
void ExDOTFPE(int N, PointerOrValue1 a, PointerOrValue2 b, PointerOrValue3 c, int64_t* h_superacc, bool* err) {
int const linesize = 16;    
int maxthreads = omp_get_max_threads();
std::vector<int64_t> acc(maxthreads*BIN_COUNT,0);
std::vector<int32_t> ready(maxthreads * linesize);
std::vector<bool> error( maxthreads, false);

#pragma omp parallel
{
unsigned int tid = omp_get_thread_num();
unsigned int tnum = omp_get_num_threads();

CACHE cache(&acc[tid*BIN_COUNT]);
*(int32_t volatile *)(&ready[tid * linesize]) = 0;  

#ifndef _WITHOUT_VCL
int l = ((tid * int64_t(N)) / tnum) & ~7ul;
int r = ((((tid+1) * int64_t(N)) / tnum) & ~7ul) - 1;

for(int i = l; i < r; i+=8) {
#ifndef _MSC_VER
asm ("# myloop");
#endif
vcl::Vec8d x1  = make_vcl_vec8d(a,i)*make_vcl_vec8d(b,i);
vcl::Vec8d x2  =  x1                *make_vcl_vec8d(c,i);
vcl::Vec8db finite = vcl::is_finite( x2);
if( !vcl::horizontal_and( finite) ) error[tid] = true;
cache.Accumulate(x2);
}
if( tid+1 == tnum && r != N-1) {
r+=1;
vcl::Vec8d x1  = make_vcl_vec8d(a,r,N-r)*make_vcl_vec8d(b,r,N-r);
vcl::Vec8d x2  =  x1                    *make_vcl_vec8d(c,r,N-r);
vcl::Vec8db finite = vcl::is_finite( x2);
if( !vcl::horizontal_and( finite) ) error[tid] = true;
cache.Accumulate(x2);
}
#else
int l = ((tid * int64_t(N)) / tnum);
int r = ((((tid+1) * int64_t(N)) / tnum) ) - 1;
for(int i = l; i <= r; i++) {
double x1 = get_element(a,i)*get_element(b,i);
double x2 = x1*get_element(c,i);
cache.Accumulate(x2);
}
#endif
cache.Flush();
int imin=IMIN, imax=IMAX;
Normalize(&acc[tid*BIN_COUNT], imin, imax);

Reduction(tid, tnum, ready, acc, linesize);
}
for( int i=IMIN; i<=IMAX; i++)
h_superacc[i] = acc[i];
for ( int i=0; i<maxthreads; i++)
if( error[i] == true) *err = true;
}
}

template<class PointerOrValue1, class PointerOrValue2, size_t NBFPE=8>
void exdot_omp(unsigned size, PointerOrValue1 x1_ptr, PointerOrValue2 x2_ptr, int64_t* h_superacc, int* status){
static_assert( has_floating_value<PointerOrValue1>::value, "PointerOrValue1 needs to be T or T* with T one of (const) float or (const) double");
static_assert( has_floating_value<PointerOrValue2>::value, "PointerOrValue2 needs to be T or T* with T one of (const) float or (const) double");
bool error = false;
#ifndef _WITHOUT_VCL
cpu::ExDOTFPE<cpu::FPExpansionVect<vcl::Vec8d, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, h_superacc, &error);
#else
cpu::ExDOTFPE<cpu::FPExpansionVect<double, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, h_superacc, &error);
#endif
*status = 0;
if( error ) *status = 1;
}
template<class PointerOrValue1, class PointerOrValue2, class PointerOrValue3, size_t NBFPE=8>
void exdot_omp(unsigned size, PointerOrValue1 x1_ptr, PointerOrValue2 x2_ptr, PointerOrValue3 x3_ptr, int64_t* h_superacc, int* status) {
static_assert( has_floating_value<PointerOrValue1>::value, "PointerOrValue1 needs to be T or T* with T one of (const) float or (const) double");
static_assert( has_floating_value<PointerOrValue2>::value, "PointerOrValue2 needs to be T or T* with T one of (const) float or (const) double");
static_assert( has_floating_value<PointerOrValue3>::value, "PointerOrValue3 needs to be T or T* with T one of (const) float or (const) double");
bool error = false;
#ifndef _WITHOUT_VCL
cpu::ExDOTFPE<cpu::FPExpansionVect<vcl::Vec8d, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, x3_ptr, h_superacc, &error);
#else
cpu::ExDOTFPE<cpu::FPExpansionVect<double, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, x3_ptr, h_superacc, &error);
#endif
*status = 0;
if( error ) *status = 1;
}

}
} 

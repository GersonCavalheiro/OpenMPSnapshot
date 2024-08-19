


#pragma once
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>

#include "accumulate.h"
#include "ExSUM.FPE.hpp"

namespace dg
{
namespace exblas{


union udouble{
double d; 
int64_t i; 
};

namespace cpu{

template<typename CACHE, typename PointerOrValue1, typename PointerOrValue2>
void ExDOTFPE_cpu(int N, PointerOrValue1 a, PointerOrValue2 b, int64_t* acc, bool* error) {
CACHE cache(acc);
#ifndef _WITHOUT_VCL
int r = (( int64_t(N) ) & ~7ul);
for(int i = 0; i < r; i+=8) {
#ifndef _MSC_VER
asm ("# myloop");
#endif
vcl::Vec8d x  = make_vcl_vec8d(a,i)* make_vcl_vec8d(b,i);
vcl::Vec8db finite = vcl::is_finite( x);
if( !vcl::horizontal_and( finite) ) *error = true;
cache.Accumulate(x);
}
if( r != N) {
vcl::Vec8d x  = make_vcl_vec8d(a,r,N-r)*make_vcl_vec8d(b,r,N-r);
vcl::Vec8db finite = vcl::is_finite( x);
if( !vcl::horizontal_and( finite) ) *error = true;
cache.Accumulate(x);
}
#else
for(int i = 0; i < N; i++) {
double x = get_element(a,i)*get_element(b,i);
if( !std::isfinite(x) ) *error = true;
cache.Accumulate(x);
}
#endif
cache.Flush();
}

template<typename CACHE, typename PointerOrValue1, typename PointerOrValue2, typename PointerOrValue3>
void ExDOTFPE_cpu(int N, PointerOrValue1 a, PointerOrValue2 b, PointerOrValue3 c, int64_t* acc, bool* error) {
CACHE cache(acc);
#ifndef _WITHOUT_VCL
int r = (( int64_t(N))  & ~7ul);
for(int i = 0; i < r; i+=8) {
#ifndef _MSC_VER
asm ("# myloop");
#endif
vcl::Vec8d x1  = vcl::mul_add(make_vcl_vec8d(a,i),make_vcl_vec8d(b,i), 0);
vcl::Vec8d x2  = vcl::mul_add( x1                ,make_vcl_vec8d(c,i), 0);
vcl::Vec8db finite = vcl::is_finite( x2);
if( !vcl::horizontal_and( finite) ) *error = true;
cache.Accumulate(x2);
}
if( r != N) {
vcl::Vec8d x1  = vcl::mul_add(make_vcl_vec8d(a,r,N-r),make_vcl_vec8d(b,r,N-r), 0);
vcl::Vec8d x2  = vcl::mul_add( x1                    ,make_vcl_vec8d(c,r,N-r), 0);
vcl::Vec8db finite = vcl::is_finite( x2);
if( !vcl::horizontal_and( finite) ) *error = true;
cache.Accumulate(x2);
}
#else
for(int i = 0; i < N; i++) {
double x1 = get_element(a,i)*get_element(b,i);
double x2 = x1*get_element(c,i);
if( !std::isfinite(x2) ) *error = true;
cache.Accumulate(x2);
}
#endif
cache.Flush();
}
}







template<class PointerOrValue1, class PointerOrValue2, size_t NBFPE=8>
void exdot_cpu(unsigned size, PointerOrValue1 x1_ptr, PointerOrValue2 x2_ptr, int64_t* h_superacc, int* status){
static_assert( has_floating_value<PointerOrValue1>::value, "PointerOrValue1 needs to be T or T* with T one of (const) float or (const) double");
static_assert( has_floating_value<PointerOrValue2>::value, "PointerOrValue2 needs to be T or T* with T one of (const) float or (const) double");
for( int i=0; i<exblas::BIN_COUNT; i++)
h_superacc[i] = 0;
bool error = false;
#ifndef _WITHOUT_VCL
cpu::ExDOTFPE_cpu<cpu::FPExpansionVect<vcl::Vec8d, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, h_superacc, &error);
#else
cpu::ExDOTFPE_cpu<cpu::FPExpansionVect<double, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, h_superacc, &error);
#endif
*status = 0;
if( error ) *status = 1;
}

template<class PointerOrValue1, class PointerOrValue2, class PointerOrValue3, size_t NBFPE=8>
void exdot_cpu(unsigned size, PointerOrValue1 x1_ptr, PointerOrValue2 x2_ptr, PointerOrValue3 x3_ptr, int64_t* h_superacc, int* status) {
static_assert( has_floating_value<PointerOrValue1>::value, "PointerOrValue1 needs to be T or T* with T one of (const) float or (const) double");
static_assert( has_floating_value<PointerOrValue2>::value, "PointerOrValue2 needs to be T or T* with T one of (const) float or (const) double");
static_assert( has_floating_value<PointerOrValue3>::value, "PointerOrValue3 needs to be T or T* with T one of (const) float or (const) double");
for( int i=0; i<exblas::BIN_COUNT; i++)
h_superacc[i] = 0;
bool error = false;
#ifndef _WITHOUT_VCL
cpu::ExDOTFPE_cpu<cpu::FPExpansionVect<vcl::Vec8d, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, x3_ptr, h_superacc, &error);
#else
cpu::ExDOTFPE_cpu<cpu::FPExpansionVect<double, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, x3_ptr, h_superacc, &error);
#endif
*status = 0;
if( error ) *status = 1;
}



}
} 

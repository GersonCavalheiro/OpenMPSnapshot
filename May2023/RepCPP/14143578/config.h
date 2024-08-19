
#pragma once

#include <cstdint> 
#include <cmath>
#include <cassert>
#ifdef __NVCC__
#define _WITHOUT_VCL
#endif 
#ifdef __APPLE__ 
#define _WITHOUT_VCL
#endif
#ifdef WITHOUT_VCL
#define _WITHOUT_VCL
#endif

#ifndef _WITHOUT_VCL

#define MAX_VECTOR_SIZE 512 
#define VCL_NAMESPACE vcl
#include "vcl/vectorclass.h" 
#if INSTRSET <5
#define _WITHOUT_VCL
#pragma message("WARNING: Instruction set below SSE4.1! Deactivating vectorization!")
#elif INSTRSET <7
#pragma message( "NOTE: If available, it is recommended to activate AVX instruction set (-mavx) or higher")
#endif

#endif

#if defined __INTEL_COMPILER
#define UNROLL_ATTRIBUTE
#elif defined __GNUC__

#ifdef __APPLE__ 
#define UNROLL_ATTRIBUTE
#else
#define UNROLL_ATTRIBUTE __attribute__((optimize("unroll-loops")))
#endif 

#else
#define UNROLL_ATTRIBUTE
#endif

#ifdef ATT_SYNTAX
#define ASM_BEGIN ".intel_syntax;"
#define ASM_END ";.att_syntax"
#else
#define ASM_BEGIN
#define ASM_END
#endif

#define paranoid_assert(x) assert(x)
#if not defined _MSC_VER 
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

namespace dg
{
namespace exblas
{
static constexpr int KRX            =  8; 
static constexpr int DIGITS         =  64 - KRX; 
static constexpr int F_WORDS        =  20;  
static constexpr int E_WORDS        =  19;  
static constexpr int BIN_COUNT     =  F_WORDS+E_WORDS; 
static constexpr int IMIN           = 0; 
static constexpr int IMAX           = BIN_COUNT-1; 
static constexpr double DELTASCALE = double(1ull << DIGITS); 

enum Status
{
Exact, 
Inexact, 
MinusInfinity, 
PlusInfinity, 
Overflow, 
sNaN, 
qNaN 
};

template<class T>
struct ValueTraits
{
using value_type = T;
};
template<class T>
struct ValueTraits<T*>
{
using value_type = T;
};
template<class U>
using has_floating_value = std::conditional_t< std::is_floating_point<typename ValueTraits<U>::value_type>::value, std::true_type, std::false_type>;

}
} 

#pragma once

#ifndef __SYS_Types__
#define __SYS_Types__


#include <limits>
#include <type_traits>
#include <sys/types.h>


typedef signed char	int8;
typedef	unsigned char	uint8;
typedef short		int16;
typedef unsigned short	uint16;
typedef	int		int32;
typedef unsigned int	uint32;

#ifndef MBSD
typedef unsigned int	uint;
#endif


#if defined(_WIN32)
typedef __int64		int64;
typedef unsigned __int64	uint64;
#elif defined(MBSD)
#include <stdint.h>
typedef int64_t		int64;
typedef uint64_t		uint64;
#elif defined(AMD64)
typedef long		int64;
typedef unsigned long	uint64;
#else
typedef long long		int64;
typedef unsigned long long	uint64;
#endif

typedef int64 exint;

#if defined(__GNUC__) || defined(__clang__)
#define SYS_FORCE_INLINE	__attribute__ ((always_inline)) inline
#elif defined(_MSC_VER)
#define SYS_FORCE_INLINE	__forceinline
#else
#define SYS_FORCE_INLINE	inline
#endif

typedef float   fpreal32;
typedef double  fpreal64;

template <typename T>
union SYS_FPRealUnionT;

template <>
union SYS_FPRealUnionT<fpreal32>
{
typedef int32	int_type;
typedef uint32	uint_type;
typedef fpreal32	fpreal_type;

enum {
EXPONENT_BITS = 8,
MANTISSA_BITS = 23,
EXPONENT_BIAS = 127 };

int_type		ival;
uint_type		uval;
fpreal_type		fval;

struct
{
uint_type mantissa_val: 23;
uint_type exponent_val: 8;
uint_type sign_val: 1;
};
};

template <>
union SYS_FPRealUnionT<fpreal64>
{
typedef int64	int_type;
typedef uint64	uint_type;
typedef fpreal64	fpreal_type;

enum {
EXPONENT_BITS = 11,
MANTISSA_BITS = 52,
EXPONENT_BIAS = 1023 };

int_type		ival;
uint_type		uval;
fpreal_type		fval;

struct
{
uint_type mantissa_val: 52;
uint_type exponent_val: 11;
uint_type sign_val: 1;
};
};

typedef union SYS_FPRealUnionT<fpreal32>    SYS_FPRealUnionF;
typedef union SYS_FPRealUnionT<fpreal64>    SYS_FPRealUnionD;

#define UT_ASSERT_P(ZZ)         ((void)0)
#define UT_ASSERT(ZZ)           ((void)0)
#define UT_ASSERT_MSG_P(ZZ, MM) ((void)0)
#define UT_ASSERT_MSG(ZZ, MM)   ((void)0)

#endif

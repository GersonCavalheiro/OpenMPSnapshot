#pragma GCC target ("fpu=vfp")
#pragma GCC push_options
#pragma GCC target ("fpu=fp-armv8")
#ifndef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
#error __ARM_FEATURE_FP16_SCALAR_ARITHMETIC not defined.
#endif
#pragma GCC push_options
#pragma GCC target ("fpu=neon-fp-armv8")
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#error __ARM_FEATURE_FP16_VECTOR_ARITHMETIC not defined.
#endif
#ifndef __ARM_NEON
#error __ARM_NEON not defined.
#endif
#if !defined (__ARM_FP) || !(__ARM_FP & 0x2)
#error Invalid value for __ARM_FP
#endif
#include "arm_neon.h"
float16_t
foo (float16x4_t b)
{
float16x4_t a = {2.0, 3.0, 4.0, 5.0};
float16x4_t res = vadd_f16 (a, b);
return res[0];
}
#pragma GCC pop_options
#if !defined (__ARM_FP) || !(__ARM_FP & 0x2)
#error __ARM_FP should record FP16 support.
#endif
#pragma GCC pop_options
#if !defined (__ARM_FP) || (__ARM_FP & 0x2)
#error Unexpected value for __ARM_FP.
#endif

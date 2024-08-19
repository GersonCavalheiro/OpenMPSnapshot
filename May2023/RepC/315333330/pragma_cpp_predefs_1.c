#pragma GCC push_options
#pragma GCC target ("arch=armv8-a+nofp+nosimd")
#ifdef __ARM_FEATURE_FMA
#error "__ARM_FEATURE_FMA is defined but should not be!"
#endif
#ifdef __ARM_FP
#error "__ARM_FP is defined but should not be!"
#endif
#pragma GCC push_options
#pragma GCC target ("arch=armv8-a+fp+nosimd")
#ifndef __ARM_FP
#error "__ARM_FP is not defined but should be!"
#endif
#ifdef __ARM_NEON
#error "__ARM_NEON is defined but should not be!"
#endif
#pragma GCC push_options
#pragma GCC target ("arch=armv8-a+fp+simd")
#ifndef __ARM_NEON
#error "__ARM_NEON is not defined but should be!"
#endif
#ifdef __ARM_FEATURE_CRYPTO
#error "__ARM_FEATURE_CRYPTO is defined but should not be!"
#endif
#pragma GCC push_options
#pragma GCC target ("arch=armv8-a+fp+simd+crypto")
#ifndef __ARM_FEATURE_CRYPTO
#error "__ARM_FEATURE_CRYPTO is not defined but should be!"
#endif
#pragma GCC pop_options
#ifndef __ARM_NEON
#error "__ARM_NEON is not defined but should be!"
#endif
#ifdef __ARM_FEATURE_CRYPTO
#error "__ARM_FEATURE_CRYPTO is defined but should not be!"
#endif
#pragma GCC pop_options
#ifndef __ARM_FP
#error "__ARM_FP is not defined but should be!"
#endif
#ifdef __ARM_NEON
#error "__ARM_NEON is defined but should not be!"
#endif
#pragma GCC pop_options
#ifdef __ARM_FP
#error "__ARM_FP is defined but should not be!"
#endif
#ifdef __ARM_NEON
#error "__ARM_NEON is defined but should not be!"
#endif
#pragma GCC push_options
#pragma GCC target ("cpu=cortex-a53+nofp+nosimd")
#ifdef __ARM_FEATURE_FMA
#error "__ARM_FEATURE_FMA is defined but should not be!"
#endif
#ifdef __ARM_FP
#error "__ARM_FP is defined but should not be!"
#endif
#pragma GCC push_options
#pragma GCC target ("cpu=cortex-a53+fp+nosimd")
#ifndef __ARM_FP
#error "__ARM_FP is not defined but should be!"
#endif
#ifdef __ARM_NEON
#error "__ARM_NEON is defined but should not be!"
#endif
#pragma GCC push_options
#pragma GCC target ("cpu=cortex-a53+fp+simd+nocrypto")
#ifndef __ARM_NEON
#error "__ARM_NEON is not defined but should be!"
#endif
#ifdef __ARM_FEATURE_CRYPTO
#error "__ARM_FEATURE_CRYPTO is defined but should not be!"
#endif
#pragma GCC push_options
#pragma GCC target ("cpu=cortex-a53+fp+simd+crypto")
#ifndef __ARM_FEATURE_CRYPTO
#error "__ARM_FEATURE_CRYPTO is not defined but should be!"
#endif
#pragma GCC pop_options
#ifndef __ARM_NEON
#error "__ARM_NEON is not defined but should be!"
#endif
#ifdef __ARM_FEATURE_CRYPTO
#error "__ARM_FEATURE_CRYPTO is defined but should not be!"
#endif
#pragma GCC pop_options
#ifndef __ARM_FP
#error "__ARM_FP is not defined but should be!"
#endif
#ifdef __ARM_NEON
#error "__ARM_NEON is defined but should not be!"
#endif
#pragma GCC pop_options
#ifdef __ARM_FP
#error "__ARM_FP is defined but should not be!"
#endif
#ifdef __ARM_NEON
#error "__ARM_NEON is defined but should not be!"
#endif
#pragma GCC push_options
#pragma GCC target ("+nofp")
#ifdef __ARM_FEATURE_FMA
#error "__ARM_FEATURE_FMA is defined but should not be!"
#endif
#ifdef __ARM_FP
#error "__ARM_FP is defined but should not be!"
#endif
#pragma GCC push_options
#pragma GCC target ("+fp+nosimd")
#ifndef __ARM_FP
#error "__ARM_FP is not defined but should be!"
#endif
#ifdef __ARM_NEON
#error "__ARM_NEON is defined but should not be!"
#endif
#pragma GCC push_options
#pragma GCC target ("+fp+simd+nocrypto")
#ifndef __ARM_NEON
#error "__ARM_NEON is not defined but should be!"
#endif
#ifdef __ARM_FEATURE_CRYPTO
#error "__ARM_FEATURE_CRYPTO is defined but should not be!"
#endif
#pragma GCC push_options
#pragma GCC target ("+fp+simd+crypto")
#ifndef __ARM_FEATURE_CRYPTO
#error "__ARM_FEATURE_CRYPTO is not defined but should be!"
#endif
#pragma GCC pop_options
#ifndef __ARM_NEON
#error "__ARM_NEON is not defined but should be!"
#endif
#ifdef __ARM_FEATURE_CRYPTO
#error "__ARM_FEATURE_CRYPTO is defined but should not be!"
#endif
#pragma GCC pop_options
#ifndef __ARM_FP
#error "__ARM_FP is not defined but should be!"
#endif
#ifdef __ARM_NEON
#error "__ARM_NEON is defined but should not be!"
#endif
#pragma GCC pop_options
#ifdef __ARM_FP
#error "__ARM_FP is defined but should not be!"
#endif
#ifdef __ARM_NEON
#error "__ARM_NEON is defined but should not be!"
#endif
#pragma GCC push_options
#pragma GCC target ("general-regs-only")
#ifdef __ARM_FEATURE_FMA
#error "__ARM_FEATURE_FMA is defined but should not be!"
#endif
#ifdef __ARM_FP
#error "__ARM_FP is defined but should not be!"
#endif
#ifdef __ARM_NEON
#error "__ARM_NEON is defined but should not be!"
#endif
#ifdef __ARM_FEATURE_CRYPTO
#error "__ARM_FEATURE_CRYPTO is defined but should not be!"
#endif
#pragma GCC pop_options
#pragma GCC target ("+nocrc")
#ifdef __ARM_FEATURE_CRC32
#error "__ARM_FEATURE_CRC32 is defined but should not be!"
#endif
#pragma GCC target ("+crc")
#ifndef __ARM_FEATURE_CRC32
#error "__ARM_FEATURE_CRC32 is not defined but should be!"
#endif
int
foo (int a)
{
return a;
}

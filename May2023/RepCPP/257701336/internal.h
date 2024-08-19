



#ifndef OPENSSL_HEADER_BN_INTERNAL_H
#define OPENSSL_HEADER_BN_INTERNAL_H

#include <openssl/base.h>

#if defined(OPENSSL_X86_64) && defined(_MSC_VER)
OPENSSL_MSVC_PRAGMA(warning(push, 3))
#include <intrin.h>
OPENSSL_MSVC_PRAGMA(warning(pop))
#pragma intrinsic(__umulh, _umul128)
#endif

#include "../../internal.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(OPENSSL_64_BIT)

#if defined(BORINGSSL_HAS_UINT128)
#define BN_ULLONG uint128_t
#if defined(BORINGSSL_CAN_DIVIDE_UINT128)
#define BN_CAN_DIVIDE_ULLONG
#endif
#endif

#define BN_BITS2 64
#define BN_BYTES 8
#define BN_BITS4 32
#define BN_MASK2 (0xffffffffffffffffUL)
#define BN_MASK2l (0xffffffffUL)
#define BN_MASK2h (0xffffffff00000000UL)
#define BN_MASK2h1 (0xffffffff80000000UL)
#define BN_MONT_CTX_N0_LIMBS 1
#define BN_DEC_CONV (10000000000000000000UL)
#define BN_DEC_NUM 19
#define TOBN(hi, lo) ((BN_ULONG)(hi) << 32 | (lo))

#elif defined(OPENSSL_32_BIT)

#define BN_ULLONG uint64_t
#define BN_CAN_DIVIDE_ULLONG
#define BN_BITS2 32
#define BN_BYTES 4
#define BN_BITS4 16
#define BN_MASK2 (0xffffffffUL)
#define BN_MASK2l (0xffffUL)
#define BN_MASK2h1 (0xffff8000UL)
#define BN_MASK2h (0xffff0000UL)
#define BN_MONT_CTX_N0_LIMBS 2
#define BN_DEC_CONV (1000000000UL)
#define BN_DEC_NUM 9
#define TOBN(hi, lo) (lo), (hi)

#else
#error "Must define either OPENSSL_32_BIT or OPENSSL_64_BIT"
#endif

#define MOD_EXP_CTIME_MIN_CACHE_LINE_WIDTH 64

#define MOD_EXP_CTIME_STORAGE_LEN \
(((320u * 3u) + (32u * 9u * 16u)) / sizeof(BN_ULONG))

#define STATIC_BIGNUM(x)                                    \
{                                                         \
(BN_ULONG *)(x), sizeof(x) / sizeof(BN_ULONG),          \
sizeof(x) / sizeof(BN_ULONG), 0, BN_FLG_STATIC_DATA \
}

#if defined(BN_ULLONG)
#define Lw(t) ((BN_ULONG)(t))
#define Hw(t) ((BN_ULONG)((t) >> BN_BITS2))
#endif

int bn_minimal_width(const BIGNUM *bn);

void bn_set_minimal_width(BIGNUM *bn);

int bn_wexpand(BIGNUM *bn, size_t words);

int bn_expand(BIGNUM *bn, size_t bits);

OPENSSL_EXPORT int bn_resize_words(BIGNUM *bn, size_t words);

void bn_select_words(BN_ULONG *r, BN_ULONG mask, const BN_ULONG *a,
const BN_ULONG *b, size_t num);

int bn_set_words(BIGNUM *bn, const BN_ULONG *words, size_t num);

int bn_fits_in_words(const BIGNUM *bn, size_t num);

int bn_copy_words(BN_ULONG *out, size_t num, const BIGNUM *bn);

BN_ULONG bn_mul_add_words(BN_ULONG *rp, const BN_ULONG *ap, size_t num,
BN_ULONG w);

BN_ULONG bn_mul_words(BN_ULONG *rp, const BN_ULONG *ap, size_t num, BN_ULONG w);

void bn_sqr_words(BN_ULONG *rp, const BN_ULONG *ap, size_t num);

BN_ULONG bn_add_words(BN_ULONG *rp, const BN_ULONG *ap, const BN_ULONG *bp,
size_t num);

BN_ULONG bn_sub_words(BN_ULONG *rp, const BN_ULONG *ap, const BN_ULONG *bp,
size_t num);

void bn_mul_comba4(BN_ULONG r[8], const BN_ULONG a[4], const BN_ULONG b[4]);

void bn_mul_comba8(BN_ULONG r[16], const BN_ULONG a[8], const BN_ULONG b[8]);

void bn_sqr_comba8(BN_ULONG r[16], const BN_ULONG a[4]);

void bn_sqr_comba4(BN_ULONG r[8], const BN_ULONG a[4]);

int bn_less_than_words(const BN_ULONG *a, const BN_ULONG *b, size_t len);

int bn_in_range_words(const BN_ULONG *a, BN_ULONG min_inclusive,
const BN_ULONG *max_exclusive, size_t len);

int bn_rand_range_words(BN_ULONG *out, BN_ULONG min_inclusive,
const BN_ULONG *max_exclusive, size_t len,
const uint8_t additional_data[32]);

int bn_rand_secret_range(BIGNUM *r, int *out_is_uniform, BN_ULONG min_inclusive,
const BIGNUM *max_exclusive);

int bn_mul_mont(BN_ULONG *rp, const BN_ULONG *ap, const BN_ULONG *bp,
const BN_ULONG *np, const BN_ULONG *n0, int num);

uint64_t bn_mont_n0(const BIGNUM *n);

int bn_mod_exp_base_2_consttime(BIGNUM *r, unsigned p, const BIGNUM *n,
BN_CTX *ctx);

#if defined(OPENSSL_X86_64) && defined(_MSC_VER)
#define BN_UMULT_LOHI(low, high, a, b) ((low) = _umul128((a), (b), &(high)))
#endif

#if !defined(BN_ULLONG) && !defined(BN_UMULT_LOHI)
#error "Either BN_ULLONG or BN_UMULT_LOHI must be defined on every platform."
#endif

int bn_jacobi(const BIGNUM *a, const BIGNUM *b, BN_CTX *ctx);

int bn_is_bit_set_words(const BN_ULONG *a, size_t num, unsigned bit);

int bn_one_to_montgomery(BIGNUM *r, const BN_MONT_CTX *mont, BN_CTX *ctx);

int bn_less_than_montgomery_R(const BIGNUM *bn, const BN_MONT_CTX *mont);

OPENSSL_EXPORT uint16_t bn_mod_u16_consttime(const BIGNUM *bn, uint16_t d);

int bn_odd_number_is_obviously_composite(const BIGNUM *bn);

void bn_rshift1_words(BN_ULONG *r, const BN_ULONG *a, size_t num);

void bn_rshift_words(BN_ULONG *r, const BN_ULONG *a, unsigned shift,
size_t num);

OPENSSL_EXPORT int bn_rshift_secret_shift(BIGNUM *r, const BIGNUM *a,
unsigned n, BN_CTX *ctx);

BN_ULONG bn_reduce_once(BN_ULONG *r, const BN_ULONG *a, BN_ULONG carry,
const BN_ULONG *m, size_t num);

BN_ULONG bn_reduce_once_in_place(BN_ULONG *r, BN_ULONG carry, const BN_ULONG *m,
BN_ULONG *tmp, size_t num);



int bn_uadd_consttime(BIGNUM *r, const BIGNUM *a, const BIGNUM *b);

int bn_usub_consttime(BIGNUM *r, const BIGNUM *a, const BIGNUM *b);

OPENSSL_EXPORT int bn_abs_sub_consttime(BIGNUM *r, const BIGNUM *a,
const BIGNUM *b, BN_CTX *ctx);

int bn_mul_consttime(BIGNUM *r, const BIGNUM *a, const BIGNUM *b, BN_CTX *ctx);

int bn_sqr_consttime(BIGNUM *r, const BIGNUM *a, BN_CTX *ctx);

OPENSSL_EXPORT int bn_div_consttime(BIGNUM *quotient, BIGNUM *remainder,
const BIGNUM *numerator,
const BIGNUM *divisor, BN_CTX *ctx);

OPENSSL_EXPORT int bn_is_relatively_prime(int *out_relatively_prime,
const BIGNUM *x, const BIGNUM *y,
BN_CTX *ctx);

OPENSSL_EXPORT int bn_lcm_consttime(BIGNUM *r, const BIGNUM *a, const BIGNUM *b,
BN_CTX *ctx);



void bn_mod_add_words(BN_ULONG *r, const BN_ULONG *a, const BN_ULONG *b,
const BN_ULONG *m, BN_ULONG *tmp, size_t num);

int bn_mod_add_consttime(BIGNUM *r, const BIGNUM *a, const BIGNUM *b,
const BIGNUM *m, BN_CTX *ctx);

void bn_mod_sub_words(BN_ULONG *r, const BN_ULONG *a, const BN_ULONG *b,
const BN_ULONG *m, BN_ULONG *tmp, size_t num);

int bn_mod_sub_consttime(BIGNUM *r, const BIGNUM *a, const BIGNUM *b,
const BIGNUM *m, BN_CTX *ctx);

int bn_mod_lshift1_consttime(BIGNUM *r, const BIGNUM *a, const BIGNUM *m,
BN_CTX *ctx);

int bn_mod_lshift_consttime(BIGNUM *r, const BIGNUM *a, int n, const BIGNUM *m,
BN_CTX *ctx);

OPENSSL_EXPORT int bn_mod_inverse_consttime(BIGNUM *r, int *out_no_inverse,
const BIGNUM *a, const BIGNUM *n,
BN_CTX *ctx);

int bn_mod_inverse_prime(BIGNUM *out, const BIGNUM *a, const BIGNUM *p,
BN_CTX *ctx, const BN_MONT_CTX *mont_p);

int bn_mod_inverse_secret_prime(BIGNUM *out, const BIGNUM *a, const BIGNUM *p,
BN_CTX *ctx, const BN_MONT_CTX *mont_p);



#if defined(OPENSSL_32_BIT)
#define BN_SMALL_MAX_WORDS 17
#else
#define BN_SMALL_MAX_WORDS 9
#endif

void bn_mul_small(BN_ULONG *r, size_t num_r, const BN_ULONG *a, size_t num_a,
const BN_ULONG *b, size_t num_b);

void bn_sqr_small(BN_ULONG *r, size_t num_r, const BN_ULONG *a, size_t num_a);


void bn_to_montgomery_small(BN_ULONG *r, const BN_ULONG *a, size_t num,
const BN_MONT_CTX *mont);

void bn_from_montgomery_small(BN_ULONG *r, const BN_ULONG *a, size_t num,
const BN_MONT_CTX *mont);

void bn_mod_mul_montgomery_small(BN_ULONG *r, const BN_ULONG *a,
const BN_ULONG *b, size_t num,
const BN_MONT_CTX *mont);

void bn_mod_exp_mont_small(BN_ULONG *r, const BN_ULONG *a, size_t num,
const BN_ULONG *p, size_t num_p,
const BN_MONT_CTX *mont);

void bn_mod_inverse_prime_mont_small(BN_ULONG *r, const BN_ULONG *a, size_t num,
const BN_MONT_CTX *mont);


#if defined(__cplusplus)
}  
#endif

#endif  


#ifndef LIBDIVIDE_H
#define LIBDIVIDE_H

#define LIBDIVIDE_VERSION "2.0"
#define LIBDIVIDE_VERSION_MAJOR 2
#define LIBDIVIDE_VERSION_MINOR 0

#include <stdint.h>

#if defined(__cplusplus)
#include <cstdlib>
#include <cstdio>
#else
#include <stdlib.h>
#include <stdio.h>
#endif

#if defined(LIBDIVIDE_AVX512)
#include <immintrin.h>
#elif defined(LIBDIVIDE_AVX2)
#include <immintrin.h>
#elif defined(LIBDIVIDE_SSE2)
#include <emmintrin.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#pragma warning(disable: 4146)
#define LIBDIVIDE_VC
#endif

#if !defined(__has_builtin)
#define __has_builtin(x) 0
#endif

#if defined(__SIZEOF_INT128__)
#define HAS_INT128_T
#if !(defined(__clang__) && defined(LIBDIVIDE_VC))
#define HAS_INT128_DIV
#endif
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define LIBDIVIDE_X86_64
#endif

#if defined(__i386__)
#define LIBDIVIDE_i386
#endif

#if defined(__GNUC__) || defined(__clang__)
#define LIBDIVIDE_GCC_STYLE_ASM
#endif

#if defined(__cplusplus) || defined(LIBDIVIDE_VC)
#define LIBDIVIDE_FUNCTION __FUNCTION__
#else
#define LIBDIVIDE_FUNCTION __func__
#endif

#define LIBDIVIDE_ERROR(msg) \
do { \
fprintf(stderr, "libdivide.h:%d: %s(): Error: %s\n", \
__LINE__, LIBDIVIDE_FUNCTION, msg); \
exit(-1); \
} while (0)

#if defined(LIBDIVIDE_ASSERTIONS_ON)
#define LIBDIVIDE_ASSERT(x) \
do { \
if (!(x)) { \
fprintf(stderr, "libdivide.h:%d: %s(): Assertion failed: %s\n", \
__LINE__, LIBDIVIDE_FUNCTION, #x); \
exit(-1); \
} \
} while (0)
#else
#define LIBDIVIDE_ASSERT(x)
#endif

#ifdef __cplusplus
namespace libdivide {
#endif

#pragma pack(push, 1)

struct libdivide_u32_t {
uint32_t magic;
uint8_t more;
};

struct libdivide_s32_t {
int32_t magic;
uint8_t more;
};

struct libdivide_u64_t {
uint64_t magic;
uint8_t more;
};

struct libdivide_s64_t {
int64_t magic;
uint8_t more;
};

struct libdivide_u32_branchfree_t {
uint32_t magic;
uint8_t more;
};

struct libdivide_s32_branchfree_t {
int32_t magic;
uint8_t more;
};

struct libdivide_u64_branchfree_t {
uint64_t magic;
uint8_t more;
};

struct libdivide_s64_branchfree_t {
int64_t magic;
uint8_t more;
};

#pragma pack(pop)


enum {
LIBDIVIDE_32_SHIFT_MASK = 0x1F,
LIBDIVIDE_64_SHIFT_MASK = 0x3F,
LIBDIVIDE_ADD_MARKER = 0x40,
LIBDIVIDE_NEGATIVE_DIVISOR = 0x80
};

static inline struct libdivide_s32_t libdivide_s32_gen(int32_t d);
static inline struct libdivide_u32_t libdivide_u32_gen(uint32_t d);
static inline struct libdivide_s64_t libdivide_s64_gen(int64_t d);
static inline struct libdivide_u64_t libdivide_u64_gen(uint64_t d);

static inline struct libdivide_s32_branchfree_t libdivide_s32_branchfree_gen(int32_t d);
static inline struct libdivide_u32_branchfree_t libdivide_u32_branchfree_gen(uint32_t d);
static inline struct libdivide_s64_branchfree_t libdivide_s64_branchfree_gen(int64_t d);
static inline struct libdivide_u64_branchfree_t libdivide_u64_branchfree_gen(uint64_t d);

static inline int32_t  libdivide_s32_do(int32_t numer, const struct libdivide_s32_t *denom);
static inline uint32_t libdivide_u32_do(uint32_t numer, const struct libdivide_u32_t *denom);
static inline int64_t  libdivide_s64_do(int64_t numer, const struct libdivide_s64_t *denom);
static inline uint64_t libdivide_u64_do(uint64_t numer, const struct libdivide_u64_t *denom);

static inline int32_t  libdivide_s32_branchfree_do(int32_t numer, const struct libdivide_s32_branchfree_t *denom);
static inline uint32_t libdivide_u32_branchfree_do(uint32_t numer, const struct libdivide_u32_branchfree_t *denom);
static inline int64_t  libdivide_s64_branchfree_do(int64_t numer, const struct libdivide_s64_branchfree_t *denom);
static inline uint64_t libdivide_u64_branchfree_do(uint64_t numer, const struct libdivide_u64_branchfree_t *denom);

static inline int32_t  libdivide_s32_recover(const struct libdivide_s32_t *denom);
static inline uint32_t libdivide_u32_recover(const struct libdivide_u32_t *denom);
static inline int64_t  libdivide_s64_recover(const struct libdivide_s64_t *denom);
static inline uint64_t libdivide_u64_recover(const struct libdivide_u64_t *denom);

static inline int32_t  libdivide_s32_branchfree_recover(const struct libdivide_s32_branchfree_t *denom);
static inline uint32_t libdivide_u32_branchfree_recover(const struct libdivide_u32_branchfree_t *denom);
static inline int64_t  libdivide_s64_branchfree_recover(const struct libdivide_s64_branchfree_t *denom);
static inline uint64_t libdivide_u64_branchfree_recover(const struct libdivide_u64_branchfree_t *denom);


static inline uint32_t libdivide_mullhi_u32(uint32_t x, uint32_t y) {
uint64_t xl = x, yl = y;
uint64_t rl = xl * yl;
return (uint32_t)(rl >> 32);
}

static inline int32_t libdivide_mullhi_s32(int32_t x, int32_t y) {
int64_t xl = x, yl = y;
int64_t rl = xl * yl;
return (int32_t)(rl >> 32);
}

static inline uint64_t libdivide_mullhi_u64(uint64_t x, uint64_t y) {
#if defined(LIBDIVIDE_VC) && \
defined(LIBDIVIDE_X86_64)
return __umulh(x, y);
#elif defined(HAS_INT128_T)
__uint128_t xl = x, yl = y;
__uint128_t rl = xl * yl;
return (uint64_t)(rl >> 64);
#else
uint32_t mask = 0xFFFFFFFF;
uint32_t x0 = (uint32_t)(x & mask);
uint32_t x1 = (uint32_t)(x >> 32);
uint32_t y0 = (uint32_t)(y & mask);
uint32_t y1 = (uint32_t)(y >> 32);
uint32_t x0y0_hi = libdivide_mullhi_u32(x0, y0);
uint64_t x0y1 = x0 * (uint64_t)y1;
uint64_t x1y0 = x1 * (uint64_t)y0;
uint64_t x1y1 = x1 * (uint64_t)y1;
uint64_t temp = x1y0 + x0y0_hi;
uint64_t temp_lo = temp & mask;
uint64_t temp_hi = temp >> 32;

return x1y1 + temp_hi + ((temp_lo + x0y1) >> 32);
#endif
}

static inline int64_t libdivide_mullhi_s64(int64_t x, int64_t y) {
#if defined(LIBDIVIDE_VC) && \
defined(LIBDIVIDE_X86_64)
return __mulh(x, y);
#elif defined(HAS_INT128_T)
__int128_t xl = x, yl = y;
__int128_t rl = xl * yl;
return (int64_t)(rl >> 64);
#else
uint32_t mask = 0xFFFFFFFF;
uint32_t x0 = (uint32_t)(x & mask);
uint32_t y0 = (uint32_t)(y & mask);
int32_t x1 = (int32_t)(x >> 32);
int32_t y1 = (int32_t)(y >> 32);
uint32_t x0y0_hi = libdivide_mullhi_u32(x0, y0);
int64_t t = x1 * (int64_t)y0 + x0y0_hi;
int64_t w1 = x0 * (int64_t)y1 + (t & mask);

return x1 * (int64_t)y1 + (t >> 32) + (w1 >> 32);
#endif
}

static inline int32_t libdivide_count_leading_zeros32(uint32_t val) {
#if defined(__GNUC__) || \
__has_builtin(__builtin_clz)
return __builtin_clz(val);
#elif defined(LIBDIVIDE_VC)
unsigned long result;
if (_BitScanReverse(&result, val)) {
return 31 - result;
}
return 0;
#else
int32_t result = 0;
uint32_t hi = 1U << 31;
for (; ~val & hi; hi >>= 1) {
result++;
}
return result;
#endif
}

static inline int32_t libdivide_count_leading_zeros64(uint64_t val) {
#if defined(__GNUC__) || \
__has_builtin(__builtin_clzll)
return __builtin_clzll(val);
#elif defined(LIBDIVIDE_VC) && defined(_WIN64)
unsigned long result;
if (_BitScanReverse64(&result, val)) {
return 63 - result;
}
return 0;
#else
uint32_t hi = val >> 32;
uint32_t lo = val & 0xFFFFFFFF;
if (hi != 0) return libdivide_count_leading_zeros32(hi);
return 32 + libdivide_count_leading_zeros32(lo);
#endif
}

static inline uint32_t libdivide_64_div_32_to_32(uint32_t u1, uint32_t u0, uint32_t v, uint32_t *r) {
#if (defined(LIBDIVIDE_i386) || defined(LIBDIVIDE_X86_64)) && \
defined(LIBDIVIDE_GCC_STYLE_ASM)
uint32_t result;
__asm__("divl %[v]"
: "=a"(result), "=d"(*r)
: [v] "r"(v), "a"(u0), "d"(u1)
);
return result;
#else
uint64_t n = ((uint64_t)u1 << 32) | u0;
uint32_t result = (uint32_t)(n / v);
*r = (uint32_t)(n - result * (uint64_t)v);
return result;
#endif
}

static uint64_t libdivide_128_div_64_to_64(uint64_t u1, uint64_t u0, uint64_t v, uint64_t *r) {
#if defined(LIBDIVIDE_X86_64) && \
defined(LIBDIVIDE_GCC_STYLE_ASM)
uint64_t result;
__asm__("divq %[v]"
: "=a"(result), "=d"(*r)
: [v] "r"(v), "a"(u0), "d"(u1)
);
return result;
#elif defined(HAS_INT128_T) && \
defined(HAS_INT128_DIV)
__uint128_t n = ((__uint128_t)u1 << 64) | u0;
uint64_t result = (uint64_t)(n / v);
*r = (uint64_t)(n - result * (__uint128_t)v);
return result;
#else

const uint64_t b = (1ULL << 32); 
uint64_t un1, un0; 
uint64_t vn1, vn0; 
uint64_t q1, q0; 
uint64_t un64, un21, un10; 
uint64_t rhat; 
int32_t s; 

if (u1 >= v) {
*r = (uint64_t) -1;
return (uint64_t) -1;
}

s = libdivide_count_leading_zeros64(v);
if (s > 0) {
v = v << s;
un64 = (u1 << s) | (u0 >> (64 - s));
un10 = u0 << s; 
} else {
un64 = u1;
un10 = u0;
}

vn1 = v >> 32;
vn0 = v & 0xFFFFFFFF;

un1 = un10 >> 32;
un0 = un10 & 0xFFFFFFFF;

q1 = un64 / vn1;
rhat = un64 - q1 * vn1;

while (q1 >= b || q1 * vn0 > b * rhat + un1) {
q1 = q1 - 1;
rhat = rhat + vn1;
if (rhat >= b)
break;
}

un21 = un64 * b + un1 - q1 * v;

q0 = un21 / vn1;
rhat = un21 - q0 * vn1;

while (q0 >= b || q0 * vn0 > b * rhat + un0) {
q0 = q0 - 1;
rhat = rhat + vn1;
if (rhat >= b)
break;
}

*r = (un21 * b + un0 - q0 * v) >> s;
return q1 * b + q0;
#endif
}

static inline void libdivide_u128_shift(uint64_t *u1, uint64_t *u0, int32_t signed_shift) {
if (signed_shift > 0) {
uint32_t shift = signed_shift;
*u1 <<= shift;
*u1 |= *u0 >> (64 - shift);
*u0 <<= shift;
}
else if (signed_shift < 0) {
uint32_t shift = -signed_shift;
*u0 >>= shift;
*u0 |= *u1 << (64 - shift);
*u1 >>= shift;
}
}

static uint64_t libdivide_128_div_128_to_64(uint64_t u_hi, uint64_t u_lo, uint64_t v_hi, uint64_t v_lo, uint64_t *r_hi, uint64_t *r_lo) {
#if defined(HAS_INT128_T) && \
defined(HAS_INT128_DIV)
__uint128_t ufull = u_hi;
__uint128_t vfull = v_hi;
ufull = (ufull << 64) | u_lo;
vfull = (vfull << 64) | v_lo;
uint64_t res = (uint64_t)(ufull / vfull);
__uint128_t remainder = ufull - (vfull * res);
*r_lo = (uint64_t)remainder;
*r_hi = (uint64_t)(remainder >> 64);
return res;
#else
typedef struct { uint64_t hi; uint64_t lo; } u128_t;
u128_t u = {u_hi, u_lo};
u128_t v = {v_hi, v_lo};

if (v.hi == 0) {
*r_hi = 0;
return libdivide_128_div_64_to_64(u.hi, u.lo, v.lo, r_lo);
}
uint32_t n = libdivide_count_leading_zeros64(v.hi);

u128_t v1t = v;
libdivide_u128_shift(&v1t.hi, &v1t.lo, n);
uint64_t v1 = v1t.hi; 

u128_t u1 = u;
libdivide_u128_shift(&u1.hi, &u1.lo, -1);

uint64_t rem_ignored;
uint64_t q1 = libdivide_128_div_64_to_64(u1.hi, u1.lo, v1, &rem_ignored);

u128_t q0 = {0, q1};
libdivide_u128_shift(&q0.hi, &q0.lo, n);
libdivide_u128_shift(&q0.hi, &q0.lo, -63);

if (q0.hi != 0 || q0.lo != 0) {
q0.hi -= (q0.lo == 0); 
q0.lo -= 1;
}

u128_t q0v = {0, 0};
q0v.hi = q0.hi*v.lo + q0.lo*v.hi + libdivide_mullhi_u64(q0.lo, v.lo);
q0v.lo = q0.lo*v.lo;

u128_t u_q0v = u;
u_q0v.hi -= q0v.hi + (u.lo < q0v.lo); 
u_q0v.lo -= q0v.lo;

if ((u_q0v.hi > v.hi) ||
(u_q0v.hi == v.hi && u_q0v.lo >= v.lo)) {
q0.lo += 1;
q0.hi += (q0.lo == 0); 

u_q0v.hi -= v.hi + (u_q0v.lo < v.lo);
u_q0v.lo -= v.lo;
}

*r_hi = u_q0v.hi;
*r_lo = u_q0v.lo;

LIBDIVIDE_ASSERT(q0.hi == 0);
return q0.lo;
#endif
}


static inline struct libdivide_u32_t libdivide_internal_u32_gen(uint32_t d, int branchfree) {
if (d == 0) {
LIBDIVIDE_ERROR("divider must be != 0");
}

struct libdivide_u32_t result;
uint32_t floor_log_2_d = 31 - libdivide_count_leading_zeros32(d);

if ((d & (d - 1)) == 0) {
result.magic = 0;
result.more = (uint8_t)(floor_log_2_d - (branchfree != 0));
} else {
uint8_t more;
uint32_t rem, proposed_m;
proposed_m = libdivide_64_div_32_to_32(1U << floor_log_2_d, 0, d, &rem);

LIBDIVIDE_ASSERT(rem > 0 && rem < d);
const uint32_t e = d - rem;

if (!branchfree && (e < (1U << floor_log_2_d))) {
more = floor_log_2_d;
} else {
proposed_m += proposed_m;
const uint32_t twice_rem = rem + rem;
if (twice_rem >= d || twice_rem < rem) proposed_m += 1;
more = floor_log_2_d | LIBDIVIDE_ADD_MARKER;
}
result.magic = 1 + proposed_m;
result.more = more;
}
return result;
}

struct libdivide_u32_t libdivide_u32_gen(uint32_t d) {
return libdivide_internal_u32_gen(d, 0);
}

struct libdivide_u32_branchfree_t libdivide_u32_branchfree_gen(uint32_t d) {
if (d == 1) {
LIBDIVIDE_ERROR("branchfree divider must be != 1");
}
struct libdivide_u32_t tmp = libdivide_internal_u32_gen(d, 1);
struct libdivide_u32_branchfree_t ret = {tmp.magic, (uint8_t)(tmp.more & LIBDIVIDE_32_SHIFT_MASK)};
return ret;
}

uint32_t libdivide_u32_do(uint32_t numer, const struct libdivide_u32_t *denom) {
uint8_t more = denom->more;
if (!denom->magic) {
return numer >> more;
}
else {
uint32_t q = libdivide_mullhi_u32(denom->magic, numer);
if (more & LIBDIVIDE_ADD_MARKER) {
uint32_t t = ((numer - q) >> 1) + q;
return t >> (more & LIBDIVIDE_32_SHIFT_MASK);
}
else {
return q >> more;
}
}
}

uint32_t libdivide_u32_branchfree_do(uint32_t numer, const struct libdivide_u32_branchfree_t *denom) {
uint32_t q = libdivide_mullhi_u32(denom->magic, numer);
uint32_t t = ((numer - q) >> 1) + q;
return t >> denom->more;
}

uint32_t libdivide_u32_recover(const struct libdivide_u32_t *denom) {
uint8_t more = denom->more;
uint8_t shift = more & LIBDIVIDE_32_SHIFT_MASK;

if (!denom->magic) {
return 1U << shift;
} else if (!(more & LIBDIVIDE_ADD_MARKER)) {
uint32_t hi_dividend = 1U << shift;
uint32_t rem_ignored;
return 1 + libdivide_64_div_32_to_32(hi_dividend, 0, denom->magic, &rem_ignored);
} else {
uint64_t half_n = 1ULL << (32 + shift);
uint64_t d = (1ULL << 32) | denom->magic;
uint32_t half_q = (uint32_t)(half_n / d);
uint64_t rem = half_n % d;
uint32_t full_q = half_q + half_q + ((rem<<1) >= d);

return full_q + 1;
}
}

uint32_t libdivide_u32_branchfree_recover(const struct libdivide_u32_branchfree_t *denom) {
uint8_t more = denom->more;
uint8_t shift = more & LIBDIVIDE_32_SHIFT_MASK;

if (!denom->magic) {
return 1U << (shift + 1);
} else {
uint64_t half_n = 1ULL << (32 + shift);
uint64_t d = (1ULL << 32) | denom->magic;
uint32_t half_q = (uint32_t)(half_n / d);
uint64_t rem = half_n % d;
uint32_t full_q = half_q + half_q + ((rem<<1) >= d);

return full_q + 1;
}
}


static inline struct libdivide_u64_t libdivide_internal_u64_gen(uint64_t d, int branchfree) {
if (d == 0) {
LIBDIVIDE_ERROR("divider must be != 0");
}

struct libdivide_u64_t result;
uint32_t floor_log_2_d = 63 - libdivide_count_leading_zeros64(d);

if ((d & (d - 1)) == 0) {
result.magic = 0;
result.more = (uint8_t)(floor_log_2_d - (branchfree != 0));
} else {
uint64_t proposed_m, rem;
uint8_t more;
proposed_m = libdivide_128_div_64_to_64(1ULL << floor_log_2_d, 0, d, &rem);

LIBDIVIDE_ASSERT(rem > 0 && rem < d);
const uint64_t e = d - rem;

if (!branchfree && e < (1ULL << floor_log_2_d)) {
more = floor_log_2_d;
} else {
proposed_m += proposed_m;
const uint64_t twice_rem = rem + rem;
if (twice_rem >= d || twice_rem < rem) proposed_m += 1;
more = floor_log_2_d | LIBDIVIDE_ADD_MARKER;
}
result.magic = 1 + proposed_m;
result.more = more;
}
return result;
}

struct libdivide_u64_t libdivide_u64_gen(uint64_t d) {
return libdivide_internal_u64_gen(d, 0);
}

struct libdivide_u64_branchfree_t libdivide_u64_branchfree_gen(uint64_t d) {
if (d == 1) {
LIBDIVIDE_ERROR("branchfree divider must be != 1");
}
struct libdivide_u64_t tmp = libdivide_internal_u64_gen(d, 1);
struct libdivide_u64_branchfree_t ret = {tmp.magic, (uint8_t)(tmp.more & LIBDIVIDE_64_SHIFT_MASK)};
return ret;
}

uint64_t libdivide_u64_do(uint64_t numer, const struct libdivide_u64_t *denom) {
uint8_t more = denom->more;
if (!denom->magic) {
return numer >> more;
}
else {
uint64_t q = libdivide_mullhi_u64(denom->magic, numer);
if (more & LIBDIVIDE_ADD_MARKER) {
uint64_t t = ((numer - q) >> 1) + q;
return t >> (more & LIBDIVIDE_64_SHIFT_MASK);
}
else {
return q >> more;
}
}
}

uint64_t libdivide_u64_branchfree_do(uint64_t numer, const struct libdivide_u64_branchfree_t *denom) {
uint64_t q = libdivide_mullhi_u64(denom->magic, numer);
uint64_t t = ((numer - q) >> 1) + q;
return t >> denom->more;
}

uint64_t libdivide_u64_recover(const struct libdivide_u64_t *denom) {
uint8_t more = denom->more;
uint8_t shift = more & LIBDIVIDE_64_SHIFT_MASK;

if (!denom->magic) {
return 1ULL << shift;
} else if (!(more & LIBDIVIDE_ADD_MARKER)) {
uint64_t hi_dividend = 1ULL << shift;
uint64_t rem_ignored;
return 1 + libdivide_128_div_64_to_64(hi_dividend, 0, denom->magic, &rem_ignored);
} else {

uint64_t half_n_hi = 1ULL << shift, half_n_lo = 0;
const uint64_t d_hi = 1, d_lo = denom->magic;
uint64_t r_hi, r_lo;
uint64_t half_q = libdivide_128_div_128_to_64(half_n_hi, half_n_lo, d_hi, d_lo, &r_hi, &r_lo);
uint64_t dr_lo = r_lo + r_lo;
uint64_t dr_hi = r_hi + r_hi + (dr_lo < r_lo); 
int dr_exceeds_d = (dr_hi > d_hi) || (dr_hi == d_hi && dr_lo >= d_lo);
uint64_t full_q = half_q + half_q + (dr_exceeds_d ? 1 : 0);
return full_q + 1;
}
}

uint64_t libdivide_u64_branchfree_recover(const struct libdivide_u64_branchfree_t *denom) {
uint8_t more = denom->more;
uint8_t shift = more & LIBDIVIDE_64_SHIFT_MASK;

if (!denom->magic) {
return 1ULL << (shift + 1);
} else {

uint64_t half_n_hi = 1ULL << shift, half_n_lo = 0;
const uint64_t d_hi = 1, d_lo = denom->magic;
uint64_t r_hi, r_lo;
uint64_t half_q = libdivide_128_div_128_to_64(half_n_hi, half_n_lo, d_hi, d_lo, &r_hi, &r_lo);
uint64_t dr_lo = r_lo + r_lo;
uint64_t dr_hi = r_hi + r_hi + (dr_lo < r_lo); 
int dr_exceeds_d = (dr_hi > d_hi) || (dr_hi == d_hi && dr_lo >= d_lo);
uint64_t full_q = half_q + half_q + (dr_exceeds_d ? 1 : 0);
return full_q + 1;
}
}


static inline struct libdivide_s32_t libdivide_internal_s32_gen(int32_t d, int branchfree) {
if (d == 0) {
LIBDIVIDE_ERROR("divider must be != 0");
}

struct libdivide_s32_t result;

uint32_t ud = (uint32_t)d;
uint32_t absD = (d < 0) ? -ud : ud;
uint32_t floor_log_2_d = 31 - libdivide_count_leading_zeros32(absD);
if ((absD & (absD - 1)) == 0) {
result.magic = 0;
result.more = floor_log_2_d | (d < 0 ? LIBDIVIDE_NEGATIVE_DIVISOR : 0);
} else {
LIBDIVIDE_ASSERT(floor_log_2_d >= 1);

uint8_t more;
uint32_t rem, proposed_m;
proposed_m = libdivide_64_div_32_to_32(1U << (floor_log_2_d - 1), 0, absD, &rem);
const uint32_t e = absD - rem;

if (!branchfree && e < (1U << floor_log_2_d)) {
more = floor_log_2_d - 1;
} else {
proposed_m += proposed_m;
const uint32_t twice_rem = rem + rem;
if (twice_rem >= absD || twice_rem < rem) proposed_m += 1;
more = floor_log_2_d | LIBDIVIDE_ADD_MARKER;
}

proposed_m += 1;
int32_t magic = (int32_t)proposed_m;

if (d < 0) {
more |= LIBDIVIDE_NEGATIVE_DIVISOR;
if (!branchfree) {
magic = -magic;
}
}

result.more = more;
result.magic = magic;
}
return result;
}

struct libdivide_s32_t libdivide_s32_gen(int32_t d) {
return libdivide_internal_s32_gen(d, 0);
}

struct libdivide_s32_branchfree_t libdivide_s32_branchfree_gen(int32_t d) {
struct libdivide_s32_t tmp = libdivide_internal_s32_gen(d, 1);
struct libdivide_s32_branchfree_t result = {tmp.magic, tmp.more};
return result;
}

int32_t libdivide_s32_do(int32_t numer, const struct libdivide_s32_t *denom) {
uint8_t more = denom->more;
uint8_t shift = more & LIBDIVIDE_32_SHIFT_MASK;

if (!denom->magic) {
uint32_t sign = (int8_t)more >> 7;
uint32_t mask = (1U << shift) - 1;
uint32_t uq = numer + ((numer >> 31) & mask);
int32_t q = (int32_t)uq;
q >>= shift;
q = (q ^ sign) - sign;
return q;
} else {
uint32_t uq = (uint32_t)libdivide_mullhi_s32(denom->magic, numer);
if (more & LIBDIVIDE_ADD_MARKER) {
int32_t sign = (int8_t)more >> 7;
uq += ((uint32_t)numer ^ sign) - sign;
}
int32_t q = (int32_t)uq;
q >>= shift;
q += (q < 0);
return q;
}
}

int32_t libdivide_s32_branchfree_do(int32_t numer, const struct libdivide_s32_branchfree_t *denom) {
uint8_t more = denom->more;
uint8_t shift = more & LIBDIVIDE_32_SHIFT_MASK;
int32_t sign = (int8_t)more >> 7;
int32_t magic = denom->magic;
int32_t q = libdivide_mullhi_s32(magic, numer);
q += numer;

uint32_t is_power_of_2 = (magic == 0);
uint32_t q_sign = (uint32_t)(q >> 31);
q += q_sign & ((1U << shift) - is_power_of_2);

q >>= shift;
q = (q ^ sign) - sign;

return q;
}

int32_t libdivide_s32_recover(const struct libdivide_s32_t *denom) {
uint8_t more = denom->more;
uint8_t shift = more & LIBDIVIDE_32_SHIFT_MASK;
if (!denom->magic) {
uint32_t absD = 1U << shift;
if (more & LIBDIVIDE_NEGATIVE_DIVISOR) {
absD = -absD;
}
return (int32_t)absD;
} else {
int negative_divisor = (more & LIBDIVIDE_NEGATIVE_DIVISOR);
int magic_was_negated = (more & LIBDIVIDE_ADD_MARKER)
? denom->magic > 0 : denom->magic < 0;

if (denom->magic == 0) {
int32_t result = 1U << shift;
return negative_divisor ? -result : result;
}

uint32_t d = (uint32_t)(magic_was_negated ? -denom->magic : denom->magic);
uint64_t n = 1ULL << (32 + shift); 
uint32_t q = (uint32_t)(n / d);
int32_t result = (int32_t)q;
result += 1;
return negative_divisor ? -result : result;
}
}

int32_t libdivide_s32_branchfree_recover(const struct libdivide_s32_branchfree_t *denom) {
return libdivide_s32_recover((const struct libdivide_s32_t *)denom);
}


static inline struct libdivide_s64_t libdivide_internal_s64_gen(int64_t d, int branchfree) {
if (d == 0) {
LIBDIVIDE_ERROR("divider must be != 0");
}

struct libdivide_s64_t result;

uint64_t ud = (uint64_t)d;
uint64_t absD = (d < 0) ? -ud : ud;
uint32_t floor_log_2_d = 63 - libdivide_count_leading_zeros64(absD);
if ((absD & (absD - 1)) == 0) {
result.magic = 0;
result.more = floor_log_2_d | (d < 0 ? LIBDIVIDE_NEGATIVE_DIVISOR : 0);
} else {
uint8_t more;
uint64_t rem, proposed_m;
proposed_m = libdivide_128_div_64_to_64(1ULL << (floor_log_2_d - 1), 0, absD, &rem);
const uint64_t e = absD - rem;

if (!branchfree && e < (1ULL << floor_log_2_d)) {
more = floor_log_2_d - 1;
} else {
proposed_m += proposed_m;
const uint64_t twice_rem = rem + rem;
if (twice_rem >= absD || twice_rem < rem) proposed_m += 1;
more = floor_log_2_d | LIBDIVIDE_ADD_MARKER;
}
proposed_m += 1;
int64_t magic = (int64_t)proposed_m;

if (d < 0) {
more |= LIBDIVIDE_NEGATIVE_DIVISOR;
if (!branchfree) {
magic = -magic;
}
}

result.more = more;
result.magic = magic;
}
return result;
}

struct libdivide_s64_t libdivide_s64_gen(int64_t d) {
return libdivide_internal_s64_gen(d, 0);
}

struct libdivide_s64_branchfree_t libdivide_s64_branchfree_gen(int64_t d) {
struct libdivide_s64_t tmp = libdivide_internal_s64_gen(d, 1);
struct libdivide_s64_branchfree_t ret = {tmp.magic, tmp.more};
return ret;
}

int64_t libdivide_s64_do(int64_t numer, const struct libdivide_s64_t *denom) {
uint8_t more = denom->more;
uint8_t shift = more & LIBDIVIDE_64_SHIFT_MASK;

if (!denom->magic) { 
uint64_t mask = (1ULL << shift) - 1;
uint64_t uq = numer + ((numer >> 63) & mask);
int64_t q = (int64_t)uq;
q >>= shift;
int64_t sign = (int8_t)more >> 7;
q = (q ^ sign) - sign;
return q;
} else {
uint64_t uq = (uint64_t)libdivide_mullhi_s64(denom->magic, numer);
if (more & LIBDIVIDE_ADD_MARKER) {
int64_t sign = (int8_t)more >> 7;
uq += ((uint64_t)numer ^ sign) - sign;
}
int64_t q = (int64_t)uq;
q >>= shift;
q += (q < 0);
return q;
}
}

int64_t libdivide_s64_branchfree_do(int64_t numer, const struct libdivide_s64_branchfree_t *denom) {
uint8_t more = denom->more;
uint8_t shift = more & LIBDIVIDE_64_SHIFT_MASK;
int64_t sign = (int8_t)more >> 7;
int64_t magic = denom->magic;
int64_t q = libdivide_mullhi_s64(magic, numer);
q += numer;

uint64_t is_power_of_2 = (magic == 0);
uint64_t q_sign = (uint64_t)(q >> 63);
q += q_sign & ((1ULL << shift) - is_power_of_2);

q >>= shift;
q = (q ^ sign) - sign;

return q;
}

int64_t libdivide_s64_recover(const struct libdivide_s64_t *denom) {
uint8_t more = denom->more;
uint8_t shift = more & LIBDIVIDE_64_SHIFT_MASK;
if (denom->magic == 0) { 
uint64_t absD = 1ULL << shift;
if (more & LIBDIVIDE_NEGATIVE_DIVISOR) {
absD = -absD;
}
return (int64_t)absD;
} else {
int negative_divisor = (more & LIBDIVIDE_NEGATIVE_DIVISOR);
int magic_was_negated = (more & LIBDIVIDE_ADD_MARKER)
? denom->magic > 0 : denom->magic < 0;

uint64_t d = (uint64_t)(magic_was_negated ? -denom->magic : denom->magic);
uint64_t n_hi = 1ULL << shift, n_lo = 0;
uint64_t rem_ignored;
uint64_t q = libdivide_128_div_64_to_64(n_hi, n_lo, d, &rem_ignored);
int64_t result = (int64_t)(q + 1);
if (negative_divisor) {
result = -result;
}
return result;
}
}

int64_t libdivide_s64_branchfree_recover(const struct libdivide_s64_branchfree_t *denom) {
return libdivide_s64_recover((const struct libdivide_s64_t *)denom);
}

#if defined(LIBDIVIDE_AVX512)

static inline __m512i libdivide_u32_do_vector(__m512i numers, const struct libdivide_u32_t *denom);
static inline __m512i libdivide_s32_do_vector(__m512i numers, const struct libdivide_s32_t *denom);
static inline __m512i libdivide_u64_do_vector(__m512i numers, const struct libdivide_u64_t *denom);
static inline __m512i libdivide_s64_do_vector(__m512i numers, const struct libdivide_s64_t *denom);

static inline __m512i libdivide_u32_branchfree_do_vector(__m512i numers, const struct libdivide_u32_branchfree_t *denom);
static inline __m512i libdivide_s32_branchfree_do_vector(__m512i numers, const struct libdivide_s32_branchfree_t *denom);
static inline __m512i libdivide_u64_branchfree_do_vector(__m512i numers, const struct libdivide_u64_branchfree_t *denom);
static inline __m512i libdivide_s64_branchfree_do_vector(__m512i numers, const struct libdivide_s64_branchfree_t *denom);


static inline __m512i libdivide_s64_signbits(__m512i v) {;
return _mm512_srai_epi64(v, 63);
}

static inline __m512i libdivide_s64_shift_right_vector(__m512i v, int amt) {
return _mm512_srai_epi64(v, amt);
}

static inline __m512i libdivide_mullhi_u32_vector(__m512i a, __m512i b) {
__m512i hi_product_0Z2Z = _mm512_srli_epi64(_mm512_mul_epu32(a, b), 32);
__m512i a1X3X = _mm512_srli_epi64(a, 32);
__m512i mask = _mm512_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0);
__m512i hi_product_Z1Z3 = _mm512_and_si512(_mm512_mul_epu32(a1X3X, b), mask);
return _mm512_or_si512(hi_product_0Z2Z, hi_product_Z1Z3);
}

static inline __m512i libdivide_mullhi_s32_vector(__m512i a, __m512i b) {
__m512i hi_product_0Z2Z = _mm512_srli_epi64(_mm512_mul_epi32(a, b), 32);
__m512i a1X3X = _mm512_srli_epi64(a, 32);
__m512i mask = _mm512_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0);
__m512i hi_product_Z1Z3 = _mm512_and_si512(_mm512_mul_epi32(a1X3X, b), mask);
return _mm512_or_si512(hi_product_0Z2Z, hi_product_Z1Z3);
}

static inline __m512i libdivide_mullhi_u64_vector(__m512i x, __m512i y) {
__m512i lomask = _mm512_set1_epi64(0xffffffff);
__m512i xh = _mm512_shuffle_epi32(x, (_MM_PERM_ENUM) 0xB1);
__m512i yh = _mm512_shuffle_epi32(y, (_MM_PERM_ENUM) 0xB1);
__m512i w0 = _mm512_mul_epu32(x, y);
__m512i w1 = _mm512_mul_epu32(x, yh);
__m512i w2 = _mm512_mul_epu32(xh, y);
__m512i w3 = _mm512_mul_epu32(xh, yh);
__m512i w0h = _mm512_srli_epi64(w0, 32);
__m512i s1 = _mm512_add_epi64(w1, w0h);
__m512i s1l = _mm512_and_si512(s1, lomask);
__m512i s1h = _mm512_srli_epi64(s1, 32);
__m512i s2 = _mm512_add_epi64(w2, s1l);
__m512i s2h = _mm512_srli_epi64(s2, 32);
__m512i hi = _mm512_add_epi64(w3, s1h);
hi = _mm512_add_epi64(hi, s2h);

return hi;
}

static inline __m512i libdivide_mullhi_s64_vector(__m512i x, __m512i y) {
__m512i p = libdivide_mullhi_u64_vector(x, y);
__m512i t1 = _mm512_and_si512(libdivide_s64_signbits(x), y);
__m512i t2 = _mm512_and_si512(libdivide_s64_signbits(y), x);
p = _mm512_sub_epi64(p, t1);
p = _mm512_sub_epi64(p, t2);
return p;
}


__m512i libdivide_u32_do_vector(__m512i numers, const struct libdivide_u32_t *denom) {
uint8_t more = denom->more;
if (!denom->magic) {
return _mm512_srli_epi32(numers, more);
}
else {
__m512i q = libdivide_mullhi_u32_vector(numers, _mm512_set1_epi32(denom->magic));
if (more & LIBDIVIDE_ADD_MARKER) {
uint32_t shift = more & LIBDIVIDE_32_SHIFT_MASK;
__m512i t = _mm512_add_epi32(_mm512_srli_epi32(_mm512_sub_epi32(numers, q), 1), q);
return _mm512_srli_epi32(t, shift);
}
else {
return _mm512_srli_epi32(q, more);
}
}
}

__m512i libdivide_u32_branchfree_do_vector(__m512i numers, const struct libdivide_u32_branchfree_t *denom) {
__m512i q = libdivide_mullhi_u32_vector(numers, _mm512_set1_epi32(denom->magic));
__m512i t = _mm512_add_epi32(_mm512_srli_epi32(_mm512_sub_epi32(numers, q), 1), q);
return _mm512_srli_epi32(t, denom->more);
}


__m512i libdivide_u64_do_vector(__m512i numers, const struct libdivide_u64_t *denom) {
uint8_t more = denom->more;
if (!denom->magic) {
return _mm512_srli_epi64(numers, more);
}
else {
__m512i q = libdivide_mullhi_u64_vector(numers, _mm512_set1_epi64(denom->magic));
if (more & LIBDIVIDE_ADD_MARKER) {
uint32_t shift = more & LIBDIVIDE_64_SHIFT_MASK;
__m512i t = _mm512_add_epi64(_mm512_srli_epi64(_mm512_sub_epi64(numers, q), 1), q);
return _mm512_srli_epi64(t, shift);
}
else {
return _mm512_srli_epi64(q, more);
}
}
}

__m512i libdivide_u64_branchfree_do_vector(__m512i numers, const struct libdivide_u64_branchfree_t *denom) {
__m512i q = libdivide_mullhi_u64_vector(numers, _mm512_set1_epi64(denom->magic));
__m512i t = _mm512_add_epi64(_mm512_srli_epi64(_mm512_sub_epi64(numers, q), 1), q);
return _mm512_srli_epi64(t, denom->more);
}


__m512i libdivide_s32_do_vector(__m512i numers, const struct libdivide_s32_t *denom) {
uint8_t more = denom->more;
if (!denom->magic) {
uint32_t shift = more & LIBDIVIDE_32_SHIFT_MASK;
uint32_t mask = (1U << shift) - 1;
__m512i roundToZeroTweak = _mm512_set1_epi32(mask);
__m512i q = _mm512_add_epi32(numers, _mm512_and_si512(_mm512_srai_epi32(numers, 31), roundToZeroTweak));
q = _mm512_srai_epi32(q, shift);
__m512i sign = _mm512_set1_epi32((int8_t)more >> 7);
q = _mm512_sub_epi32(_mm512_xor_si512(q, sign), sign);
return q;
}
else {
__m512i q = libdivide_mullhi_s32_vector(numers, _mm512_set1_epi32(denom->magic));
if (more & LIBDIVIDE_ADD_MARKER) {
__m512i sign = _mm512_set1_epi32((int8_t)more >> 7);
q = _mm512_add_epi32(q, _mm512_sub_epi32(_mm512_xor_si512(numers, sign), sign));
}
q = _mm512_srai_epi32(q, more & LIBDIVIDE_32_SHIFT_MASK);
q = _mm512_add_epi32(q, _mm512_srli_epi32(q, 31)); 
return q;
}
}

__m512i libdivide_s32_branchfree_do_vector(__m512i numers, const struct libdivide_s32_branchfree_t *denom) {
int32_t magic = denom->magic;
uint8_t more = denom->more;
uint8_t shift = more & LIBDIVIDE_32_SHIFT_MASK;
__m512i sign = _mm512_set1_epi32((int8_t)more >> 7);
__m512i q = libdivide_mullhi_s32_vector(numers, _mm512_set1_epi32(magic));
q = _mm512_add_epi32(q, numers); 

uint32_t is_power_of_2 = (magic == 0);
__m512i q_sign = _mm512_srai_epi32(q, 31); 
__m512i mask = _mm512_set1_epi32((1U << shift) - is_power_of_2);
q = _mm512_add_epi32(q, _mm512_and_si512(q_sign, mask)); 
q = _mm512_srai_epi32(q, shift); 
q = _mm512_sub_epi32(_mm512_xor_si512(q, sign), sign); 
return q;
}


__m512i libdivide_s64_do_vector(__m512i numers, const struct libdivide_s64_t *denom) {
uint8_t more = denom->more;
int64_t magic = denom->magic;
if (magic == 0) { 
uint32_t shift = more & LIBDIVIDE_64_SHIFT_MASK;
uint64_t mask = (1ULL << shift) - 1;
__m512i roundToZeroTweak = _mm512_set1_epi64(mask);
__m512i q = _mm512_add_epi64(numers, _mm512_and_si512(libdivide_s64_signbits(numers), roundToZeroTweak));
q = libdivide_s64_shift_right_vector(q, shift);
__m512i sign = _mm512_set1_epi32((int8_t)more >> 7);
q = _mm512_sub_epi64(_mm512_xor_si512(q, sign), sign);
return q;
}
else {
__m512i q = libdivide_mullhi_s64_vector(numers, _mm512_set1_epi64(magic));
if (more & LIBDIVIDE_ADD_MARKER) {
__m512i sign = _mm512_set1_epi32((int8_t)more >> 7);
q = _mm512_add_epi64(q, _mm512_sub_epi64(_mm512_xor_si512(numers, sign), sign));
}
q = libdivide_s64_shift_right_vector(q, more & LIBDIVIDE_64_SHIFT_MASK);
q = _mm512_add_epi64(q, _mm512_srli_epi64(q, 63)); 
return q;
}
}

__m512i libdivide_s64_branchfree_do_vector(__m512i numers, const struct libdivide_s64_branchfree_t *denom) {
int64_t magic = denom->magic;
uint8_t more = denom->more;
uint8_t shift = more & LIBDIVIDE_64_SHIFT_MASK;
__m512i sign = _mm512_set1_epi32((int8_t)more >> 7);

__m512i q = libdivide_mullhi_s64_vector(numers, _mm512_set1_epi64(magic));
q = _mm512_add_epi64(q, numers); 

uint32_t is_power_of_2 = (magic == 0);
__m512i q_sign = libdivide_s64_signbits(q); 
__m512i mask = _mm512_set1_epi64((1ULL << shift) - is_power_of_2);
q = _mm512_add_epi64(q, _mm512_and_si512(q_sign, mask)); 
q = libdivide_s64_shift_right_vector(q, shift); 
q = _mm512_sub_epi64(_mm512_xor_si512(q, sign), sign); 
return q;
}

#elif defined(LIBDIVIDE_AVX2)

static inline __m256i libdivide_u32_do_vector(__m256i numers, const struct libdivide_u32_t *denom);
static inline __m256i libdivide_s32_do_vector(__m256i numers, const struct libdivide_s32_t *denom);
static inline __m256i libdivide_u64_do_vector(__m256i numers, const struct libdivide_u64_t *denom);
static inline __m256i libdivide_s64_do_vector(__m256i numers, const struct libdivide_s64_t *denom);

static inline __m256i libdivide_u32_branchfree_do_vector(__m256i numers, const struct libdivide_u32_branchfree_t *denom);
static inline __m256i libdivide_s32_branchfree_do_vector(__m256i numers, const struct libdivide_s32_branchfree_t *denom);
static inline __m256i libdivide_u64_branchfree_do_vector(__m256i numers, const struct libdivide_u64_branchfree_t *denom);
static inline __m256i libdivide_s64_branchfree_do_vector(__m256i numers, const struct libdivide_s64_branchfree_t *denom);


static inline __m256i libdivide_s64_signbits(__m256i v) {
__m256i hiBitsDuped = _mm256_shuffle_epi32(v, _MM_SHUFFLE(3, 3, 1, 1));
__m256i signBits = _mm256_srai_epi32(hiBitsDuped, 31);
return signBits;
}

static inline __m256i libdivide_s64_shift_right_vector(__m256i v, int amt) {
const int b = 64 - amt;
__m256i m = _mm256_set1_epi64x(1ULL << (b - 1));
__m256i x = _mm256_srli_epi64(v, amt);
__m256i result = _mm256_sub_epi64(_mm256_xor_si256(x, m), m);
return result;
}

static inline __m256i libdivide_mullhi_u32_vector(__m256i a, __m256i b) {
__m256i hi_product_0Z2Z = _mm256_srli_epi64(_mm256_mul_epu32(a, b), 32);
__m256i a1X3X = _mm256_srli_epi64(a, 32);
__m256i mask = _mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0);
__m256i hi_product_Z1Z3 = _mm256_and_si256(_mm256_mul_epu32(a1X3X, b), mask);
return _mm256_or_si256(hi_product_0Z2Z, hi_product_Z1Z3);
}

static inline __m256i libdivide_mullhi_s32_vector(__m256i a, __m256i b) {
__m256i hi_product_0Z2Z = _mm256_srli_epi64(_mm256_mul_epi32(a, b), 32);
__m256i a1X3X = _mm256_srli_epi64(a, 32);
__m256i mask = _mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0);
__m256i hi_product_Z1Z3 = _mm256_and_si256(_mm256_mul_epi32(a1X3X, b), mask);
return _mm256_or_si256(hi_product_0Z2Z, hi_product_Z1Z3);
}

static inline __m256i libdivide_mullhi_u64_vector(__m256i x, __m256i y) {
__m256i lomask = _mm256_set1_epi64x(0xffffffff);
__m256i xh = _mm256_shuffle_epi32(x, 0xB1);        
__m256i yh = _mm256_shuffle_epi32(y, 0xB1);        
__m256i w0 = _mm256_mul_epu32(x, y);               
__m256i w1 = _mm256_mul_epu32(x, yh);              
__m256i w2 = _mm256_mul_epu32(xh, y);              
__m256i w3 = _mm256_mul_epu32(xh, yh);             
__m256i w0h = _mm256_srli_epi64(w0, 32);
__m256i s1 = _mm256_add_epi64(w1, w0h);
__m256i s1l = _mm256_and_si256(s1, lomask);
__m256i s1h = _mm256_srli_epi64(s1, 32);
__m256i s2 = _mm256_add_epi64(w2, s1l);
__m256i s2h = _mm256_srli_epi64(s2, 32);
__m256i hi = _mm256_add_epi64(w3, s1h);
hi = _mm256_add_epi64(hi, s2h);

return hi;
}

static inline __m256i libdivide_mullhi_s64_vector(__m256i x, __m256i y) {
__m256i p = libdivide_mullhi_u64_vector(x, y);
__m256i t1 = _mm256_and_si256(libdivide_s64_signbits(x), y);
__m256i t2 = _mm256_and_si256(libdivide_s64_signbits(y), x);
p = _mm256_sub_epi64(p, t1);
p = _mm256_sub_epi64(p, t2);
return p;
}


__m256i libdivide_u32_do_vector(__m256i numers, const struct libdivide_u32_t *denom) {
uint8_t more = denom->more;
if (!denom->magic) {
return _mm256_srli_epi32(numers, more);
}
else {
__m256i q = libdivide_mullhi_u32_vector(numers, _mm256_set1_epi32(denom->magic));
if (more & LIBDIVIDE_ADD_MARKER) {
uint32_t shift = more & LIBDIVIDE_32_SHIFT_MASK;
__m256i t = _mm256_add_epi32(_mm256_srli_epi32(_mm256_sub_epi32(numers, q), 1), q);
return _mm256_srli_epi32(t, shift);
}
else {
return _mm256_srli_epi32(q, more);
}
}
}

__m256i libdivide_u32_branchfree_do_vector(__m256i numers, const struct libdivide_u32_branchfree_t *denom) {
__m256i q = libdivide_mullhi_u32_vector(numers, _mm256_set1_epi32(denom->magic));
__m256i t = _mm256_add_epi32(_mm256_srli_epi32(_mm256_sub_epi32(numers, q), 1), q);
return _mm256_srli_epi32(t, denom->more);
}


__m256i libdivide_u64_do_vector(__m256i numers, const struct libdivide_u64_t *denom) {
uint8_t more = denom->more;
if (!denom->magic) {
return _mm256_srli_epi64(numers, more);
}
else {
__m256i q = libdivide_mullhi_u64_vector(numers, _mm256_set1_epi64x(denom->magic));
if (more & LIBDIVIDE_ADD_MARKER) {
uint32_t shift = more & LIBDIVIDE_64_SHIFT_MASK;
__m256i t = _mm256_add_epi64(_mm256_srli_epi64(_mm256_sub_epi64(numers, q), 1), q);
return _mm256_srli_epi64(t, shift);
}
else {
return _mm256_srli_epi64(q, more);
}
}
}

__m256i libdivide_u64_branchfree_do_vector(__m256i numers, const struct libdivide_u64_branchfree_t *denom) {
__m256i q = libdivide_mullhi_u64_vector(numers, _mm256_set1_epi64x(denom->magic));
__m256i t = _mm256_add_epi64(_mm256_srli_epi64(_mm256_sub_epi64(numers, q), 1), q);
return _mm256_srli_epi64(t, denom->more);
}


__m256i libdivide_s32_do_vector(__m256i numers, const struct libdivide_s32_t *denom) {
uint8_t more = denom->more;
if (!denom->magic) {
uint32_t shift = more & LIBDIVIDE_32_SHIFT_MASK;
uint32_t mask = (1U << shift) - 1;
__m256i roundToZeroTweak = _mm256_set1_epi32(mask);
__m256i q = _mm256_add_epi32(numers, _mm256_and_si256(_mm256_srai_epi32(numers, 31), roundToZeroTweak));
q = _mm256_srai_epi32(q, shift);
__m256i sign = _mm256_set1_epi32((int8_t)more >> 7);
q = _mm256_sub_epi32(_mm256_xor_si256(q, sign), sign);
return q;
}
else {
__m256i q = libdivide_mullhi_s32_vector(numers, _mm256_set1_epi32(denom->magic));
if (more & LIBDIVIDE_ADD_MARKER) {
__m256i sign = _mm256_set1_epi32((int8_t)more >> 7);
q = _mm256_add_epi32(q, _mm256_sub_epi32(_mm256_xor_si256(numers, sign), sign));
}
q = _mm256_srai_epi32(q, more & LIBDIVIDE_32_SHIFT_MASK);
q = _mm256_add_epi32(q, _mm256_srli_epi32(q, 31)); 
return q;
}
}

__m256i libdivide_s32_branchfree_do_vector(__m256i numers, const struct libdivide_s32_branchfree_t *denom) {
int32_t magic = denom->magic;
uint8_t more = denom->more;
uint8_t shift = more & LIBDIVIDE_32_SHIFT_MASK;
__m256i sign = _mm256_set1_epi32((int8_t)more >> 7);
__m256i q = libdivide_mullhi_s32_vector(numers, _mm256_set1_epi32(magic));
q = _mm256_add_epi32(q, numers); 

uint32_t is_power_of_2 = (magic == 0);
__m256i q_sign = _mm256_srai_epi32(q, 31); 
__m256i mask = _mm256_set1_epi32((1U << shift) - is_power_of_2);
q = _mm256_add_epi32(q, _mm256_and_si256(q_sign, mask)); 
q = _mm256_srai_epi32(q, shift); 
q = _mm256_sub_epi32(_mm256_xor_si256(q, sign), sign); 
return q;
}


__m256i libdivide_s64_do_vector(__m256i numers, const struct libdivide_s64_t *denom) {
uint8_t more = denom->more;
int64_t magic = denom->magic;
if (magic == 0) { 
uint32_t shift = more & LIBDIVIDE_64_SHIFT_MASK;
uint64_t mask = (1ULL << shift) - 1;
__m256i roundToZeroTweak = _mm256_set1_epi64x(mask);
__m256i q = _mm256_add_epi64(numers, _mm256_and_si256(libdivide_s64_signbits(numers), roundToZeroTweak));
q = libdivide_s64_shift_right_vector(q, shift);
__m256i sign = _mm256_set1_epi32((int8_t)more >> 7);
q = _mm256_sub_epi64(_mm256_xor_si256(q, sign), sign);
return q;
}
else {
__m256i q = libdivide_mullhi_s64_vector(numers, _mm256_set1_epi64x(magic));
if (more & LIBDIVIDE_ADD_MARKER) {
__m256i sign = _mm256_set1_epi32((int8_t)more >> 7);
q = _mm256_add_epi64(q, _mm256_sub_epi64(_mm256_xor_si256(numers, sign), sign));
}
q = libdivide_s64_shift_right_vector(q, more & LIBDIVIDE_64_SHIFT_MASK);
q = _mm256_add_epi64(q, _mm256_srli_epi64(q, 63)); 
return q;
}
}

__m256i libdivide_s64_branchfree_do_vector(__m256i numers, const struct libdivide_s64_branchfree_t *denom) {
int64_t magic = denom->magic;
uint8_t more = denom->more;
uint8_t shift = more & LIBDIVIDE_64_SHIFT_MASK;
__m256i sign = _mm256_set1_epi32((int8_t)more >> 7);

__m256i q = libdivide_mullhi_s64_vector(numers, _mm256_set1_epi64x(magic));
q = _mm256_add_epi64(q, numers); 

uint32_t is_power_of_2 = (magic == 0);
__m256i q_sign = libdivide_s64_signbits(q); 
__m256i mask = _mm256_set1_epi64x((1ULL << shift) - is_power_of_2);
q = _mm256_add_epi64(q, _mm256_and_si256(q_sign, mask)); 
q = libdivide_s64_shift_right_vector(q, shift); 
q = _mm256_sub_epi64(_mm256_xor_si256(q, sign), sign); 
return q;
}

#elif defined(LIBDIVIDE_SSE2)

static inline __m128i libdivide_u32_do_vector(__m128i numers, const struct libdivide_u32_t *denom);
static inline __m128i libdivide_s32_do_vector(__m128i numers, const struct libdivide_s32_t *denom);
static inline __m128i libdivide_u64_do_vector(__m128i numers, const struct libdivide_u64_t *denom);
static inline __m128i libdivide_s64_do_vector(__m128i numers, const struct libdivide_s64_t *denom);

static inline __m128i libdivide_u32_branchfree_do_vector(__m128i numers, const struct libdivide_u32_branchfree_t *denom);
static inline __m128i libdivide_s32_branchfree_do_vector(__m128i numers, const struct libdivide_s32_branchfree_t *denom);
static inline __m128i libdivide_u64_branchfree_do_vector(__m128i numers, const struct libdivide_u64_branchfree_t *denom);
static inline __m128i libdivide_s64_branchfree_do_vector(__m128i numers, const struct libdivide_s64_branchfree_t *denom);


static inline __m128i libdivide_s64_signbits(__m128i v) {
__m128i hiBitsDuped = _mm_shuffle_epi32(v, _MM_SHUFFLE(3, 3, 1, 1));
__m128i signBits = _mm_srai_epi32(hiBitsDuped, 31);
return signBits;
}

static inline __m128i libdivide_s64_shift_right_vector(__m128i v, int amt) {
const int b = 64 - amt;
__m128i m = _mm_set1_epi64x(1ULL << (b - 1));
__m128i x = _mm_srli_epi64(v, amt);
__m128i result = _mm_sub_epi64(_mm_xor_si128(x, m), m);
return result;
}

static inline __m128i libdivide_mullhi_u32_vector(__m128i a, __m128i b) {
__m128i hi_product_0Z2Z = _mm_srli_epi64(_mm_mul_epu32(a, b), 32);
__m128i a1X3X = _mm_srli_epi64(a, 32);
__m128i mask = _mm_set_epi32(-1, 0, -1, 0);
__m128i hi_product_Z1Z3 = _mm_and_si128(_mm_mul_epu32(a1X3X, b), mask);
return _mm_or_si128(hi_product_0Z2Z, hi_product_Z1Z3);
}

static inline __m128i libdivide_mullhi_s32_vector(__m128i a, __m128i b) {
__m128i p = libdivide_mullhi_u32_vector(a, b);
__m128i t1 = _mm_and_si128(_mm_srai_epi32(a, 31), b);
__m128i t2 = _mm_and_si128(_mm_srai_epi32(b, 31), a);
p = _mm_sub_epi32(p, t1);
p = _mm_sub_epi32(p, t2);
return p;
}

static inline __m128i libdivide_mullhi_u64_vector(__m128i x, __m128i y) {
__m128i lomask = _mm_set1_epi64x(0xffffffff);
__m128i xh = _mm_shuffle_epi32(x, 0xB1);        
__m128i yh = _mm_shuffle_epi32(y, 0xB1);        
__m128i w0 = _mm_mul_epu32(x, y);               
__m128i w1 = _mm_mul_epu32(x, yh);              
__m128i w2 = _mm_mul_epu32(xh, y);              
__m128i w3 = _mm_mul_epu32(xh, yh);             
__m128i w0h = _mm_srli_epi64(w0, 32);
__m128i s1 = _mm_add_epi64(w1, w0h);
__m128i s1l = _mm_and_si128(s1, lomask);
__m128i s1h = _mm_srli_epi64(s1, 32);
__m128i s2 = _mm_add_epi64(w2, s1l);
__m128i s2h = _mm_srli_epi64(s2, 32);
__m128i hi = _mm_add_epi64(w3, s1h);
hi = _mm_add_epi64(hi, s2h);

return hi;
}

static inline __m128i libdivide_mullhi_s64_vector(__m128i x, __m128i y) {
__m128i p = libdivide_mullhi_u64_vector(x, y);
__m128i t1 = _mm_and_si128(libdivide_s64_signbits(x), y);
__m128i t2 = _mm_and_si128(libdivide_s64_signbits(y), x);
p = _mm_sub_epi64(p, t1);
p = _mm_sub_epi64(p, t2);
return p;
}


__m128i libdivide_u32_do_vector(__m128i numers, const struct libdivide_u32_t *denom) {
uint8_t more = denom->more;
if (!denom->magic) {
return _mm_srli_epi32(numers, more);
}
else {
__m128i q = libdivide_mullhi_u32_vector(numers, _mm_set1_epi32(denom->magic));
if (more & LIBDIVIDE_ADD_MARKER) {
uint32_t shift = more & LIBDIVIDE_32_SHIFT_MASK;
__m128i t = _mm_add_epi32(_mm_srli_epi32(_mm_sub_epi32(numers, q), 1), q);
return _mm_srli_epi32(t, shift);
}
else {
return _mm_srli_epi32(q, more);
}
}
}

__m128i libdivide_u32_branchfree_do_vector(__m128i numers, const struct libdivide_u32_branchfree_t *denom) {
__m128i q = libdivide_mullhi_u32_vector(numers, _mm_set1_epi32(denom->magic));
__m128i t = _mm_add_epi32(_mm_srli_epi32(_mm_sub_epi32(numers, q), 1), q);
return _mm_srli_epi32(t, denom->more);
}


__m128i libdivide_u64_do_vector(__m128i numers, const struct libdivide_u64_t *denom) {
uint8_t more = denom->more;
if (!denom->magic) {
return _mm_srli_epi64(numers, more);
}
else {
__m128i q = libdivide_mullhi_u64_vector(numers, _mm_set1_epi64x(denom->magic));
if (more & LIBDIVIDE_ADD_MARKER) {
uint32_t shift = more & LIBDIVIDE_64_SHIFT_MASK;
__m128i t = _mm_add_epi64(_mm_srli_epi64(_mm_sub_epi64(numers, q), 1), q);
return _mm_srli_epi64(t, shift);
}
else {
return _mm_srli_epi64(q, more);
}
}
}

__m128i libdivide_u64_branchfree_do_vector(__m128i numers, const struct libdivide_u64_branchfree_t *denom) {
__m128i q = libdivide_mullhi_u64_vector(numers, _mm_set1_epi64x(denom->magic));
__m128i t = _mm_add_epi64(_mm_srli_epi64(_mm_sub_epi64(numers, q), 1), q);
return _mm_srli_epi64(t, denom->more);
}


__m128i libdivide_s32_do_vector(__m128i numers, const struct libdivide_s32_t *denom) {
uint8_t more = denom->more;
if (!denom->magic) {
uint32_t shift = more & LIBDIVIDE_32_SHIFT_MASK;
uint32_t mask = (1U << shift) - 1;
__m128i roundToZeroTweak = _mm_set1_epi32(mask);
__m128i q = _mm_add_epi32(numers, _mm_and_si128(_mm_srai_epi32(numers, 31), roundToZeroTweak));
q = _mm_srai_epi32(q, shift);
__m128i sign = _mm_set1_epi32((int8_t)more >> 7);
q = _mm_sub_epi32(_mm_xor_si128(q, sign), sign);
return q;
}
else {
__m128i q = libdivide_mullhi_s32_vector(numers, _mm_set1_epi32(denom->magic));
if (more & LIBDIVIDE_ADD_MARKER) {
__m128i sign = _mm_set1_epi32((int8_t)more >> 7);
q = _mm_add_epi32(q, _mm_sub_epi32(_mm_xor_si128(numers, sign), sign));
}
q = _mm_srai_epi32(q, more & LIBDIVIDE_32_SHIFT_MASK);
q = _mm_add_epi32(q, _mm_srli_epi32(q, 31)); 
return q;
}
}

__m128i libdivide_s32_branchfree_do_vector(__m128i numers, const struct libdivide_s32_branchfree_t *denom) {
int32_t magic = denom->magic;
uint8_t more = denom->more;
uint8_t shift = more & LIBDIVIDE_32_SHIFT_MASK;
__m128i sign = _mm_set1_epi32((int8_t)more >> 7);
__m128i q = libdivide_mullhi_s32_vector(numers, _mm_set1_epi32(magic));
q = _mm_add_epi32(q, numers); 

uint32_t is_power_of_2 = (magic == 0);
__m128i q_sign = _mm_srai_epi32(q, 31); 
__m128i mask = _mm_set1_epi32((1U << shift) - is_power_of_2);
q = _mm_add_epi32(q, _mm_and_si128(q_sign, mask)); 
q = _mm_srai_epi32(q, shift); 
q = _mm_sub_epi32(_mm_xor_si128(q, sign), sign); 
return q;
}


__m128i libdivide_s64_do_vector(__m128i numers, const struct libdivide_s64_t *denom) {
uint8_t more = denom->more;
int64_t magic = denom->magic;
if (magic == 0) { 
uint32_t shift = more & LIBDIVIDE_64_SHIFT_MASK;
uint64_t mask = (1ULL << shift) - 1;
__m128i roundToZeroTweak = _mm_set1_epi64x(mask);
__m128i q = _mm_add_epi64(numers, _mm_and_si128(libdivide_s64_signbits(numers), roundToZeroTweak));
q = libdivide_s64_shift_right_vector(q, shift);
__m128i sign = _mm_set1_epi32((int8_t)more >> 7);
q = _mm_sub_epi64(_mm_xor_si128(q, sign), sign);
return q;
}
else {
__m128i q = libdivide_mullhi_s64_vector(numers, _mm_set1_epi64x(magic));
if (more & LIBDIVIDE_ADD_MARKER) {
__m128i sign = _mm_set1_epi32((int8_t)more >> 7);
q = _mm_add_epi64(q, _mm_sub_epi64(_mm_xor_si128(numers, sign), sign));
}
q = libdivide_s64_shift_right_vector(q, more & LIBDIVIDE_64_SHIFT_MASK);
q = _mm_add_epi64(q, _mm_srli_epi64(q, 63)); 
return q;
}
}

__m128i libdivide_s64_branchfree_do_vector(__m128i numers, const struct libdivide_s64_branchfree_t *denom) {
int64_t magic = denom->magic;
uint8_t more = denom->more;
uint8_t shift = more & LIBDIVIDE_64_SHIFT_MASK;
__m128i sign = _mm_set1_epi32((int8_t)more >> 7);

__m128i q = libdivide_mullhi_s64_vector(numers, _mm_set1_epi64x(magic));
q = _mm_add_epi64(q, numers); 

uint32_t is_power_of_2 = (magic == 0);
__m128i q_sign = libdivide_s64_signbits(q); 
__m128i mask = _mm_set1_epi64x((1ULL << shift) - is_power_of_2);
q = _mm_add_epi64(q, _mm_and_si128(q_sign, mask)); 
q = libdivide_s64_shift_right_vector(q, shift); 
q = _mm_sub_epi64(_mm_xor_si128(q, sign), sign); 
return q;
}

#endif


#ifdef __cplusplus

enum {
BRANCHFULL,
BRANCHFREE
};

#if defined(LIBDIVIDE_AVX512)
#define LIBDIVIDE_VECTOR_TYPE __m512i
#elif defined(LIBDIVIDE_AVX2)
#define LIBDIVIDE_VECTOR_TYPE __m256i
#elif defined(LIBDIVIDE_SSE2)
#define LIBDIVIDE_VECTOR_TYPE __m128i
#endif

#if !defined(LIBDIVIDE_VECTOR_TYPE)
#define LIBDIVIDE_DIVIDE_VECTOR(ALGO)
#else
#define LIBDIVIDE_DIVIDE_VECTOR(ALGO) \
LIBDIVIDE_VECTOR_TYPE divide(LIBDIVIDE_VECTOR_TYPE n) const { \
return libdivide_##ALGO##_do_vector(n, &denom); \
}
#endif

#define DISPATCHER_GEN(T, ALGO) \
libdivide_##ALGO##_t denom; \
dispatcher() { } \
dispatcher(T d) \
: denom(libdivide_##ALGO##_gen(d)) \
{ } \
T divide(T n) const { \
return libdivide_##ALGO##_do(n, &denom); \
} \
LIBDIVIDE_DIVIDE_VECTOR(ALGO) \
T recover() const { \
return libdivide_##ALGO##_recover(&denom); \
}

template<typename T, int ALGO> struct dispatcher { };

template<> struct dispatcher<int32_t, BRANCHFULL> { DISPATCHER_GEN(int32_t, s32) };
template<> struct dispatcher<int32_t, BRANCHFREE> { DISPATCHER_GEN(int32_t, s32_branchfree) };
template<> struct dispatcher<uint32_t, BRANCHFULL> { DISPATCHER_GEN(uint32_t, u32) };
template<> struct dispatcher<uint32_t, BRANCHFREE> { DISPATCHER_GEN(uint32_t, u32_branchfree) };
template<> struct dispatcher<int64_t, BRANCHFULL> { DISPATCHER_GEN(int64_t, s64) };
template<> struct dispatcher<int64_t, BRANCHFREE> { DISPATCHER_GEN(int64_t, s64_branchfree) };
template<> struct dispatcher<uint64_t, BRANCHFULL> { DISPATCHER_GEN(uint64_t, u64) };
template<> struct dispatcher<uint64_t, BRANCHFREE> { DISPATCHER_GEN(uint64_t, u64_branchfree) };

template<typename T, int ALGO = BRANCHFULL>
class divider {
public:
divider() { }

divider(T d) : div(d) { }

T divide(T n) const {
return div.divide(n);
}

T recover() const {
return div.recover();
}

bool operator==(const divider<T, ALGO>& other) const {
return div.denom.magic == other.denom.magic &&
div.denom.more == other.denom.more;
}

bool operator!=(const divider<T, ALGO>& other) const {
return !(*this == other);
}

#if defined(LIBDIVIDE_VECTOR_TYPE)
LIBDIVIDE_VECTOR_TYPE divide(LIBDIVIDE_VECTOR_TYPE n) const {
return div.divide(n);
}
#endif
private:
dispatcher<T, ALGO> div;
};

template<typename T, int ALGO>
T operator/(T n, const divider<T, ALGO>& div) {
return div.divide(n);
}

template<typename T, int ALGO>
T& operator/=(T& n, const divider<T, ALGO>& div) {
n = div.divide(n);
return n;
}

#if defined(LIBDIVIDE_VECTOR_TYPE)
template<typename T, int ALGO>
LIBDIVIDE_VECTOR_TYPE operator/(LIBDIVIDE_VECTOR_TYPE n, const divider<T, ALGO>& div) {
return div.divide(n);
}
template<typename T, int ALGO>
LIBDIVIDE_VECTOR_TYPE& operator/=(LIBDIVIDE_VECTOR_TYPE& n, const divider<T, ALGO>& div) {
n = div.divide(n);
return n;
}
#endif

#if __cplusplus >= 201103L || \
(defined(_MSC_VER) && _MSC_VER >= 1800)
template <typename T>
using branchfree_divider = divider<T, BRANCHFREE>;
#endif

} 

#endif 

#endif 

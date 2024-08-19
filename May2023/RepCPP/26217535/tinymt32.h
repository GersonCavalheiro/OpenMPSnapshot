
#ifndef TINYMT32_H
#define TINYMT32_H


#include <alpaka/core/BoostPredef.hpp>

#include <cstdint>

#ifndef UINT32_MAX
#   define UINT32_MAX ((uint32_t)-1u)
#endif
#ifndef UINT32_C
#   define UINT32_C(value) uint_least32_t(value)
#endif
#include <cinttypes>

#if BOOST_COMP_CLANG
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wold-style-cast"
#endif
#if BOOST_COMP_GNUC
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#pragma warning(push)
#pragma warning(disable: 4100)  
#endif

#define TINYMT32_MEXP 127
#define TINYMT32_SH0 1
#define TINYMT32_SH1 10
#define TINYMT32_SH8 8
#define TINYMT32_MASK UINT32_C(0x7fffffff)
#define TINYMT32_MUL (1.0f / 16777216.0f)

#if defined(__cplusplus)
extern "C" {
#endif


struct TINYMT32_T {
uint32_t status[4];
uint32_t mat1;
uint32_t mat2;
uint32_t tmat;
};

typedef struct TINYMT32_T tinymt32_t;

inline void tinymt32_init(tinymt32_t * random, uint32_t seed);
inline void tinymt32_init_by_array(tinymt32_t * random, uint32_t init_key[],
int key_length);

#if defined(__GNUC__)

inline static int tinymt32_get_mexp(
tinymt32_t * random  __attribute__((unused))) {
return TINYMT32_MEXP;
}
#else
inline static int tinymt32_get_mexp(tinymt32_t * random) {
return TINYMT32_MEXP;
}
#endif


inline static void tinymt32_next_state(tinymt32_t * random) {
uint32_t x;
uint32_t y;

y = random->status[3];
x = (random->status[0] & TINYMT32_MASK)
^ random->status[1]
^ random->status[2];
x ^= (x << TINYMT32_SH0);
y ^= (y >> TINYMT32_SH0) ^ x;
random->status[0] = random->status[1];
random->status[1] = random->status[2];
random->status[2] = x ^ (y << TINYMT32_SH1);
random->status[3] = y;
int32_t const a = -((int32_t)(y & 1)) & (int32_t)random->mat1;
int32_t const b = -((int32_t)(y & 1)) & (int32_t)random->mat2;
random->status[1] ^= (uint32_t)a;
random->status[2] ^= (uint32_t)b;
}


inline static uint32_t tinymt32_temper(tinymt32_t * random) {
uint32_t t0, t1;
t0 = random->status[3];
#if defined(LINEARITY_CHECK)
t1 = random->status[0]
^ (random->status[2] >> TINYMT32_SH8);
#else
t1 = random->status[0]
+ (random->status[2] >> TINYMT32_SH8);
#endif
t0 ^= t1;
if ((t1 & 1) != 0) {
t0 ^= random->tmat;
}
return t0;
}


inline static float tinymt32_temper_conv(tinymt32_t * random) {
uint32_t t0, t1;
union {
uint32_t u;
float f;
} conv;

t0 = random->status[3];
#if defined(LINEARITY_CHECK)
t1 = random->status[0]
^ (random->status[2] >> TINYMT32_SH8);
#else
t1 = random->status[0]
+ (random->status[2] >> TINYMT32_SH8);
#endif
t0 ^= t1;
if ((t1 & 1) != 0) {
conv.u  = ((t0 ^ random->tmat) >> 9) | UINT32_C(0x3f800000);
} else {
conv.u  = (t0 >> 9) | UINT32_C(0x3f800000);
}
return conv.f;
}


inline static float tinymt32_temper_conv_open(tinymt32_t * random) {
uint32_t t0, t1;
union {
uint32_t u;
float f;
} conv;

t0 = random->status[3];
#if defined(LINEARITY_CHECK)
t1 = random->status[0]
^ (random->status[2] >> TINYMT32_SH8);
#else
t1 = random->status[0]
+ (random->status[2] >> TINYMT32_SH8);
#endif
t0 ^= t1;
if ((t1 & 1) != 0) {
conv.u  = ((t0 ^ random->tmat) >> 9) | UINT32_C(0x3f800001);
} else {
conv.u  = (t0 >> 9) | UINT32_C(0x3f800001);
}
return conv.f;
}


inline static uint32_t tinymt32_generate_uint32(tinymt32_t * random) {
tinymt32_next_state(random);
return tinymt32_temper(random);
}


inline static float tinymt32_generate_float(tinymt32_t * random) {
tinymt32_next_state(random);
return (float)(tinymt32_temper(random) >> 8) * TINYMT32_MUL;
}


inline static float tinymt32_generate_float12(tinymt32_t * random) {
tinymt32_next_state(random);
return tinymt32_temper_conv(random);
}


inline static float tinymt32_generate_float01(tinymt32_t * random) {
tinymt32_next_state(random);
return tinymt32_temper_conv(random) - 1.0f;
}


inline static float tinymt32_generate_floatOC(tinymt32_t * random) {
tinymt32_next_state(random);
return 1.0f - tinymt32_generate_float(random);
}


inline static float tinymt32_generate_floatOO(tinymt32_t * random) {
tinymt32_next_state(random);
return tinymt32_temper_conv_open(random) - 1.0f;
}


inline static double tinymt32_generate_32double(tinymt32_t * random) {
tinymt32_next_state(random);
return tinymt32_temper(random) * (1.0 / 4294967296.0);
}

#if defined(__cplusplus)
}
#endif

#define MIN_LOOP 8
#define PRE_LOOP 8


static uint32_t ini_func1(uint32_t x) {
return (x ^ (x >> 27)) * UINT32_C(1664525);
}


static uint32_t ini_func2(uint32_t x) {
return (x ^ (x >> 27)) * UINT32_C(1566083941);
}


static void period_certification(tinymt32_t * random) {
if ((random->status[0] & TINYMT32_MASK) == 0 &&
random->status[1] == 0 &&
random->status[2] == 0 &&
random->status[3] == 0) {
random->status[0] = 'T';
random->status[1] = 'I';
random->status[2] = 'N';
random->status[3] = 'Y';
}
}


void tinymt32_init(tinymt32_t * random, uint32_t seed) {
random->status[0] = seed;
random->status[1] = random->mat1;
random->status[2] = random->mat2;
random->status[3] = random->tmat;
for (unsigned int i = 1; i < MIN_LOOP; i++) {
random->status[i & 3] ^= i + UINT32_C(1812433253)
* (random->status[(i - 1) & 3]
^ (random->status[(i - 1) & 3] >> 30));
}
period_certification(random);
for (unsigned int i = 0; i < PRE_LOOP; i++) {
tinymt32_next_state(random);
}
}


void tinymt32_init_by_array(tinymt32_t * random, uint32_t init_key[],
int key_length) {
const unsigned int lag = 1;
const unsigned int mid = 1;
const unsigned int size = 4;
unsigned int i, j;
unsigned int count;
uint32_t r;
uint32_t * st = &random->status[0];

st[0] = 0;
st[1] = random->mat1;
st[2] = random->mat2;
st[3] = random->tmat;
if (key_length + 1 > MIN_LOOP) {
count = (unsigned int)key_length + 1;
} else {
count = MIN_LOOP;
}
r = ini_func1(st[0] ^ st[mid % size]
^ st[(size - 1) % size]);
st[mid % size] += r;
r += (unsigned int)key_length;
st[(mid + lag) % size] += r;
st[0] = r;
count--;
for (i = 1, j = 0; (j < count) && (j < (unsigned int)key_length); j++) {
r = ini_func1(st[i % size]
^ st[(i + mid) % size]
^ st[(i + size - 1) % size]);
st[(i + mid) % size] += r;
r += init_key[j] + i;
st[(i + mid + lag) % size] += r;
st[i % size] = r;
i = (i + 1) % size;
}
for (; j < count; j++) {
r = ini_func1(st[i % size]
^ st[(i + mid) % size]
^ st[(i + size - 1) % size]);
st[(i + mid) % size] += r;
r += i;
st[(i + mid + lag) % size] += r;
st[i % size] = r;
i = (i + 1) % size;
}
for (j = 0; j < size; j++) {
r = ini_func2(st[i % size]
+ st[(i + mid) % size]
+ st[(i + size - 1) % size]);
st[(i + mid) % size] ^= r;
r -= i;
st[(i + mid + lag) % size] ^= r;
st[i % size] = r;
i = (i + 1) % size;
}
period_certification(random);
for (i = 0; i < PRE_LOOP; i++) {
tinymt32_next_state(random);
}
}

#undef MIN_LOOP
#undef PRE_LOOP

#if BOOST_COMP_CLANG
#   pragma clang diagnostic pop
#endif
#if BOOST_COMP_GNUC
#   pragma GCC diagnostic pop
#endif
#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#   pragma warning(pop)
#endif

#endif

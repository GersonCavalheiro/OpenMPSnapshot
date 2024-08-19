

#pragma once

#include <iostream>
#include "tools.h"

#define MIX2(x0, x1, rx, z0, z1, rz) \
x0 += x1; \
z0 += z1; \
x1 = (x1 << rx) | (x1 >> (64-rx)); \
z1 = (z1 << rz) | (z1 >> (64-rz)); \
x1 ^= x0; \
z1 ^= z0;


#define MIXK(x0, x1, rx, z0, z1, rz, k0, k1, l0, l1) \
x1 += k1; \
z1 += l1; \
x0 += x1+k0; \
z0 += z1+l0; \
x1 = (x1 << rx) | (x1 >> (64-rx)); \
z1 = (z1 << rz) | (z1 >> (64-rz)); \
x1 ^= x0; \
z1 ^= z0; \

namespace trinity {

template<bool C, typename T = void>
struct enable_if {
typedef T type;
};

template<typename T>
struct enable_if<false, T> {
};

template<typename T>
struct has_generate_template {
typedef char (& Two)[2];;
template<typename F, void (F::*)(int*, int*)>
struct helper {
};
template<typename C>
static char test(helper<C, &C::template generate<int*> >*);
template<typename C>
static Two test(...);
static bool const value = sizeof(test<T>(0)) == sizeof(char);
};


class random_engine {
public:
typedef uint32_t result_type;

#if __cplusplus <= 199711L
static result_type (min)() { return 0; }
static result_type (max)() { return 0xFFFFFFFF; }
#else

static constexpr result_type (min)() { return 0; }

static constexpr result_type (max)() { return 0xFFFFFFFF; }

#endif


random_engine() {
seed();
}

random_engine(const random_engine& x) {
for (unsigned short i = 0; i < 4; ++i) {
_s[i] = x._s[i];
_k[i] = x._k[i];
_o[i] = x._o[i];
}
_o_counter = x._o_counter;
}


explicit random_engine(uint32_t s) {
seed(s);
}

template<class Seq>
random_engine(
Seq& q, typename enable_if<has_generate_template<Seq>::value>::type* = 0
) {
seed(q);
}


void seed() {
for (unsigned short i = 0; i < 4; ++i) {
_k[i] = 0;
_s[i] = 0;
}
_o_counter = 0;

_o[0] = 0x09218ebde6c85537;
_o[1] = 0x55941f5266d86105;
_o[2] = 0x4bd25e16282434dc;
_o[3] = 0xee29ec846bd2e40b;
}

void seed(uint32_t s) {
for (unsigned short i = 0; i < 4; ++i) {
_k[i] = 0;
_s[i] = 0;
}
_k[0] = s;
_o_counter = 0;

encrypt_counter();
}

template<class Seq>
void seed(
Seq& q, typename enable_if<has_generate_template<Seq>::value>::type* = 0
) {
typename Seq::result_type w[8];
q.generate(&w[0], &w[8]);

for (unsigned short i = 0; i < 4; ++i) {
_k[i] = (static_cast<uint64_t>(w[2 * i]) << 32) | w[2 * i + 1];
_s[i] = 0;
}
_o_counter = 0;

encrypt_counter();
}

uint32_t operator()() {
if (_o_counter < 8) {
unsigned short _o_index = _o_counter >> 1;
_o_counter++;
if (_o_counter & 1)
return _o[_o_index] & 0xFFFFFFFF;
else
return _o[_o_index] >> 32;
}

inc_counter();
encrypt_counter();
_o_counter = 1; 
return _o[0] & 0xFFFFFFFF;   
}


void discard(uint64_t z) {
if (z < 8 - _o_counter) {
_o_counter += static_cast<unsigned short>(z);
return;
}

z -= (8 - _o_counter);  
_o_counter = z % 8;     
z -= _o_counter;        
z >>= 3;                
++z;                    
inc_counter(z);
encrypt_counter();
}

template<class CharT, class Traits>
friend std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, const random_engine& s) {
for (unsigned short i = 0; i < 4; ++i)
os << s._k[i] << ' ' << s._s[i] << ' ' << s._o[i] << ' ';
os << s._o_counter;
return os;
}

template<class CharT, class Traits>
friend std::basic_istream<CharT, Traits>&
operator>>(std::basic_istream<CharT, Traits>& is, random_engine& s) {
for (unsigned short i = 0; i < 4; ++i)
is >> s._k[i] >> s._s[i] >> s._o[i];
is >> s._o_counter;
return is;
}

bool operator==(const random_engine& y) {
if (_o_counter != y._o_counter) return false;
for (unsigned short i = 0; i < 4; ++i) {
if (_s[i] != y._s[i]) return false;
if (_k[i] != y._k[i]) return false;
if (_o[i] != y._o[i]) return false;
}
return true;
}

bool operator!=(const random_engine& y) {
return !(*this == y);
}

void set_key(
uint64_t k0 = 0,
uint64_t k1 = 0,
uint64_t k2 = 0,
uint64_t k3 = 0
) {
_k[0] = k0;
_k[1] = k1;
_k[2] = k2;
_k[3] = k3;
encrypt_counter();
}

void set_counter(
uint64_t s0 = 0,
uint64_t s1 = 0,
uint64_t s2 = 0,
uint64_t s3 = 0,
unsigned short o_counter = 0
) {
_s[0] = s0;
_s[1] = s1;
_s[2] = s2;
_s[3] = s3;
_o_counter = o_counter % 8;
encrypt_counter();
}


uint32_t version() {
return 5;
}

private:
void encrypt_counter() {
uint64_t b[4];
uint64_t k[5];

for (unsigned short i = 0; i < 4; ++i) b[i] = _s[i];
for (unsigned short i = 0; i < 4; ++i) k[i] = _k[i];

k[4] = 0x1BD11BDAA9FC1A22 ^ k[0] ^ k[1] ^ k[2] ^ k[3];

MIXK(b[0], b[1], 14, b[2], b[3], 16, k[0], k[1], k[2], k[3]);
MIX2(b[0], b[3], 52, b[2], b[1], 57);
MIX2(b[0], b[1], 23, b[2], b[3], 40);
MIX2(b[0], b[3], 5, b[2], b[1], 37);
MIXK(b[0], b[1], 25, b[2], b[3], 33, k[1], k[2], k[3], k[4] + 1);
MIX2(b[0], b[3], 46, b[2], b[1], 12);
MIX2(b[0], b[1], 58, b[2], b[3], 22);
MIX2(b[0], b[3], 32, b[2], b[1], 32);

MIXK(b[0], b[1], 14, b[2], b[3], 16, k[2], k[3], k[4], k[0] + 2);
MIX2(b[0], b[3], 52, b[2], b[1], 57);
MIX2(b[0], b[1], 23, b[2], b[3], 40);
MIX2(b[0], b[3], 5, b[2], b[1], 37);
MIXK(b[0], b[1], 25, b[2], b[3], 33, k[3], k[4], k[0], k[1] + 3);

MIX2(b[0], b[3], 46, b[2], b[1], 12);
MIX2(b[0], b[1], 58, b[2], b[3], 22);
MIX2(b[0], b[3], 32, b[2], b[1], 32);

MIXK(b[0], b[1], 14, b[2], b[3], 16, k[4], k[0], k[1], k[2] + 4);
MIX2(b[0], b[3], 52, b[2], b[1], 57);
MIX2(b[0], b[1], 23, b[2], b[3], 40);
MIX2(b[0], b[3], 5, b[2], b[1], 37);

for (unsigned int i = 0; i < 4; ++i) _o[i] = b[i] + k[i];
_o[3] += 5;
}

void inc_counter() {
++_s[0];
if (_s[0] != 0) return;

++_s[1];
if (_s[1] != 0) return;

++_s[2];
if (_s[2] != 0) return;

++_s[3];
}

void inc_counter(uint64_t z) {
if (z > 0xFFFFFFFFFFFFFFFF - _s[0]) {
++_s[1];
if (_s[1] == 0) {
++_s[2];
if (_s[2] == 0) {
++_s[3];
}
}
}
_s[0] += z;
}

private:
uint64_t _k[4];             
uint64_t _s[4];             
uint64_t _o[4];             
unsigned short _o_counter;  
};

} 

#undef MIXK
#undef MIX2

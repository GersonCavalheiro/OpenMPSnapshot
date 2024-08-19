#pragma once

#include <stdint.h>
#include <chrono>
#include <type_traits>
#include <cfloat>

namespace fastPRNG {
#define UNI_32BIT_INV 2.3283064365386962890625e-10
#define VNI_32BIT_INV 4.6566128730773925781250e-10   

#define UNI_64BIT_INV 5.42101086242752217003726400434970e-20
#define VNI_64BIT_INV 1.08420217248550443400745280086994e-19 

#define FPRNG_SEED_INIT64 std::chrono::system_clock::now().time_since_epoch().count()
#define FPRNG_SEED_INIT32 FPRNG_SEED_INIT64

inline static uint32_t splitMix32(const uint32_t val) {
uint32_t z = val + 0x9e3779b9;
z ^= z >> 15; 
z *= 0x85ebca6b;
z ^= z >> 13;
z *= 0xc2b2ae35;
return z ^ (z >> 16);
}

inline static uint64_t splitMix64(const uint64_t val) {
uint64_t z = val    + 0x9e3779b97f4a7c15;
z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
return z ^ (z >> 31);
}

template <typename T> inline static T rotl(const T x, const int k) { return (x << k) | (x >> (sizeof(T)*8 - k)); } 


#define XOSHIRO128\
const uint32_t t = s1 << 9;\
s2 ^= s0;\
s3 ^= s1;\
s1 ^= s2;\
s0 ^= s3;\
s2 ^= t;\
s3 = rotl<uint32_t>(s3, 11);\
return result;

#define XOROSHIRO64\
s1 ^= s0;\
s0 = rotl<uint32_t>(s0, 26) ^ s1 ^ (s1 << 9);\
s1 = rotl<uint32_t>(s1, 13);\
return result;

#define XORSHIFT32\
s0 ^= s0 << 13;\
s0 ^= s0 >> 17;\
s0 ^= s0 << 5;\
return s0;

#define XOSHIRO128_STATIC(FUNC)\
static const uint32_t seed = uint32_t(FPRNG_SEED_INIT32);\
static uint32_t s0 = splitMix32(seed), s1 = splitMix32(s0), s2 = splitMix32(s1), s3 = splitMix32(s2);\
FUNC; XOSHIRO128

#define XOROSHIRO64_STATIC(FUNC)\
static const uint32_t seed = uint32_t(FPRNG_SEED_INIT32);\
static uint32_t s0 = splitMix32(seed), s1 = splitMix32(s0);\
FUNC; XOROSHIRO64

#define XORSHIFT32_STATIC\
static uint32_t s0 = uint32_t(FPRNG_SEED_INIT32);\
XORSHIFT32

class fastXS32
{
public:
fastXS32(const uint32_t seedVal = uint32_t(FPRNG_SEED_INIT32)) { seed(seedVal); }

inline uint32_t xoshiro128p()  { return xoshiro128(s0 + s3); }
inline uint32_t xoshiro128pp() { return xoshiro128(rotl<uint32_t>(s0 + s3, 7) + s0); }
inline uint32_t xoshiro128xx() { return xoshiro128(rotl<uint32_t>(s1 * 5, 7) * 9); }

template <typename T> inline T xoshiro128p_UNI() { return T(        xoshiro128p() ) * UNI_32BIT_INV; } 
template <typename T> inline T xoshiro128p_VNI() { return T(int32_t(xoshiro128p())) * VNI_32BIT_INV; } 
template <typename T> inline T xoshiro128p_Range(T min, T max)                                         
{ return min + (max-min) * xoshiro128p_UNI<T>(); }

inline uint32_t xoroshiro64x()  { return xoroshiro64(               s0 * 0x9E3779BB); }
inline uint32_t xoroshiro64xx() { return xoroshiro64(rotl<uint32_t>(s0 * 0x9E3779BB, 5) * 5); }

template <typename T> inline T xoroshiro64x_UNI() { return T(         xoroshiro64x() ) * UNI_32BIT_INV; } 
template <typename T> inline T xoroshiro64x_VNI() { return T(int32_t(xoroshiro64x()))  * VNI_32BIT_INV; } 
template <typename T> inline T xoroshiro64x_Range(T min, T max)                                           
{ return min + (max-min) * xoroshiro64x_UNI<T>(); }

inline uint32_t xorShift() { XORSHIFT32 } 

template <typename T> inline T xorShift_UNI() { return         xorShift()  * UNI_32BIT_INV; } 
template <typename T> inline T xorShift_VNI() { return int32_t(xorShift()) * VNI_32BIT_INV; } 
template <typename T> inline T xorShift_Range(T min, T max)                                   
{ return min + (max-min) * xorShift_UNI<T>(); }

void seed(const uint32_t seedVal = uint32_t(FPRNG_SEED_INIT32)) {
s0 = splitMix32(seedVal);
s1 = splitMix32(s0);
s2 = splitMix32(s1);
s3 = splitMix32(s2);
}

private:
inline uint32_t xoroshiro64(const uint32_t result)  { XOROSHIRO64 }
inline uint32_t xoshiro128(const uint32_t result)   { XOSHIRO128  }

uint32_t s0, s1, s2, s3;
};

class fastXS32s
{
public:
fastXS32s() = default;

inline static uint32_t xoshiro128p()  { XOSHIRO128_STATIC(const uint32_t result = s0 + s3)               }
inline static uint32_t xoshiro128pp() { XOSHIRO128_STATIC(const uint32_t result = rotl<uint32_t>(s0 + s3, 7) + s0) }
inline static uint32_t xoshiro128xx() { XOSHIRO128_STATIC(const uint32_t result = rotl<uint32_t>(s1 * 5, 7) * 9)   }

template <typename T> inline static T xoshiro128p_UNI() { return T(        xoshiro128p() ) * UNI_32BIT_INV; } 
template <typename T> inline static T xoshiro128p_VNI() { return T(int32_t(xoshiro128p())) * VNI_32BIT_INV; } 
template <typename T> inline static T xoshiro128p_Range(T min, T max)                                         
{ return min + (max-min) * xoshiro128p_UNI<T>(); }

inline static uint32_t xoroshiro64x()  { XOROSHIRO64_STATIC(const uint32_t result =                s0 * 0x9E3779BB)         }
inline static uint32_t xoroshiro64xx() { XOROSHIRO64_STATIC(const uint32_t result = rotl<uint32_t>(s0 * 0x9E3779BB, 5) * 5) }

template <typename T> inline static T xoroshiro64x_UNI() { return T(         xoroshiro64x() ) * UNI_32BIT_INV; } 
template <typename T> inline static T xoroshiro64x_VNI() { return T(int32_t(xoroshiro64x()))  * VNI_32BIT_INV; } 
template <typename T> inline static T xoroshiro64x_Range(T min, T max)                                           
{ return min + (max-min) * xoroshiro64x_UNI<T>(); }

inline static uint32_t xorShift() { XORSHIFT32_STATIC } 

template <typename T> inline static T xorShift_UNI() { return         xorShift()  * UNI_32BIT_INV; } 
template <typename T> inline static T xorShift_VNI() { return int32_t(xorShift()) * VNI_32BIT_INV; } 
template <typename T> inline static T xorShift_Range(T min, T max)                                   
{ return min + (max-min) * xorShift_UNI<T>(); }
};

#undef XOSHIRO128
#undef XOROSHIRO64
#undef XORSHIFT32
#undef XOSHIRO128_STATIC
#undef XOROSHIRO64_STATIC
#undef XORSHIFT32_STATIC


#define XOSHIRO256\
const uint64_t t = s1 << 17;\
s2 ^= s0;\
s3 ^= s1;\
s1 ^= s2;\
s0 ^= s3;\
s2 ^= t;\
s3 = rotl<uint64_t>(s3, 45);\
return result;

#define XOROSHIRO128(A,B,C)\
s1 ^= s0;\
s0 = rotl<uint64_t>(s0, A) ^ s1 ^ (s1 << B);\
s1 = rotl<uint64_t>(s1, C);\
return result;

#define XORSHIFT64\
s0 ^= s0 << 13;\
s0 ^= s0 >> 7;\
s0 ^= s0 << 17;\
return s0;

#define XOSHIRO256_STATIC(FUNC)\
static const uint64_t seed = uint64_t(FPRNG_SEED_INIT64);\
static uint64_t s0 = splitMix64(seed), s1 = splitMix64(s0), s2 = splitMix64(s1), s3 = splitMix64(s2);\
FUNC; XOSHIRO256

#define XOROSHIRO128_STATIC(FUNC, A, B, C)\
static const uint64_t seed = uint64_t(FPRNG_SEED_INIT64);\
static uint64_t s0 = splitMix64(seed), s1 = splitMix64(s0);\
FUNC; XOROSHIRO128(A,B,C)

#define XORSHIFT64_STATIC\
static uint64_t s0 = uint64_t(FPRNG_SEED_INIT64);\
XORSHIFT64

class fastXS64
{
public:
fastXS64(const uint64_t seedVal = uint64_t(FPRNG_SEED_INIT64)) { seed(seedVal); }

inline uint64_t xoshiro256p()  { return xoshiro256(s0 + s3); }
inline uint64_t xoshiro256pp() { return xoshiro256(rotl<uint64_t>(s0 + s3, 23) + s0); }
inline uint64_t xoshiro256xx() { return xoshiro256(rotl<uint64_t>(s1 * 5, 7) * 9); }

template <typename T> inline T xoshiro256p_UNI() { return T(        xoshiro256p())  * UNI_64BIT_INV; } 
template <typename T> inline T xoshiro256p_VNI() { return T(int64_t(xoshiro256p())) * VNI_64BIT_INV; } 
template <typename T> inline T xoshiro256p_Range(T min, T max)                                         
{ return min + (max-min) * xoshiro256p_UNI<T>(); }

inline uint64_t xoroshiro128p()  { return xoroshiro128(     s0 + s1); }
inline uint64_t xoroshiro128pp() { return xoroshiro128(rotl<uint64_t>(s0 + s1, 17) + s0, 49, 21, 28); }
inline uint64_t xoroshiro128xx() { return xoroshiro128(rotl<uint64_t>(s0 * 5, 7) * 9); }

template <typename T> inline T xoroshiro128p_UNI() { return T(        xoshiro256p())  * UNI_64BIT_INV; } 
template <typename T> inline T xoroshiro128p_VNI() { return T(int64_t(xoshiro256p())) * VNI_64BIT_INV; } 
template <typename T> inline T xoroshiro128p_Range(T min, T max)                                         
{ return min + (max-min) * xoroshiro128p_UNI<T>(); }

inline uint64_t xorShift() { XORSHIFT64 } 

template <typename T> inline T xorShift_UNI() { return         xorShift()  * UNI_64BIT_INV; } 
template <typename T> inline T xorShift_VNI() { return int64_t(xorShift()) * VNI_64BIT_INV; } 
template <typename T> inline T xorShift_Range(T min, T max)                                   
{ return min + (max-min) * xorShift_UNI<T>(); }

void seed(const uint64_t seedVal = uint64_t(FPRNG_SEED_INIT64)) {
s0 = splitMix64(seedVal);
s1 = splitMix64(s0);
s2 = splitMix64(s1);
s3 = splitMix64(s2);
}
private:
inline uint64_t xoshiro256(const uint64_t result)   { XOSHIRO256 }
inline uint64_t xoroshiro128(const uint64_t result, const int A = 24, const int B = 16, const int C = 37) { XOROSHIRO128(A,B,C) }

uint64_t s0, s1, s2, s3;
};

class fastXS64s
{
public:
fastXS64s() = default;

inline static uint64_t xoshiro256p()  { XOSHIRO256_STATIC(const uint64_t result = s0 + s3) }
inline static uint64_t xoshiro256pp() { XOSHIRO256_STATIC(const uint64_t result = rotl<uint64_t>(s0 + s3, 23) + s0) }
inline static uint64_t xoshiro256xx() { XOSHIRO256_STATIC(const uint64_t result = rotl<uint64_t>(s1 * 5, 7) * 9) }

template <typename T> inline static T xoshiro256p_UNI() { return T(        xoshiro256p())  * UNI_64BIT_INV; } 
template <typename T> inline static T xoshiro256p_VNI() { return T(int64_t(xoshiro256p())) * VNI_64BIT_INV; } 
template <typename T> inline static T xoshiro256p_Range(T min, T max)                                         
{ return min + (max-min) * xoshiro256p_UNI<T>(); }

inline static uint64_t xoroshiro128p()  { XOROSHIRO128_STATIC(const uint64_t result =      s0 + s1,           24, 13, 27) }
inline static uint64_t xoroshiro128pp() { XOROSHIRO128_STATIC(const uint64_t result = rotl<uint64_t>(s0 + s1, 17) + s0, 49, 21, 28) }
inline static uint64_t xoroshiro128xx() { XOROSHIRO128_STATIC(const uint64_t result = rotl<uint64_t>(s0 * 5, 7) * 9,    24, 13, 27) }

template <typename T> inline static T xoroshiro128p_UNI() { return T(        xoshiro256p())  * UNI_64BIT_INV; } 
template <typename T> inline static T xoroshiro128p_VNI() { return T(int64_t(xoshiro256p())) * VNI_64BIT_INV; } 
template <typename T> inline static T xoroshiro128p_Range(T min, T max)                                         
{ return min + (max-min) * xoroshiro128p_UNI<T>(); }

inline static uint64_t xorShift() { XORSHIFT64_STATIC } 

template <typename T> inline static T xorShift_UNI() { return         xorShift()  * UNI_64BIT_INV; } 
template <typename T> inline static T xorShift_VNI() { return int64_t(xorShift()) * VNI_64BIT_INV; } 
template <typename T> inline static T xorShift_Range(T min, T max)                                   
{ return min + (max-min) * xorShift_UNI<T>(); }
};

#undef XOSHIRO256
#undef XOROSHIRO128
#undef XORSHIFT64
#undef XOSHIRO256_STATIC
#undef XOROSHIRO128_STATIC
#undef XORSHIFT64_STATIC



class fastRandom32Class
{
public:

fastRandom32Class(const uint32_t seedVal = uint32_t(FPRNG_SEED_INIT32))
{ reset(); seed(seedVal); }

void seed(const uint32_t seed = uint32_t(FPRNG_SEED_INIT32)) {
uint32_t s[6];
s[0] = splitMix32(seed);
for(int i=1; i<6; i++) s[i] = splitMix32(s[i-1]);
initialize(s);
}

void reset() {
z   = 362436069; w     = 521288629;
jsr = 123456789; jcong = 380116160;
a   = 224466889; b     = 7584631;
}

inline uint32_t znew() { return z=36969*(z&65535)+(z>>16); }
inline uint32_t wnew() { return w=18000*(w&65535)+(w>>16); }
inline uint32_t MWC()  { return (znew()<<16)+wnew()      ; }
inline uint32_t CNG()  { return jcong=69069*jcong+1234567; }
inline uint32_t FIB()  { return (b=a+b),(a=b-a)          ; }
inline uint32_t XSH()  { return jsr^=(jsr<<17), jsr^=(jsr>>13), jsr^=(jsr<<5); }

inline uint32_t KISS() { return (MWC()^CNG())+XSH(); } 

template <typename T> inline T KISS_UNI() { return         KISS()  * UNI_32BIT_INV; } 
template <typename T> inline T KISS_VNI() { return int32_t(KISS()) * VNI_32BIT_INV; } 
template <typename T> inline T KISS_Range(T min, T max)                               
{ return min + (max-min) * KISS_UNI<T>(); }

#ifdef FSTRND_USES_BUILT_TABLE
inline uint32_t LFIB4() { return c++,t[c]=t[c]+t[uint8_t(c+58)]+t[uint8_t(c+119)]+t[uint8_t(c+178)];}
inline uint32_t SWB()   { uint32_t bro; return c++,bro=(x<y),t[c]=(x=t[uint8_t(c+34)])-(y=t[uint8_t(c+19)]+bro); }
#endif
private:
void initialize(const uint32_t *i) { z+=i[0]; w+=i[1]; jsr+=i[2]; jcong+=i[3]; a=+i[4]; b=+i[5]; }

uint32_t z, w, jsr, jcong;
uint32_t a, b;

#ifdef FSTRND_USES_BUILT_TABLE
uint32_t t[256];
unsigned char c=0;
#endif
};

#undef FSTRND_USES_BUILT_TABLE




class fastRandom64Class
{
public:
fastRandom64Class(const uint64_t seedVal = uint64_t(FPRNG_SEED_INIT64)) { reset(); seed(seedVal);  }

void seed(const uint64_t seed = uint64_t(FPRNG_SEED_INIT64)) {
uint64_t s[6];
s[0] = splitMix64(seed);
for(int i=1; i<6; i++) s[i] = splitMix64(s[i-1]);
initialize(s);
}
void reset() {
x=uint64_t(1234567890987654321ULL); c=uint64_t(123456123456123456ULL);
y=uint64_t(362436362436362436ULL ); z=uint64_t(1066149217761810ULL  );
a=uint64_t(224466889);              b=uint64_t(7584631);
}

inline uint64_t MWC() { uint64_t t; return t=(x<<58)+c, c=(x>>6), x+=t, c+=(x<t), x; }
inline uint64_t CNG() { return z=6906969069LL*z+1234567;            }
inline uint64_t XSH() { return y^=(y<<13), y^=(y>>17), y^=(y<<43);  }
inline uint64_t FIB() { return (b=a+b),(a=b-a);                     }

inline uint64_t KISS () { return MWC()+XSH()+CNG(); } 

template <typename T> inline T KISS_UNI() { return         KISS()  * UNI_64BIT_INV; } 
template <typename T> inline T KISS_VNI() { return int64_t(KISS()) * VNI_64BIT_INV; } 
template <typename T> inline T KISS_Range(T min, T max)                               
{ return min + (max-min) * KISS_UNI<T>(); }

private:
void initialize(const uint64_t *i){ x+=i[0]; y+=i[1]; z+=i[2]; c+=i[3]; a=+i[4]; b=+i[5]; }

uint64_t x, c, y, z;
uint64_t a, b;
};

using fastRand32 = fastRandom32Class;
using fastRand64 = fastRandom64Class;

} 

#undef UNI_32BIT_INV
#undef VNI_32BIT_INV
#undef UNI_64BIT_INV
#undef VNI_64BIT_INV
#undef FPRNG_SEED_INIT32
#undef FPRNG_SEED_INIT64











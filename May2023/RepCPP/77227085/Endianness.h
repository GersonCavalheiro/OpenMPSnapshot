

#pragma once

#include "common/Common.h" 
#include <cassert>         
#include <cstdint>         
#include <cstring>         

namespace rawspeed {

enum class Endianness { little = 0xDEAD, big = 0xBEEF, unknown = 0x0BAD };

inline Endianness getHostEndiannessRuntime() {
uint16_t testvar = 0xfeff;
uint32_t firstbyte = (reinterpret_cast<uint8_t*>(&testvar))[0];
if (firstbyte == 0xff)
return Endianness::little;
if (firstbyte == 0xfe)
return Endianness::big;

assert(false);

return Endianness::unknown;
}

inline Endianness getHostEndianness() {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
return Endianness::little;
#elif defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
return Endianness::big;
#elif defined(__BYTE_ORDER__)
#error "uhm, __BYTE_ORDER__ has some strange value"
#else
return getHostEndiannessRuntime();
#endif
}

#ifdef _MSC_VER
#include <intrin.h>

#define BSWAP16(A) _byteswap_ushort(A)
#define BSWAP32(A) _byteswap_ulong(A)
#define BSWAP64(A) _byteswap_uint64(A)
#else
#define BSWAP16(A) __builtin_bswap16(A)
#define BSWAP32(A) __builtin_bswap32(A)
#define BSWAP64(A) __builtin_bswap64(A)
#endif

inline int16_t getByteSwapped(int16_t v) {
return static_cast<int16_t>(BSWAP16(static_cast<uint16_t>(v)));
}
inline uint16_t getByteSwapped(uint16_t v) { return BSWAP16(v); }
inline int32_t getByteSwapped(int32_t v) {
return static_cast<int32_t>(BSWAP32(static_cast<uint32_t>(v)));
}
inline uint32_t getByteSwapped(uint32_t v) { return BSWAP32(v); }
inline uint64_t getByteSwapped(uint64_t v) { return BSWAP64(v); }

inline float getByteSwapped(float f) {
auto i = bit_cast<uint32_t>(f);
i = getByteSwapped(i);
return bit_cast<float>(i);
}
inline double getByteSwapped(double d) {
auto i = bit_cast<uint64_t>(d);
i = getByteSwapped(i);
return bit_cast<double>(i);
}

template <typename T> inline T getByteSwapped(const void* data, bool bswap) {
T ret;
memcpy(&ret, data, sizeof(T));
return bswap ? getByteSwapped(ret) : ret;
}


template <typename T> inline T getBE(const void* data) {
return getByteSwapped<T>(data, getHostEndianness() == Endianness::little);
}

template <typename T> inline T getLE(const void* data) {
return getByteSwapped<T>(data, getHostEndianness() == Endianness::big);
}

inline uint16_t getU16BE(const void* data) { return getBE<uint16_t>(data); }
inline uint16_t getU16LE(const void* data) { return getLE<uint16_t>(data); }
inline uint32_t getU32BE(const void* data) { return getBE<uint32_t>(data); }
inline uint32_t getU32LE(const void* data) { return getLE<uint32_t>(data); }

#undef BSWAP64
#undef BSWAP32
#undef BSWAP16

} 

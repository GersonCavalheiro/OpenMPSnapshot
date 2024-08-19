
#ifndef ABSL_BASE_INTERNAL_ENDIAN_H_
#define ABSL_BASE_INTERNAL_ENDIAN_H_

#ifdef _MSC_VER
#include <stdlib.h>  
#elif defined(__APPLE__)
#include <libkern/OSByteOrder.h>
#elif defined(__FreeBSD__)
#include <sys/endian.h>
#elif defined(__GLIBC__)
#include <byteswap.h>  
#endif

#include <cstdint>
#include "absl/base/config.h"
#include "absl/base/internal/unaligned_access.h"
#include "absl/base/port.h"

namespace absl {

#if defined(__clang__) || \
(defined(__GNUC__) && \
((__GNUC__ == 4 && __GNUC_MINOR__ >= 8) || __GNUC__ >= 5))
inline uint64_t gbswap_64(uint64_t host_int) {
return __builtin_bswap64(host_int);
}
inline uint32_t gbswap_32(uint32_t host_int) {
return __builtin_bswap32(host_int);
}
inline uint16_t gbswap_16(uint16_t host_int) {
return __builtin_bswap16(host_int);
}

#elif defined(_MSC_VER)
inline uint64_t gbswap_64(uint64_t host_int) {
return _byteswap_uint64(host_int);
}
inline uint32_t gbswap_32(uint32_t host_int) {
return _byteswap_ulong(host_int);
}
inline uint16_t gbswap_16(uint16_t host_int) {
return _byteswap_ushort(host_int);
}

#elif defined(__APPLE__)
inline uint64_t gbswap_64(uint64_t host_int) { return OSSwapInt16(host_int); }
inline uint32_t gbswap_32(uint32_t host_int) { return OSSwapInt32(host_int); }
inline uint16_t gbswap_16(uint16_t host_int) { return OSSwapInt64(host_int); }

#else
inline uint64_t gbswap_64(uint64_t host_int) {
#if defined(__GNUC__) && defined(__x86_64__) && !defined(__APPLE__)
if (__builtin_constant_p(host_int)) {
return __bswap_constant_64(host_int);
} else {
register uint64_t result;
__asm__("bswap %0" : "=r"(result) : "0"(host_int));
return result;
}
#elif defined(__GLIBC__)
return bswap_64(host_int);
#else
return (((host_int & uint64_t{0xFF}) << 56) |
((host_int & uint64_t{0xFF00}) << 40) |
((host_int & uint64_t{0xFF0000}) << 24) |
((host_int & uint64_t{0xFF000000}) << 8) |
((host_int & uint64_t{0xFF00000000}) >> 8) |
((host_int & uint64_t{0xFF0000000000}) >> 24) |
((host_int & uint64_t{0xFF000000000000}) >> 40) |
((host_int & uint64_t{0xFF00000000000000}) >> 56));
#endif  
}

inline uint32_t gbswap_32(uint32_t host_int) {
#if defined(__GLIBC__)
return bswap_32(host_int);
#else
return (((host_int & uint32_t{0xFF}) << 24) |
((host_int & uint32_t{0xFF00}) << 8) |
((host_int & uint32_t{0xFF0000}) >> 8) |
((host_int & uint32_t{0xFF000000}) >> 24));
#endif
}

inline uint16_t gbswap_16(uint16_t host_int) {
#if defined(__GLIBC__)
return bswap_16(host_int);
#else
return (((host_int & uint16_t{0xFF}) << 8) |
((host_int & uint16_t{0xFF00}) >> 8));
#endif
}

#endif  

#ifdef ABSL_IS_LITTLE_ENDIAN

inline uint16_t ghtons(uint16_t x) { return gbswap_16(x); }
inline uint32_t ghtonl(uint32_t x) { return gbswap_32(x); }
inline uint64_t ghtonll(uint64_t x) { return gbswap_64(x); }

#elif defined ABSL_IS_BIG_ENDIAN

inline uint16_t ghtons(uint16_t x) { return x; }
inline uint32_t ghtonl(uint32_t x) { return x; }
inline uint64_t ghtonll(uint64_t x) { return x; }

#else
#error \
"Unsupported byte order: Either ABSL_IS_BIG_ENDIAN or " \
"ABSL_IS_LITTLE_ENDIAN must be defined"
#endif  

inline uint16_t gntohs(uint16_t x) { return ghtons(x); }
inline uint32_t gntohl(uint32_t x) { return ghtonl(x); }
inline uint64_t gntohll(uint64_t x) { return ghtonll(x); }

namespace little_endian {
#ifdef ABSL_IS_LITTLE_ENDIAN

inline uint16_t FromHost16(uint16_t x) { return x; }
inline uint16_t ToHost16(uint16_t x) { return x; }

inline uint32_t FromHost32(uint32_t x) { return x; }
inline uint32_t ToHost32(uint32_t x) { return x; }

inline uint64_t FromHost64(uint64_t x) { return x; }
inline uint64_t ToHost64(uint64_t x) { return x; }

inline constexpr bool IsLittleEndian() { return true; }

#elif defined ABSL_IS_BIG_ENDIAN

inline uint16_t FromHost16(uint16_t x) { return gbswap_16(x); }
inline uint16_t ToHost16(uint16_t x) { return gbswap_16(x); }

inline uint32_t FromHost32(uint32_t x) { return gbswap_32(x); }
inline uint32_t ToHost32(uint32_t x) { return gbswap_32(x); }

inline uint64_t FromHost64(uint64_t x) { return gbswap_64(x); }
inline uint64_t ToHost64(uint64_t x) { return gbswap_64(x); }

inline constexpr bool IsLittleEndian() { return false; }

#endif 

inline uint16_t Load16(const void *p) {
return ToHost16(ABSL_INTERNAL_UNALIGNED_LOAD16(p));
}

inline void Store16(void *p, uint16_t v) {
ABSL_INTERNAL_UNALIGNED_STORE16(p, FromHost16(v));
}

inline uint32_t Load32(const void *p) {
return ToHost32(ABSL_INTERNAL_UNALIGNED_LOAD32(p));
}

inline void Store32(void *p, uint32_t v) {
ABSL_INTERNAL_UNALIGNED_STORE32(p, FromHost32(v));
}

inline uint64_t Load64(const void *p) {
return ToHost64(ABSL_INTERNAL_UNALIGNED_LOAD64(p));
}

inline void Store64(void *p, uint64_t v) {
ABSL_INTERNAL_UNALIGNED_STORE64(p, FromHost64(v));
}

}  

namespace big_endian {
#ifdef ABSL_IS_LITTLE_ENDIAN

inline uint16_t FromHost16(uint16_t x) { return gbswap_16(x); }
inline uint16_t ToHost16(uint16_t x) { return gbswap_16(x); }

inline uint32_t FromHost32(uint32_t x) { return gbswap_32(x); }
inline uint32_t ToHost32(uint32_t x) { return gbswap_32(x); }

inline uint64_t FromHost64(uint64_t x) { return gbswap_64(x); }
inline uint64_t ToHost64(uint64_t x) { return gbswap_64(x); }

inline constexpr bool IsLittleEndian() { return true; }

#elif defined ABSL_IS_BIG_ENDIAN

inline uint16_t FromHost16(uint16_t x) { return x; }
inline uint16_t ToHost16(uint16_t x) { return x; }

inline uint32_t FromHost32(uint32_t x) { return x; }
inline uint32_t ToHost32(uint32_t x) { return x; }

inline uint64_t FromHost64(uint64_t x) { return x; }
inline uint64_t ToHost64(uint64_t x) { return x; }

inline constexpr bool IsLittleEndian() { return false; }

#endif 

inline uint16_t Load16(const void *p) {
return ToHost16(ABSL_INTERNAL_UNALIGNED_LOAD16(p));
}

inline void Store16(void *p, uint16_t v) {
ABSL_INTERNAL_UNALIGNED_STORE16(p, FromHost16(v));
}

inline uint32_t Load32(const void *p) {
return ToHost32(ABSL_INTERNAL_UNALIGNED_LOAD32(p));
}

inline void Store32(void *p, uint32_t v) {
ABSL_INTERNAL_UNALIGNED_STORE32(p, FromHost32(v));
}

inline uint64_t Load64(const void *p) {
return ToHost64(ABSL_INTERNAL_UNALIGNED_LOAD64(p));
}

inline void Store64(void *p, uint64_t v) {
ABSL_INTERNAL_UNALIGNED_STORE64(p, FromHost64(v));
}

}  

}  

#endif  

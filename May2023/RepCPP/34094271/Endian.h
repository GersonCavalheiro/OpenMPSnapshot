#pragma once



#include <stdlib.h>
#include <stdint.h>

#define bswap_64(x)                            \
((((x) & 0xff00000000000000ull) >> 56)       \
| (((x) & 0x00ff000000000000ull) >> 40)     \
| (((x) & 0x0000ff0000000000ull) >> 24)     \
| (((x) & 0x000000ff00000000ull) >> 8)      \
| (((x) & 0x00000000ff000000ull) << 8)      \
| (((x) & 0x0000000000ff0000ull) << 24)     \
| (((x) & 0x000000000000ff00ull) << 40)     \
| (((x) & 0x00000000000000ffull) << 56))

#define bswap_32(x)               \
((((x) & 0xff000000) >> 24)     \
| (((x) & 0x00ff0000) >> 8)    \
| (((x) & 0x0000ff00) << 8)    \
| (((x) & 0x000000ff) << 24))

#define bswap_16(x)          \
((((x) & 0xff00) >> 8)     \
| (((x) & 0x00ff) << 8))

enum ENDIANNESS {
LT_ENDIAN,
BG_ENDIAN
};

inline ENDIANNESS GetEndian() {
long int endian = 0x0000000000000001;
return (*(char *) &endian == 0x01) ? LT_ENDIAN : BG_ENDIAN;
}

inline uint64_t htof64(uint64_t host_integer) {
if(GetEndian() == LT_ENDIAN) {
return host_integer;
} else {
return bswap_64(host_integer);
}
}

inline uint64_t ftoh64(uint64_t file_integer) {
if(GetEndian() == LT_ENDIAN) {
return file_integer;
} else {
return bswap_64(file_integer);
}
}

inline uint32_t htof32(uint32_t host_integer) {
if(GetEndian() == LT_ENDIAN) {
return host_integer;
} else {
return bswap_32(host_integer);
}
}

inline uint32_t ftoh32(uint32_t file_integer) {
if(GetEndian() == LT_ENDIAN) {
return file_integer;
} else {
return bswap_32(file_integer);
}
}

inline uint16_t htof16(uint32_t host_integer) {
if(GetEndian() == LT_ENDIAN) {
return host_integer;
} else {
return bswap_16(host_integer);
}
}

inline uint16_t ftoh16(uint16_t file_integer) {
if(GetEndian() == LT_ENDIAN) {
return file_integer;
} else {
return bswap_16(file_integer);
}
}

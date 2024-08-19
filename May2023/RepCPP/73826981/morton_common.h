#pragma once


#include <stdint.h>
#if _MSC_VER
#include <intrin.h>
#endif

template<typename morton>
inline bool findFirstSetBit(const morton x, unsigned long* firstbit_location) {
#if _MSC_VER && !_WIN64
if (sizeof(morton) <= 4) {
return _BitScanReverse(firstbit_location, x) != 0;
}
else {
*firstbit_location = 0;
if (_BitScanReverse(firstbit_location, (x >> 32))) { 
firstbit_location += 32;
return true;
}
return _BitScanReverse(firstbit_location, (x & 0xFFFFFFFF)) != 0;
}
#elif  _MSC_VER && _WIN64
return _BitScanReverse64(firstbit_location, x) != 0;
#elif __GNUC__
if (x == 0) {
return false;
}
else {
*firstbit_location = static_cast<unsigned long>((sizeof(morton)*8) - __builtin_clzll(x));
return true;
}
#endif
}
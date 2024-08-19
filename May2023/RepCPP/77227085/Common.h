

#pragma once

#include "rawspeedconfig.h" 
#include "adt/Invariant.h"  
#include <algorithm>        
#include <array>            
#include <cassert>          
#include <climits>          
#include <cstdint>          
#include <cstring>          
#include <initializer_list> 
#include <string>           
#include <string_view>      
#include <type_traits>      
#include <vector>           

extern "C" int rawspeed_get_number_of_processor_cores();

namespace rawspeed {

enum class DEBUG_PRIO {
ERROR = 0x10,
WARNING = 0x100,
INFO = 0x1000,
EXTRA = 0x10000
};

void writeLog(DEBUG_PRIO priority, const char* format, ...)
__attribute__((format(printf, 2, 3)));

inline void copyPixels(uint8_t* dest, int dstPitch, const uint8_t* src,
int srcPitch, int rowSize, int height) {
if (height == 1 || (dstPitch == srcPitch && srcPitch == rowSize))
memcpy(dest, src, static_cast<size_t>(rowSize) * height);
else {
for (int y = height; y > 0; --y) {
memcpy(dest, src, rowSize);
dest += dstPitch;
src += srcPitch;
}
}
}

template <typename T_TO, typename T_FROM,
typename = std::enable_if_t<sizeof(T_TO) == sizeof(T_FROM)>,
typename = std::enable_if_t<std::is_trivially_constructible_v<T_TO>>,
typename = std::enable_if_t<std::is_trivially_copyable_v<T_TO>>,
typename = std::enable_if_t<std::is_trivially_copyable_v<T_FROM>>>
inline T_TO bit_cast(const T_FROM& from) noexcept {
T_TO to;
memcpy(&to, &from, sizeof(T_TO));
return to;
}

template <typename T> constexpr bool RAWSPEED_READNONE isPowerOfTwo(T val) {
return (val & (~val + 1)) == val;
}

template <class T>
constexpr unsigned RAWSPEED_READNONE bitwidth([[maybe_unused]] T unused = {}) {
return CHAR_BIT * sizeof(T);
}

template <typename T>
constexpr size_t RAWSPEED_READNONE getMisalignmentOffset(
T value, size_t multiple,
typename std::enable_if<std::is_pointer_v<T>>::type*  = nullptr) {
if (multiple == 0)
return 0;
static_assert(bitwidth<uintptr_t>() >= bitwidth<T>(),
"uintptr_t can not represent all pointer values?");
return reinterpret_cast<uintptr_t>(value) % multiple;
}

template <typename T>
constexpr size_t RAWSPEED_READNONE getMisalignmentOffset(
T value, size_t multiple,
typename std::enable_if<std::is_integral_v<T>>::type*  =
nullptr) {
if (multiple == 0)
return 0;
return value % multiple;
}

template <typename T>
constexpr T RAWSPEED_READNONE roundToMultiple(T value, size_t multiple,
bool roundDown) {
size_t offset = getMisalignmentOffset(value, multiple);
if (offset == 0)
return value;
T roundedDown = value - offset;
if (roundDown) 
return roundedDown;
return roundedDown + multiple;
}

constexpr size_t RAWSPEED_READNONE roundDown(size_t value, size_t multiple) {
return roundToMultiple(value, multiple, true);
}

constexpr size_t RAWSPEED_READNONE roundUp(size_t value, size_t multiple) {
return roundToMultiple(value, multiple, false);
}

constexpr size_t RAWSPEED_READNONE roundUpDivision(size_t value, size_t div) {
return (value != 0) ? (1 + ((value - 1) / div)) : 0;
}

template <class T>
constexpr RAWSPEED_READNONE bool isAligned(T value, size_t multiple) {
return (multiple == 0) || (getMisalignmentOffset(value, multiple) == 0);
}

template <typename T, typename T2>
bool RAWSPEED_READONLY isIn(const T value,
const std::initializer_list<T2>& list) {
return std::any_of(list.begin(), list.end(),
[value](const T2& t) { return t == value; });
}

template <typename T>
constexpr uint16_t RAWSPEED_READNONE clampBits(
T value, unsigned int nBits,
typename std::enable_if_t<std::is_arithmetic_v<T>>*  = nullptr) {
invariant(nBits <= 16);
invariant(bitwidth<T>() > nBits); 
const T maxVal = (T(1) << nBits) - T(1);
return std::clamp(value, T(0), maxVal);
}

template <typename T>
constexpr bool RAWSPEED_READNONE isIntN(
T value, unsigned int nBits,
typename std::enable_if_t<std::is_arithmetic_v<T>>*  = nullptr) {
invariant(nBits < bitwidth<T>() && "Check must not be tautological.");
using UnsignedT = std::make_unsigned_t<T>;
const auto highBits = static_cast<UnsignedT>(value) >> nBits;
return highBits == 0;
}

template <class T, typename std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
constexpr int countl_zero(T x) noexcept {
if (x == T(0))
return bitwidth<T>();
return __builtin_clz(x);
}

template <class T>
constexpr RAWSPEED_READNONE T extractHighBits(
T value, unsigned nBits, unsigned effectiveBitwidth = bitwidth<T>(),
typename std::enable_if_t<std::is_unsigned_v<T>>*  = nullptr) {
invariant(effectiveBitwidth <= bitwidth<T>());
invariant(nBits <= effectiveBitwidth);
auto numLowBitsToSkip = effectiveBitwidth - nBits;
invariant(numLowBitsToSkip < bitwidth<T>());
return value >> numLowBitsToSkip;
}

template <typename T>
constexpr typename std::make_signed_t<T> RAWSPEED_READNONE signExtend(
T value, unsigned int nBits,
typename std::enable_if_t<std::is_unsigned_v<T>>*  = nullptr) {
invariant(nBits != 0 && "Only valid for non-zero bit count.");
const T SpareSignBits = bitwidth<T>() - nBits;
using SignedT = std::make_signed_t<T>;
return static_cast<SignedT>(value << SpareSignBits) >> SpareSignBits;
}

inline std::string trimSpaces(std::string_view str) {
size_t startpos = str.find_first_not_of(" \t");

size_t endpos = str.find_last_not_of(" \t");

if ((startpos == std::string::npos) || (endpos == std::string::npos))
return "";

str = str.substr(startpos, endpos - startpos + 1);
return {str.begin(), str.end()};
}

inline std::vector<std::string> splitString(const std::string& input,
char c = ' ') {
std::vector<std::string> result;
const char* str = input.c_str();

while (true) {
const char* begin = str;

while (*str != c && *str != '\0')
str++;

if (begin != str)
result.emplace_back(begin, str);

const bool isNullTerminator = (*str == '\0');
str++;

if (isNullTerminator)
break;
}

return result;
}

template <int N, typename T>
inline std::array<T, N> to_array(const std::vector<T>& v) {
std::array<T, N> a;
assert(v.size() == N && "Size mismatch");
std::move(v.begin(), v.end(), a.begin());
return a;
}

enum class BitOrder {
LSB,   
MSB,   
MSB16, 
MSB32, 
};

} 

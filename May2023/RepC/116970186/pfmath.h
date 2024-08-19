#pragma once
#include "stdafx.h"
#define DIVUP(x, y) (((x) % (y) > 0 ? (x) / (y) + 1 : (x) / (y)))
#define BITCOUNT_IMP(T, NAME) static inline T NAME(const T n) \
{ \
T v = n - ((n >> 1) & (T)~(T)0 / 3); \
v = (v & (T)~(T)0 / 15 * 3) + ((v >> 2) & (T)~(T)0 / 15 * 3); \
v = (v + (v >> 4)) & (T)~(T)0 / 255 * 15; \
return (T)(v * ((T)~(T)0 / 255)) >> (sizeof(T) - 1) * CHAR_BIT; \
}
BITCOUNT_IMP(BITARRAY_WORD, bitcount)
#define OR_MASK(x) ((BITARRAY_WORD)1 << (x))
#define AND_MASK(x) (~OR_MASK(x))

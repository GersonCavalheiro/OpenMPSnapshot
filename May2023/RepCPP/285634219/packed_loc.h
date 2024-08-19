

#ifndef PACKED_LOC_H_
#define PACKED_LOC_H_

#include <stdint.h>
#include "../util/system.h"

#pragma pack(1)

struct packed_uint40_t
{
uint8_t		high;
uint32_t	low;
packed_uint40_t():
high (),
low ()
{ }
packed_uint40_t(uint64_t v):
high ((uint8_t)(v>>32)),
low ((uint32_t)(v&0xfffffffflu))
{ }
operator const uint64_t() const
{ return (uint64_t(high) << 32) | low; }
bool operator<(const packed_uint40_t &rhs) const
{ return high < rhs.high || (high == rhs.high && low < rhs.low); }
friend uint64_t operator-(const packed_uint40_t &x, const packed_uint40_t &y)
{ return (const uint64_t)(x) - (const uint64_t)(y); }
} PACKED_ATTRIBUTE ;

typedef packed_uint40_t Packed_loc;
typedef size_t Loc;

#pragma pack()

#endif 

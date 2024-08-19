

#pragma once

#include "io/BitStream.h"  
#include "io/Buffer.h"     
#include "io/Endianness.h" 
#include <cstdint>         

namespace rawspeed {

struct MSB16BitPumpTag;


using BitPumpMSB16 = BitStream<MSB16BitPumpTag, BitStreamCacheRightInLeftOut>;

template <> struct BitStreamTraits<MSB16BitPumpTag> final {
static constexpr int MaxProcessBytes = 4;
};

template <>
inline BitPumpMSB16::size_type BitPumpMSB16::fillCache(const uint8_t* input) {
static_assert(BitStreamCacheBase::MaxGetBits >= 32, "check implementation");

for (size_type i = 0; i < 4; i += sizeof(uint16_t))
cache.push(getLE<uint16_t>(input + i), 16);
return 4;
}

} 

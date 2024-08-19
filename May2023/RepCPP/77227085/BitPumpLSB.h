

#pragma once

#include "io/BitStream.h"  
#include "io/Buffer.h"     
#include "io/Endianness.h" 
#include <cstdint>         

namespace rawspeed {

struct LSBBitPumpTag;


using BitPumpLSB = BitStream<LSBBitPumpTag, BitStreamCacheLeftInRightOut>;

template <> struct BitStreamTraits<LSBBitPumpTag> final {
static constexpr int MaxProcessBytes = 4;
};

template <>
inline BitPumpLSB::size_type BitPumpLSB::fillCache(const uint8_t* input) {
static_assert(BitStreamCacheBase::MaxGetBits >= 32, "check implementation");

cache.push(getLE<uint32_t>(input), 32);
return 4;
}

} 

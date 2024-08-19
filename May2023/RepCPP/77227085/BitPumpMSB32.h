

#pragma once

#include "io/BitStream.h"  
#include "io/Buffer.h"     
#include "io/Endianness.h" 
#include <cstdint>         

namespace rawspeed {

struct MSB32BitPumpTag;


using BitPumpMSB32 = BitStream<MSB32BitPumpTag, BitStreamCacheRightInLeftOut>;

template <> struct BitStreamTraits<MSB32BitPumpTag> final {
static constexpr bool canUseWithPrefixCodeDecoder = true;

static constexpr int MaxProcessBytes = 4;
};

template <>
inline BitPumpMSB32::size_type BitPumpMSB32::fillCache(const uint8_t* input) {
static_assert(BitStreamCacheBase::MaxGetBits >= 32, "check implementation");

cache.push(getLE<uint32_t>(input), 32);
return 4;
}

} 

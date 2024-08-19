

#pragma once

#include "io/BitStream.h"  
#include "io/Buffer.h"     
#include "io/Endianness.h" 
#include <cstdint>         

namespace rawspeed {

struct MSBBitPumpTag;


using BitPumpMSB = BitStream<MSBBitPumpTag, BitStreamCacheRightInLeftOut>;

template <> struct BitStreamTraits<MSBBitPumpTag> final {
static constexpr bool canUseWithPrefixCodeDecoder = true;

static constexpr int MaxProcessBytes = 4;
};

template <>
inline BitPumpMSB::size_type BitPumpMSB::fillCache(const uint8_t* input) {
static_assert(BitStreamCacheBase::MaxGetBits >= 32, "check implementation");

cache.push(getBE<uint32_t>(input), 32);
return 4;
}

} 

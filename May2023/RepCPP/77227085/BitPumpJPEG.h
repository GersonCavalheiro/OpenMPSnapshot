

#pragma once

#include "common/Common.h" 
#include "io/BitStream.h"  
#include "io/Buffer.h"     
#include "io/Endianness.h" 
#include <algorithm>       
#include <array>           
#include <cstdint>         

namespace rawspeed {

struct JPEGBitPumpTag;

using BitPumpJPEG = BitStream<JPEGBitPumpTag, BitStreamCacheRightInLeftOut>;

template <> struct BitStreamTraits<JPEGBitPumpTag> final {
static constexpr bool canUseWithPrefixCodeDecoder = true;

static constexpr int MaxProcessBytes = 8;
};

template <>
inline BitPumpJPEG::size_type BitPumpJPEG::fillCache(const uint8_t* input) {
static_assert(BitStreamCacheBase::MaxGetBits >= 32, "check implementation");

std::array<uint8_t, BitStreamTraits<JPEGBitPumpTag>::MaxProcessBytes>
prefetch;
std::copy_n(input, prefetch.size(), prefetch.begin());

if (std::none_of(&prefetch[0], &prefetch[4],
[](uint8_t byte) { return byte == 0xFF; })) {
cache.push(getBE<uint32_t>(prefetch.data()), 32);
return 4;
}

size_type p = 0;
for (size_type i = 0; i < 4; ++i) {
const int c0 = prefetch[p];
++p;
cache.push(c0, 8);
if (c0 == 0xFF) {
const int c1 = prefetch[p];
++p;
if (c1 != 0) {

cache.fillLevel -= 8;
cache.cache &= ~((~0ULL) >> cache.fillLevel);
cache.fillLevel = 64;

return getRemainingSize();
}
}
}
return p;
}

template <>
inline BitPumpJPEG::size_type BitPumpJPEG::getStreamPosition() const {
return getInputPosition();
}

} 

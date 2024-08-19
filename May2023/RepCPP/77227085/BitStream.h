

#pragma once

#include "rawspeedconfig.h" 
#include "adt/Invariant.h"  
#include "common/Common.h"  
#include "io/Buffer.h"      
#include "io/ByteStream.h"  
#include "io/Endianness.h"  
#include "io/IOException.h" 
#include <algorithm>        
#include <array>            
#include <cstdint>          
#include <cstring>          

namespace rawspeed {

template <typename BIT_STREAM> struct BitStreamTraits final {
static constexpr bool canUseWithPrefixCodeDecoder = false;
};


struct BitStreamCacheBase {
uint64_t cache = 0;         
unsigned int fillLevel = 0; 

static constexpr unsigned Size = bitwidth<decltype(cache)>();

static constexpr unsigned MaxGetBits = bitwidth<uint32_t>();
};

struct BitStreamCacheLeftInRightOut : BitStreamCacheBase {
inline void push(uint64_t bits, uint32_t count) noexcept {
invariant(count + fillLevel <= bitwidth(cache));
cache |= bits << fillLevel;
fillLevel += count;
}

[[nodiscard]] inline uint32_t peek(uint32_t count) const noexcept {
return cache & ((1U << count) - 1U);
}

inline void skip(uint32_t count) noexcept {
cache >>= count;
fillLevel -= count;
}
};

struct BitStreamCacheRightInLeftOut : BitStreamCacheBase {
inline void push(uint64_t bits, uint32_t count) noexcept {
invariant(count + fillLevel <= Size);
invariant(count != 0);
const uint32_t vacantBits = BitStreamCacheBase::Size - fillLevel;
const uint32_t emptyBitsGap = vacantBits - count;
cache |= bits << emptyBitsGap;
fillLevel += count;
}

[[nodiscard]] inline uint32_t peek(uint32_t count) const noexcept {
return extractHighBits(cache, count,
BitStreamCacheBase::Size);
}

inline void skip(uint32_t count) noexcept {
fillLevel -= count;
cache <<= count;
}
};

template <typename Tag> struct BitStreamReplenisherBase {
using size_type = uint32_t;

const uint8_t* data;
size_type size;
unsigned pos = 0;

BitStreamReplenisherBase() = default;

explicit BitStreamReplenisherBase(Buffer input)
: data(input.getData(0, input.getSize())), size(input.getSize()) {
if (size < BitStreamTraits<Tag>::MaxProcessBytes)
ThrowIOE("Bit stream size is smaller than MaxProcessBytes");
}

std::array<uint8_t, BitStreamTraits<Tag>::MaxProcessBytes> tmp = {};
};

template <typename Tag>
struct BitStreamForwardSequentialReplenisher final
: public BitStreamReplenisherBase<Tag> {
using Base = BitStreamReplenisherBase<Tag>;

BitStreamForwardSequentialReplenisher() = default;

using Base::BitStreamReplenisherBase;

[[nodiscard]] inline typename Base::size_type getPos() const {
return Base::pos;
}
[[nodiscard]] inline typename Base::size_type getRemainingSize() const {
return Base::size - getPos();
}
inline void markNumBytesAsConsumed(typename Base::size_type numBytes) {
Base::pos += numBytes;
}

inline const uint8_t* getInput() {
#if !defined(DEBUG)
if (Base::pos + BitStreamTraits<Tag>::MaxProcessBytes <= Base::size)
return Base::data + Base::pos;
#endif


if (Base::pos > Base::size + 2 * BitStreamTraits<Tag>::MaxProcessBytes)
ThrowIOE("Buffer overflow read in BitStream");

Base::tmp.fill(0);

typename Base::size_type bytesRemaining =
(Base::pos < Base::size) ? Base::size - Base::pos : 0;
bytesRemaining = std::min<typename Base::size_type>(
BitStreamTraits<Tag>::MaxProcessBytes, bytesRemaining);

memcpy(Base::tmp.data(), Base::data + Base::pos, bytesRemaining);
return Base::tmp.data();
}
};

template <typename Tag, typename Cache,
typename Replenisher = BitStreamForwardSequentialReplenisher<Tag>>
class BitStream final {
Cache cache;

Replenisher replenisher;

using size_type = uint32_t;

size_type fillCache(const uint8_t* input);

public:
using tag = Tag;

BitStream() = default;

explicit BitStream(Buffer buf) : replenisher(buf) {}

explicit BitStream(ByteStream s)
: BitStream(s.getSubView(s.getPosition(), s.getRemainSize())) {}

inline void fill(uint32_t nbits = Cache::MaxGetBits) {
invariant(nbits <= Cache::MaxGetBits);

if (cache.fillLevel >= nbits)
return;

replenisher.markNumBytesAsConsumed(fillCache(replenisher.getInput()));
}

[[nodiscard]] inline size_type RAWSPEED_READONLY getInputPosition() const {
return replenisher.getPos();
}

[[nodiscard]] inline size_type getStreamPosition() const {
return getInputPosition() - (cache.fillLevel >> 3);
}

[[nodiscard]] inline size_type getRemainingSize() const {
return replenisher.getRemainingSize();
}

[[nodiscard]] inline size_type RAWSPEED_READONLY getFillLevel() const {
return cache.fillLevel;
}

inline uint32_t RAWSPEED_READONLY peekBitsNoFill(uint32_t nbits) {
invariant(nbits != 0);
invariant(nbits <= Cache::MaxGetBits);
invariant(nbits <= cache.fillLevel);
return cache.peek(nbits);
}

inline void skipBitsNoFill(uint32_t nbits) {
invariant(nbits <= Cache::MaxGetBits);
invariant(nbits <= cache.fillLevel);
cache.skip(nbits);
}

inline uint32_t getBitsNoFill(uint32_t nbits) {
uint32_t ret = peekBitsNoFill(nbits);
skipBitsNoFill(nbits);
return ret;
}

inline uint32_t peekBits(uint32_t nbits) {
fill(nbits);
return peekBitsNoFill(nbits);
}

inline uint32_t getBits(uint32_t nbits) {
fill(nbits);
return getBitsNoFill(nbits);
}

inline void skipBytes(uint32_t nbytes) {
uint32_t remainingBitsToSkip = 8 * nbytes;
for (; remainingBitsToSkip >= Cache::MaxGetBits;
remainingBitsToSkip -= Cache::MaxGetBits) {
fill(Cache::MaxGetBits);
skipBitsNoFill(Cache::MaxGetBits);
}
if (remainingBitsToSkip > 0) {
fill(remainingBitsToSkip);
skipBitsNoFill(remainingBitsToSkip);
}
}
};

} 

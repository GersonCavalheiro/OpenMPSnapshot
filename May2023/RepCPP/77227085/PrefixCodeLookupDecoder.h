

#pragma once

#include "codes/AbstractPrefixCodeDecoder.h" 
#include "codes/HuffmanCode.h"               
#include "decoders/RawDecoderException.h"    
#include "io/BitStream.h"                    
#include <cassert>                           
#include <cstdint>                           
#include <memory>                            
#include <tuple>                             
#include <utility>                           
#include <vector>                            



namespace rawspeed {

template <typename CodeTag>
class PrefixCodeLookupDecoder : public AbstractPrefixCodeDecoder<CodeTag> {
public:
using Tag = CodeTag;
using Base = AbstractPrefixCodeDecoder<CodeTag>;
using Traits = typename Base::Traits;

explicit PrefixCodeLookupDecoder(HuffmanCode<CodeTag>&& hc)
: Base(hc.operator rawspeed::PrefixCode<CodeTag>()) {}

PrefixCodeLookupDecoder(PrefixCode<CodeTag>) = delete;
PrefixCodeLookupDecoder(const PrefixCode<CodeTag>&) = delete;
PrefixCodeLookupDecoder(PrefixCode<CodeTag>&&) = delete;

protected:
std::vector<typename Traits::CodeTy> maxCodeOL;    
std::vector<typename Traits::CodeTy> codeOffsetOL; 

static constexpr auto MaxCodeValue =
std::numeric_limits<typename Traits::CodeTy>::max();

public:
void setup(bool fullDecode_, bool fixDNGBug16_) {
AbstractPrefixCodeDecoder<CodeTag>::setup(fullDecode_, fixDNGBug16_);

codeOffsetOL.resize(Base::maxCodeLength() + 1UL, MaxCodeValue);
maxCodeOL.resize(Base::maxCodeLength() + 1UL, MaxCodeValue);
for (unsigned int numCodesSoFar = 0, codeLen = 1;
codeLen <= Base::maxCodeLength(); codeLen++) {
if (!Base::code.nCodesPerLength[codeLen])
continue;
codeOffsetOL[codeLen] =
Base::code.symbols[numCodesSoFar].code - numCodesSoFar;
assert(codeOffsetOL[codeLen] != MaxCodeValue);
numCodesSoFar += Base::code.nCodesPerLength[codeLen];
maxCodeOL[codeLen] = Base::code.symbols[numCodesSoFar - 1].code;
}
}

template <typename BIT_STREAM>
inline typename Traits::CodeValueTy decodeCodeValue(BIT_STREAM& bs) const {
static_assert(
BitStreamTraits<typename BIT_STREAM::tag>::canUseWithPrefixCodeDecoder,
"This BitStream specialization is not marked as usable here");
invariant(!Base::fullDecode);
return decode<BIT_STREAM, false>(bs);
}

template <typename BIT_STREAM>
inline int decodeDifference(BIT_STREAM& bs) const {
static_assert(
BitStreamTraits<typename BIT_STREAM::tag>::canUseWithPrefixCodeDecoder,
"This BitStream specialization is not marked as usable here");
invariant(Base::fullDecode);
return decode<BIT_STREAM, true>(bs);
}

protected:
template <typename BIT_STREAM>
inline std::pair<typename Base::CodeSymbol, int >
finishReadingPartialSymbol(BIT_STREAM& bs,
typename Base::CodeSymbol partial) const {
static_assert(
BitStreamTraits<typename BIT_STREAM::tag>::canUseWithPrefixCodeDecoder,
"This BitStream specialization is not marked as usable here");
while (partial.code_len < Base::maxCodeLength() &&
(MaxCodeValue == maxCodeOL[partial.code_len] ||
partial.code > maxCodeOL[partial.code_len])) {
uint32_t temp = bs.getBitsNoFill(1);
partial.code = (partial.code << 1) | temp;
partial.code_len++;
}

if (partial.code_len > Base::maxCodeLength() ||
partial.code > maxCodeOL[partial.code_len])
ThrowRDE("bad Huffman code: %u (len: %u)", partial.code,
partial.code_len);

assert(MaxCodeValue != codeOffsetOL[partial.code_len]);
assert(partial.code >= codeOffsetOL[partial.code_len]);
unsigned codeIndex = partial.code - codeOffsetOL[partial.code_len];
assert(codeIndex < Base::code.codeValues.size());

typename Traits::CodeValueTy codeValue = Base::code.codeValues[codeIndex];
return {partial, codeValue};
}

template <typename BIT_STREAM>
inline std::pair<typename Base::CodeSymbol, int >
readSymbol(BIT_STREAM& bs) const {
typename Base::CodeSymbol partial;
partial.code_len = 0;
partial.code = 0;

return finishReadingPartialSymbol(bs, partial);
}

public:
template <typename BIT_STREAM, bool FULL_DECODE>
inline int decode(BIT_STREAM& bs) const {
invariant(FULL_DECODE == Base::fullDecode);
bs.fill(32);

typename Base::CodeSymbol symbol;
typename Traits::CodeValueTy codeValue;
std::tie(symbol, codeValue) = readSymbol(bs);

return Base::template processSymbol<BIT_STREAM, FULL_DECODE>(bs, symbol,
codeValue);
}
};

} 

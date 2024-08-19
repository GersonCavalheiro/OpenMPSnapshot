

#pragma once

#include "codes/AbstractPrefixCodeDecoder.h" 
#include "decoders/RawDecoderException.h"    
#include "io/BitStream.h"                    
#include <algorithm>                         
#include <cassert>                           
#include <cstdint>                           
#include <tuple>                             
#include <utility>                           
#include <vector>                            

namespace rawspeed {

template <typename CodeTag>
class PrefixCodeVectorDecoder : public AbstractPrefixCodeDecoder<CodeTag> {
public:
using Tag = CodeTag;
using Base = AbstractPrefixCodeDecoder<CodeTag>;
using Traits = typename Base::Traits;

using Base::Base;

private:
std::vector<unsigned int> extrCodeIdForLen; 

protected:
template <typename BIT_STREAM>
inline std::pair<typename Base::CodeSymbol, int >
finishReadingPartialSymbol(BIT_STREAM& bs,
typename Base::CodeSymbol partial) const {
static_assert(
BitStreamTraits<typename BIT_STREAM::tag>::canUseWithPrefixCodeDecoder,
"This BitStream specialization is not marked as usable here");

while (partial.code_len < Base::maxCodeLength()) {
const bool bit = bs.getBitsNoFill(1);

partial.code <<= 1;
partial.code |= bit;
partial.code_len++;

for (uint64_t codeId = extrCodeIdForLen[partial.code_len];
codeId < extrCodeIdForLen[1U + partial.code_len]; codeId++) {
const typename Base::CodeSymbol& symbol = Base::code.symbols[codeId];
invariant(partial.code_len == symbol.code_len);
if (symbol == partial) 
return {symbol, Base::code.codeValues[codeId]};
}
}

ThrowRDE("bad Huffman code: %u (len: %u)", partial.code, partial.code_len);
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
void setup(bool fullDecode_, bool fixDNGBug16_) {
AbstractPrefixCodeDecoder<CodeTag>::setup(fullDecode_, fixDNGBug16_);

extrCodeIdForLen.reserve(1U + Base::code.nCodesPerLength.size());
extrCodeIdForLen.resize(2); 
for (auto codeLen = 1UL; codeLen < Base::code.nCodesPerLength.size();
codeLen++) {
auto minCodeId = extrCodeIdForLen.back();
minCodeId += Base::code.nCodesPerLength[codeLen];
extrCodeIdForLen.emplace_back(minCodeId);
}
assert(extrCodeIdForLen.size() == 1U + Base::code.nCodesPerLength.size());
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

template <typename BIT_STREAM, bool FULL_DECODE>
inline int decode(BIT_STREAM& bs) const {
static_assert(
BitStreamTraits<typename BIT_STREAM::tag>::canUseWithPrefixCodeDecoder,
"This BitStream specialization is not marked as usable here");
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

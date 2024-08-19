

#pragma once

#include "codes/PrefixCodeLookupDecoder.h" 
#include "common/Common.h"                 
#include "decoders/RawDecoderException.h"  
#include "io/BitStream.h"                  
#include <cassert>                         
#include <cstddef>                         
#include <cstdint>                         
#include <memory>                          
#include <tuple>                           
#include <vector>                          



namespace rawspeed {

template <typename CodeTag, typename BackendPrefixCodeDecoder>
class PrefixCodeLUTDecoder final : public BackendPrefixCodeDecoder {
public:
using Tag = CodeTag;
using Base = BackendPrefixCodeDecoder;
using Traits = typename Base::Traits;

using Base::Base;

private:
#if 1
static constexpr unsigned PayloadShift = 9;
static constexpr unsigned FlagMask = 0x100;
static constexpr unsigned LenMask = 0xff;
static constexpr unsigned LookupDepth = 11;
using LUTEntryTy = int32_t;
using LUTUnsignedEntryTy = std::make_unsigned_t<LUTEntryTy>;
std::vector<LUTEntryTy> decodeLookup;
#else
static constexpr unsigned LookupDepth = 15;
static constexpr unsigned PayloadShift = 4;
static constexpr unsigned FlagMask = 0;
static constexpr unsigned LenMask = 0x0f;
std::vector<uint8_t> decodeLookup;
#endif

public:
void setup(bool fullDecode_, bool fixDNGBug16_) {
Base::setup(fullDecode_, fixDNGBug16_);

decodeLookup.resize(1 << LookupDepth);
for (size_t i = 0; i < Base::code.symbols.size(); i++) {
uint8_t code_l = Base::code.symbols[i].code_len;
if (code_l > static_cast<int>(LookupDepth))
break;

uint16_t ll = Base::code.symbols[i].code << (LookupDepth - code_l);
uint16_t ul = ll | ((1 << (LookupDepth - code_l)) - 1);
static_assert(Traits::MaxCodeValueLenghtBits <=
bitwidth<LUTEntryTy>() - PayloadShift);
LUTUnsignedEntryTy diff_l = Base::code.codeValues[i];
for (uint16_t c = ll; c <= ul; c++) {
if (!(c < decodeLookup.size()))
ThrowRDE("Corrupt Huffman");

if (!FlagMask || !Base::fullDecode || code_l > LookupDepth ||
(code_l + diff_l > LookupDepth && diff_l != 16)) {
invariant(!Base::fullDecode || diff_l > 0);
decodeLookup[c] = diff_l << PayloadShift | code_l;

if (!Base::fullDecode)
decodeLookup[c] |= FlagMask;
} else {
decodeLookup[c] = FlagMask | code_l;
if (diff_l != 16 || Base::fixDNGBug16)
decodeLookup[c] += diff_l;

if (diff_l) {
LUTUnsignedEntryTy diff;
if (diff_l != 16) {
diff = extractHighBits(c, code_l + diff_l,
LookupDepth);
diff &= ((1 << diff_l) - 1);
} else
diff = LUTUnsignedEntryTy(-32768);
decodeLookup[c] |= static_cast<LUTEntryTy>(
static_cast<LUTUnsignedEntryTy>(Base::extend(diff, diff_l))
<< PayloadShift);
}
}
}
}
}

template <typename BIT_STREAM>
inline __attribute__((always_inline)) int
decodeCodeValue(BIT_STREAM& bs) const {
static_assert(
BitStreamTraits<typename BIT_STREAM::tag>::canUseWithPrefixCodeDecoder,
"This BitStream specialization is not marked as usable here");
invariant(!Base::fullDecode);
return decode<BIT_STREAM, false>(bs);
}

template <typename BIT_STREAM>
inline __attribute__((always_inline)) int
decodeDifference(BIT_STREAM& bs) const {
static_assert(
BitStreamTraits<typename BIT_STREAM::tag>::canUseWithPrefixCodeDecoder,
"This BitStream specialization is not marked as usable here");
invariant(Base::fullDecode);
return decode<BIT_STREAM, true>(bs);
}

template <typename BIT_STREAM, bool FULL_DECODE>
inline __attribute__((always_inline)) int decode(BIT_STREAM& bs) const {
static_assert(
BitStreamTraits<typename BIT_STREAM::tag>::canUseWithPrefixCodeDecoder,
"This BitStream specialization is not marked as usable here");
invariant(FULL_DECODE == Base::fullDecode);
bs.fill(32);

typename Base::CodeSymbol partial;
partial.code_len = LookupDepth;
partial.code = bs.peekBitsNoFill(partial.code_len);

assert(partial.code < decodeLookup.size());
auto lutEntry = static_cast<unsigned>(decodeLookup[partial.code]);
int payload = static_cast<int>(lutEntry) >> PayloadShift;
int len = lutEntry & LenMask;

bs.skipBitsNoFill(len);

if (lutEntry & FlagMask)
return payload;

typename Traits::CodeValueTy codeValue;
if (lutEntry) {
partial.code_len = len;
codeValue = payload;
invariant(!FULL_DECODE || codeValue  > 0);
} else {
invariant(len == 0);
bs.skipBitsNoFill(partial.code_len);
std::tie(partial, codeValue) =
Base::finishReadingPartialSymbol(bs, partial);
}

return Base::template processSymbol<BIT_STREAM, FULL_DECODE>(bs, partial,
codeValue);
}
};

} 

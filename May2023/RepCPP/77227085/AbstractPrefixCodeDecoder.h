

#pragma once

#include "adt/Invariant.h"                
#include "codes/AbstractPrefixCode.h"     
#include "codes/PrefixCode.h"             
#include "decoders/RawDecoderException.h" 
#include <algorithm>                      
#include <cassert>                        
#include <cstddef>                        
#include <cstdint>                        
#include <functional>                     
#include <iterator>                       
#include <numeric>                        
#include <type_traits>                    
#include <vector>                         

namespace rawspeed {

template <typename CodeTag> class AbstractPrefixCodeDecoder {
public:
using Tag = CodeTag;
using Parent = AbstractPrefixCode<CodeTag>;
using CodeSymbol = typename AbstractPrefixCode<CodeTag>::CodeSymbol;
using Traits = typename AbstractPrefixCode<CodeTag>::Traits;

PrefixCode<CodeTag> code;

explicit AbstractPrefixCodeDecoder(PrefixCode<CodeTag> code_)
: code(std::move(code_)) {}

void verifyCodeValuesAsDiffLengths() const {
for (const auto cValue : code.Base::codeValues) {
if (cValue <= Traits::MaxDiffLength)
continue;
ThrowRDE("Corrupt Huffman code: difference length %u longer than %u",
cValue, Traits::MaxDiffLength);
}
assert(maxCodePlusDiffLength() <= 32U);
}

protected:
bool fullDecode = true;
bool fixDNGBug16 = false;

[[nodiscard]] inline size_t RAWSPEED_READONLY maxCodeLength() const {
return code.nCodesPerLength.size() - 1;
}

[[nodiscard]] inline size_t RAWSPEED_READONLY __attribute__((pure))
maxCodePlusDiffLength() const {
return maxCodeLength() + *(std::max_element(code.Base::codeValues.cbegin(),
code.Base::codeValues.cend()));
}

void setup(bool fullDecode_, bool fixDNGBug16_) {
invariant(!fullDecode_ || Traits::SupportsFullDecode);

this->fullDecode = fullDecode_;
this->fixDNGBug16 = fixDNGBug16_;

if (fullDecode) {
verifyCodeValuesAsDiffLengths();
}
}

public:
[[nodiscard]] bool isFullDecode() const { return fullDecode; }

bool operator==(const AbstractPrefixCodeDecoder& other) const {
return code.symbols == other.code.symbols &&
code.Base::codeValues == other.codeValues;
}

template <typename BIT_STREAM, bool FULL_DECODE>
inline int processSymbol(BIT_STREAM& bs, CodeSymbol symbol,
typename Traits::CodeValueTy codeValue) const {
invariant(symbol.code_len >= 0 &&
symbol.code_len <= Traits::MaxCodeLenghtBits);

if (!FULL_DECODE)
return codeValue;

int diff_l = codeValue;
invariant(diff_l >= 0 && diff_l <= 16);

if (diff_l == 16) {
if (fixDNGBug16)
bs.skipBitsNoFill(16);
return -32768;
}

invariant(symbol.code_len + diff_l <= 32);
return diff_l ? extend(bs.getBitsNoFill(diff_l), diff_l) : 0;
}

inline static int RAWSPEED_READNONE extend(uint32_t diff, uint32_t len) {
invariant(len > 0);
auto ret = static_cast<int32_t>(diff);
if ((diff & (1 << (len - 1))) == 0)
ret -= (1 << len) - 1;
return ret;
}
};

} 

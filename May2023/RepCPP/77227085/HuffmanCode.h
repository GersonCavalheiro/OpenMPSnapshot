

#pragma once

#include "adt/Array1DRef.h"               
#include "adt/Invariant.h"                
#include "codes/AbstractPrefixCode.h"     
#include "codes/PrefixCode.h"             
#include "decoders/RawDecoderException.h" 
#include "io/Buffer.h"                    
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

template <typename CodeTag>
class HuffmanCode final : public AbstractPrefixCode<CodeTag> {
public:
using Parent = AbstractPrefixCode<CodeTag>;
using CodeSymbol = typename AbstractPrefixCode<CodeTag>::CodeSymbol;
using Traits = typename AbstractPrefixCode<CodeTag>::Traits;

HuffmanCode() = default;

protected:
[[nodiscard]] inline size_t RAWSPEED_READONLY maxCodeLength() const {
return nCodesPerLength.size() - 1;
}


std::vector<unsigned int> nCodesPerLength; 

[[nodiscard]] inline unsigned int RAWSPEED_READONLY maxCodesCount() const {
return std::accumulate(nCodesPerLength.begin(), nCodesPerLength.end(), 0U);
}

public:
[[nodiscard]] std::vector<CodeSymbol> generateCodeSymbols() const {
std::vector<CodeSymbol> symbols;

assert(!nCodesPerLength.empty());
assert(maxCodesCount() > 0);

assert(this->codeValues.size() == maxCodesCount());

symbols.reserve(maxCodesCount());

uint32_t code = 0;
for (unsigned int l = 1; l <= maxCodeLength(); ++l) {
for (unsigned int i = 0; i < nCodesPerLength[l]; ++i) {
symbols.emplace_back(code, l);
code++;
}

code <<= 1;
}

assert(symbols.size() == maxCodesCount());

return symbols;
}

bool operator==(const HuffmanCode& other) const {
return nCodesPerLength == other.nCodesPerLength &&
this->codeValues == other.codeValues;
}

uint32_t setNCodesPerLength(Buffer data) {
invariant(data.getSize() == Traits::MaxCodeLenghtBits);

nCodesPerLength.resize((1 + Traits::MaxCodeLenghtBits), 0);
std::copy(data.begin(), data.end(), &nCodesPerLength[1]);
assert(nCodesPerLength[0] == 0);

while (!nCodesPerLength.empty() && nCodesPerLength.back() == 0)
nCodesPerLength.pop_back();

if (nCodesPerLength.empty())
ThrowRDE("Codes-per-length table is empty");

assert(nCodesPerLength.back() > 0);

const auto count = maxCodesCount();
invariant(count > 0);

if (count > Traits::MaxNumCodeValues)
ThrowRDE("Too big code-values table");

unsigned maxCodes = 2;

for (auto codeLen = 1UL; codeLen < nCodesPerLength.size(); codeLen++) {
const auto maxCodesInCurrLen = (1U << codeLen);
const auto nCodes = nCodesPerLength[codeLen];
if (nCodes > maxCodesInCurrLen) {
ThrowRDE("Corrupt Huffman. Can never have %u codes in %lu-bit len",
nCodes, codeLen);
}

if (nCodes > maxCodes) {
ThrowRDE(
"Corrupt Huffman. Can only fit %u out of %u codes in %lu-bit len",
maxCodes, nCodes, codeLen);
}

maxCodes -= nCodes;
maxCodes *= 2;
}

return count;
}

void setCodeValues(Array1DRef<const typename Traits::CodeValueTy> data) {
invariant(data.size() <= Traits::MaxNumCodeValues);
invariant((unsigned)data.size() == maxCodesCount());

this->codeValues.clear();
this->codeValues.reserve(maxCodesCount());
std::copy(data.begin(), data.end(), std::back_inserter(this->codeValues));
assert(this->codeValues.size() == maxCodesCount());

for (const auto& cValue : this->codeValues) {
if (cValue <= Traits::MaxCodeValue)
continue;
ThrowRDE("Corrupt Huffman code: code value %u is larger than maximum %u",
cValue, Traits::MaxCodeValue);
}
}

explicit operator PrefixCode<CodeTag>() {
std::vector<CodeSymbol> symbols = generateCodeSymbols();
return {std::move(symbols), std::move(Parent::codeValues)};
}
};

} 

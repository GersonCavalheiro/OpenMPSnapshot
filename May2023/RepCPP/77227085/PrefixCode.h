

#pragma once

#include "codes/AbstractPrefixCode.h"     
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

template <typename CodeTag>
class PrefixCode : public AbstractPrefixCode<CodeTag> {
public:
using Base = AbstractPrefixCode<CodeTag>;
using Traits = typename Base::Traits;
using CodeSymbol = typename Base::CodeSymbol;
using CodeValueTy = typename Traits::CodeValueTy;

std::vector<unsigned> nCodesPerLength;

std::vector<CodeSymbol> symbols;

PrefixCode(std::vector<CodeSymbol> symbols_,
std::vector<CodeValueTy> codeValues_)
: Base(std::move(codeValues_)), symbols(std::move(symbols_)) {
if (symbols.empty() || Base::codeValues.empty() ||
symbols.size() != Base::codeValues.size())
ThrowRDE("Malformed code");

nCodesPerLength.resize(1 + Traits::MaxCodeLenghtBits);
for (const CodeSymbol& s : symbols) {
assert(s.code_len > 0 && s.code_len <= Traits::MaxCodeLenghtBits);
++nCodesPerLength[s.code_len];
}
while (nCodesPerLength.back() == 0)
nCodesPerLength.pop_back();
assert(nCodesPerLength.size() > 1);

verifyCodeSymbols();
}

private:
void verifyCodeSymbols() {
unsigned maxCodes = 2;
for (auto codeLen = 1UL; codeLen < nCodesPerLength.size(); codeLen++) {
const unsigned nCodes = nCodesPerLength[codeLen];
if (nCodes > maxCodes)
ThrowRDE("Too many codes of of length %lu.", codeLen);
maxCodes -= nCodes;
maxCodes *= 2;
}

if (std::adjacent_find(
symbols.cbegin(), symbols.cend(),
[](const CodeSymbol& lhs, const CodeSymbol& rhs) -> bool {
return !std::less_equal<>()(lhs.code_len, rhs.code_len);
}) != symbols.cend())
ThrowRDE("Code symbols are not globally ordered");

for (auto sId = 0UL; sId < symbols.size(); sId++) {
for (auto pId = 0UL; pId < sId; pId++)
if (CodeSymbol::HaveCommonPrefix(symbols[sId], symbols[pId]))
ThrowRDE("Not prefix codes!");
}
}
};

} 

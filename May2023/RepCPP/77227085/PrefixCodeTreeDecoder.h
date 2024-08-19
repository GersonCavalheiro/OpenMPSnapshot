

#pragma once

#include "codes/AbstractPrefixCodeDecoder.h" 
#include "codes/BinaryPrefixTree.h"          
#include "decoders/RawDecoderException.h"    
#include "io/BitStream.h"                    
#include <algorithm>                         
#include <cassert>                           
#include <initializer_list>                  
#include <iterator>                          
#include <memory>                            
#include <optional>                          
#include <tuple>                             
#include <utility>                           
#include <vector>                            

namespace rawspeed {

template <typename CodeTag>
class PrefixCodeTreeDecoder : public AbstractPrefixCodeDecoder<CodeTag> {
public:
using Tag = CodeTag;
using Base = AbstractPrefixCodeDecoder<CodeTag>;
using Traits = typename Base::Traits;

using Base::Base;

private:
BinaryPrefixTree<CodeTag> tree;

protected:
template <typename BIT_STREAM>
inline std::pair<typename Base::CodeSymbol, int >
finishReadingPartialSymbol(BIT_STREAM& bs,
typename Base::CodeSymbol initialPartial) const {
typename Base::CodeSymbol partial;
partial.code = 0;
partial.code_len = 0;

const auto* top = &(tree.root->getAsBranch());

auto walkBinaryTree = [&partial, &top](bool bit)
-> std::optional<
std::pair<typename Base::CodeSymbol, int >> {
partial.code <<= 1;
partial.code |= bit;
partial.code_len++;

const auto& newNode = top->buds[bit];

if (!newNode) {
ThrowRDE("bad Huffman code: %u (len: %u)", partial.code,
partial.code_len);
}

if (static_cast<typename decltype(tree)::Node::Type>(*newNode) ==
decltype(tree)::Node::Type::Leaf) {
return {{partial, newNode->getAsLeaf().value}};
}

top = &(newNode->getAsBranch());
return std::nullopt;
};

for (unsigned bit : initialPartial.getBitsMSB()) {
if (auto sym = walkBinaryTree(bit))
return *sym;
}

while (true) {
invariant(partial.code_len <= Traits::MaxCodeLenghtBits);

const bool bit = bs.getBitsNoFill(1);

if (auto sym = walkBinaryTree(bit))
return *sym;
}

__builtin_unreachable();
}

template <typename BIT_STREAM>
inline std::pair<typename Base::CodeSymbol, int >
readSymbol(BIT_STREAM& bs) const {
static_assert(
BitStreamTraits<typename BIT_STREAM::tag>::canUseWithPrefixCodeDecoder,
"This BitStream specialization is not marked as usable here");

typename Base::CodeSymbol partial;
partial.code_len = 0;
partial.code = 0;

return finishReadingPartialSymbol(bs, partial);
}

public:
void setup(bool fullDecode_, bool fixDNGBug16_) {
AbstractPrefixCodeDecoder<CodeTag>::setup(fullDecode_, fixDNGBug16_);

assert(Base::code.symbols.size() == Base::code.codeValues.size());
for (unsigned codeIndex = 0; codeIndex != Base::code.symbols.size();
++codeIndex)
tree.add(Base::code.symbols[codeIndex], Base::code.codeValues[codeIndex]);
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

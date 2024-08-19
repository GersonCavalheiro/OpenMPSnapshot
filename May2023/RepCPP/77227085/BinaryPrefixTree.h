

#pragma once

#include "adt/Invariant.h"            
#include "codes/AbstractPrefixCode.h" 
#include <cassert>                    
#include <functional>                 
#include <initializer_list>           
#include <memory>                     

namespace rawspeed {

template <typename CodeTag>
class BinaryPrefixTree final  {
public:
using Traits = typename AbstractPrefixCode<CodeTag>::Traits;
using CodeSymbol = typename AbstractPrefixCode<CodeTag>::CodeSymbol;
using CodeTy = typename Traits::CodeTy;

struct Branch;
struct Leaf;

struct Node {
enum class Type { Branch, Leaf };

explicit virtual operator Type() const = 0;

Branch& getAsBranch() {
assert(Node::Type::Branch == static_cast<Node::Type>(*this));
return static_cast<Branch&>(*this);
}

Leaf& getAsLeaf() {
assert(Node::Type::Leaf == static_cast<Node::Type>(*this));
return static_cast<Leaf&>(*this);
}

virtual ~Node() = default;
};

struct Branch final : public Node {
explicit operator typename Node::Type() const override {
return Node::Type::Branch;
}

std::array<std::unique_ptr<Node>, 2> buds;
};

struct Leaf final : public Node {
explicit operator typename Node::Type() const override {
return Node::Type::Leaf;
}

CodeTy value;

Leaf() = default;

explicit Leaf(CodeTy value_) : value(value_) {}
};

void add(CodeSymbol symbol, CodeTy value);

std::unique_ptr<Node> root;
};

template <typename CodeTag>
void BinaryPrefixTree<CodeTag>::add(const CodeSymbol symbol, CodeTy value) {
invariant(symbol.code_len > 0);
invariant(symbol.code_len <= Traits::MaxCodeLenghtBits);

CodeSymbol partial;
partial.code = 0;
partial.code_len = 0;

std::reference_wrapper<std::unique_ptr<Node>> newBud = root;
for (unsigned bit : symbol.getBitsMSB()) {
++partial.code_len;
partial.code = (partial.code << 1) | bit;
std::unique_ptr<Node>& bud = newBud;
if (!bud)
bud = std::make_unique<Branch>();
newBud = bud->getAsBranch().buds[bit];
}
invariant(partial == symbol && "Failed to interpret symbol as bit sequence.");

std::unique_ptr<Node>& bud = newBud;
assert(!bud && "This Node should be vacant!");

bud = std::make_unique<Leaf>(value);
}

} 

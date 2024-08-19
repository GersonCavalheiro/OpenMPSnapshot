

#pragma once

#include "adt/Invariant.h" 
#include "common/Common.h" 
#include <iterator>        
#include <utility>         

namespace rawspeed {

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
struct BitMSBIterator {
T bitsPat;
int bitIdx;

using iterator_category = std::input_iterator_tag;
using difference_type = std::ptrdiff_t;
using value_type = bool;
using pointer = const value_type*;   
using reference = const value_type&; 

BitMSBIterator(T bitsPat_, int bitIdx_) : bitsPat(bitsPat_), bitIdx(bitIdx_) {
invariant(bitIdx < static_cast<int>(bitwidth<T>()) && bitIdx >= -1);
}

value_type operator*() const {
invariant(static_cast<unsigned>(bitIdx) < bitwidth<T>() &&
"Iterator overflow");
return (bitsPat >> bitIdx) & 0b1;
}
BitMSBIterator& operator++() {
--bitIdx; 
invariant(bitIdx >= -1);
return *this;
}
friend inline bool operator==(const BitMSBIterator& a,
const BitMSBIterator& b) {
invariant(a.bitsPat == b.bitsPat && "Comparing unrelated iterators.");
return a.bitIdx == b.bitIdx;
}
friend bool operator!=(const BitMSBIterator& a, const BitMSBIterator& b) {
return !(a == b);
}
};

} 

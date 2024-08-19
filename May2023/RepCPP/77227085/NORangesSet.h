

#pragma once

#include "adt/Range.h" 
#include <algorithm>   
#include <cassert>     
#include <cstddef>     
#include <iterator>    
#include <set>         
#include <utility>     

namespace rawspeed {

template <typename T> class NORangesSet final {
std::set<T> elts;

[[nodiscard]] bool
rangeIsOverlappingExistingElementOfSortedSet(const T& newElt) const {
if (elts.empty())
return false;

auto p =
std::partition_point(elts.begin(), elts.end(),
[newElt](const T& elt) { return elt < newElt; });

if (p != elts.end() && RangesOverlap(newElt, *p))
return true;

if (p == elts.begin())
return false;

auto prevBeforeP = std::prev(p);
return RangesOverlap(newElt, *prevBeforeP);
}

public:
bool insert(const T& newElt) {
if (rangeIsOverlappingExistingElementOfSortedSet(newElt))
return false;

auto i = elts.insert(newElt);
assert(i.second && "Did not insert after all?");
(void)i;
return true;
}

[[nodiscard]] std::size_t size() const { return elts.size(); }
};

} 

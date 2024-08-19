

#pragma once

#include "common/Common.h" 
#include <algorithm>       
#include <cassert>         
#include <functional>      
#include <iterator>        
#include <type_traits>     
#include <vector>          

namespace rawspeed {

template <typename T, int TableBitWidth> class SimpleLUT final {
public:
using value_type = T;

SimpleLUT() = default;

private:
std::vector<value_type> table;

public:
template <
typename F,
typename = std::enable_if_t<!std::is_same_v<
SimpleLUT,
typename std::remove_cv_t<typename std::remove_reference_t<F>>>>,
typename = std::enable_if<std::is_convertible_v<
F, std::function<value_type(typename decltype(table)::size_type,
typename decltype(table)::size_type)>>>>
explicit SimpleLUT(F&& f) {
const auto fullTableSize = 1U << TableBitWidth;
table.reserve(fullTableSize);
std::generate_n(std::back_inserter(table), fullTableSize,
[&f, table_ = &table]() {
const auto i = table_->size();
return f(i, fullTableSize);
});
assert(table.size() == fullTableSize);
}

inline value_type operator[](int x) const {
unsigned clampedX = clampBits(x, TableBitWidth);
return table[clampedX];
}
};

} 

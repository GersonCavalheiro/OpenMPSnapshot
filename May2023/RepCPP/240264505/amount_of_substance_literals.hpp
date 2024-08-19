

#ifndef LBT_UNIT_AMOUNT_OF_SUBSTANCES_LITERALS
#define LBT_UNIT_AMOUNT_OF_SUBSTANCES_LITERALS
#pragma once

#include "amount_of_substance.hpp"


namespace lbt {
namespace literals {


constexpr lbt::unit::AmountOfSubstance operator "" _mol(long double const m) noexcept {
return m*lbt::unit::AmountOfSubstance{1.0};
}

constexpr lbt::unit::AmountOfSubstance operator "" _kmol(long double const k) noexcept {
return k*lbt::unit::AmountOfSubstance{1000.0};
}

}
}

#endif 

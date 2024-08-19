

#ifndef LBT_UNIT_AMOUNT_OF_SUBSTANCE
#define LBT_UNIT_AMOUNT_OF_SUBSTANCE
#pragma once

#include "unit_base.hpp"


namespace lbt {
namespace unit {


class AmountOfSubstance : public lbt::unit::detail::UnitBase<AmountOfSubstance> {
public:

explicit constexpr AmountOfSubstance(long double const value = 0.0) noexcept
: UnitBase{value} {
return;
}
AmountOfSubstance(AmountOfSubstance const&) = default;
AmountOfSubstance& operator= (AmountOfSubstance const&) = default;
AmountOfSubstance(AmountOfSubstance&&) = default;
AmountOfSubstance& operator= (AmountOfSubstance&&) = default;
};

}
}

#endif 

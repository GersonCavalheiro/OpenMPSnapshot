

#ifndef LBT_UNIT_MOLAR_MASS
#define LBT_UNIT_MOLAR_MASS
#pragma once

#include "unit_base.hpp"


namespace lbt {
namespace unit {


class MolarMass : public lbt::unit::detail::UnitBase<MolarMass> {
public:

explicit constexpr MolarMass(long double const value = 0.0) noexcept
: UnitBase{value} {
return;
}
MolarMass(MolarMass const&) = default;
MolarMass& operator= (MolarMass const&) = default;
MolarMass(MolarMass&&) = default;
MolarMass& operator= (MolarMass&&) = default;
};

}
}

#endif 



#ifndef LBT_UNIT_DENSITY
#define LBT_UNIT_DENSITY
#pragma once

#include "unit_base.hpp"


namespace lbt {
namespace unit {


class Density : public lbt::unit::detail::UnitBase<Density> {
public:

explicit constexpr Density(long double const value = 0.0) noexcept
: UnitBase{value} {
return;
}
Density(Density const&) = default;
Density& operator= (Density const&) = default;
Density(Density&&) = default;
Density& operator= (Density&&) = default;
};

}
}

#endif 

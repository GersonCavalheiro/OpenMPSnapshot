

#ifndef LBT_UNIT_MASS
#define LBT_UNIT_MASS
#pragma once

#include "unit_base.hpp"


namespace lbt {
namespace unit {


class Mass : public lbt::unit::detail::UnitBase<Mass> {
public:

explicit constexpr Mass(long double const value = 0.0) noexcept
: UnitBase{value} {
return;
}
Mass(Mass const&) = default;
Mass& operator= (Mass const&) = default;
Mass(Mass&&) = default;
Mass& operator= (Mass&&) = default;
};

}
}

#endif 

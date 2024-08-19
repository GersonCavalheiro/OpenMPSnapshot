

#ifndef LBT_UNIT_VELOCITY
#define LBT_UNIT_VELOCITY
#pragma once

#include "unit_base.hpp"


namespace lbt {
namespace unit {


class Velocity : public lbt::unit::detail::UnitBase<Velocity> {
public:

explicit constexpr Velocity(long double const value = 0.0) noexcept
: UnitBase{value} {
return;
}
Velocity(Velocity const&) = default;
Velocity& operator= (Velocity const&) = default;
Velocity(Velocity&&) = default;
Velocity& operator= (Velocity&&) = default;
};

}
}

#endif 

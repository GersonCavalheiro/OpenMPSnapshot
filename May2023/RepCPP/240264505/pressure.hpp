

#ifndef LBT_UNIT_PRESSURE
#define LBT_UNIT_PRESSURE
#pragma once

#include "unit_base.hpp"


namespace lbt {
namespace unit {


class Pressure : public lbt::unit::detail::UnitBase<Pressure> {
public:

explicit constexpr Pressure(long double const value = 101325) noexcept
: UnitBase{value} {
return;
}
Pressure(Pressure const&) = default;
Pressure& operator= (Pressure const&) = default;
Pressure(Pressure&&) = default;
Pressure& operator= (Pressure&&) = default;
};

}
}

#endif 

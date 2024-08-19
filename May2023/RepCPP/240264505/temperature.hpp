

#ifndef LBT_UNIT_TEMPERATURE
#define LBT_UNIT_TEMPERATURE
#pragma once

#include "unit_base.hpp"


namespace lbt {
namespace unit {


class Temperature : public lbt::unit::detail::UnitBase<Temperature> {
public:

explicit constexpr Temperature(long double const value = 273.15) noexcept
: UnitBase{value} {
return;
}
Temperature(Temperature const&) = default;
Temperature& operator= (Temperature const&) = default;
Temperature(Temperature&&) = default;
Temperature& operator= (Temperature&&) = default;
};

}
}

#endif 

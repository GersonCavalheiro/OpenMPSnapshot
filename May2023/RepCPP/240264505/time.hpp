

#ifndef LBT_UNIT_TIME
#define LBT_UNIT_TIME
#pragma once

#include "unit_base.hpp"


namespace lbt {
namespace unit {


class Time : public lbt::unit::detail::UnitBase<Time> {
public:

explicit constexpr Time(long double const value = 0.0) noexcept
: UnitBase{value} {
return;
}
Time(Time const&) = default;
Time& operator= (Time const&) = default;
Time(Time&&) = default;
Time& operator= (Time&&) = default;
};
using Duration = Time;

}
}

#endif 

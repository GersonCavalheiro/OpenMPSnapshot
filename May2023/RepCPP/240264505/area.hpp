

#ifndef LBT_UNIT_AREA
#define LBT_UNIT_AREA
#pragma once

#include "unit_base.hpp"


namespace lbt {
namespace unit {


class Area : public lbt::unit::detail::UnitBase<Area> {
public:

constexpr Area(long double const value = 0.0) noexcept
: UnitBase{value} {
return;
}
Area(Area const&) = default;
Area& operator= (Area const&) = default;
Area(Area&&) = default;
Area& operator= (Area&&) = default;
};

}
}

#endif 

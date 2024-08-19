

#ifndef LBT_UNIT_VOLUME
#define LBT_UNIT_VOLUME
#pragma once

#include "unit_base.hpp"


namespace lbt {
namespace unit {


class Volume : public lbt::unit::detail::UnitBase<Volume> {
public:

explicit constexpr Volume(long double const value = 0.0) noexcept
: UnitBase{value} {
return;
}
Volume(Volume const&) = default;
Volume& operator= (Volume const&) = default;
Volume(Volume&&) = default;
Volume& operator= (Volume&&) = default;
};

}
}

#endif 

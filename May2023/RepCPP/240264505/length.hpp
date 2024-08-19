

#ifndef LBT_UNIT_LENGTH
#define LBT_UNIT_LENGTH
#pragma once

#include "unit_base.hpp"


namespace lbt {
namespace unit {


class Length : public lbt::unit::detail::UnitBase<Length> {
public:

explicit constexpr Length(long double const value = 0.0) noexcept
: UnitBase{value} {
return;
}
Length(Length const&) = default;
Length& operator= (Length const&) = default;
Length(Length&&) = default;
Length& operator= (Length&&) = default;
};
using Distance = Length;

}
}

#endif 



#ifndef LBT_UNIT_VELOCITY_LITERALS
#define LBT_UNIT_VELOCITY_LITERALS
#pragma once

#include "length.hpp"
#include "length_literals.hpp"
#include "operators.hpp"
#include "time.hpp"
#include "time_literals.hpp"
#include "velocity.hpp"


namespace lbt {
namespace literals {


constexpr lbt::unit::Velocity operator "" _kmph(long double const k) noexcept {
return k*lbt::unit::Velocity{1.0_km/1.0_h};
}

constexpr lbt::unit::Velocity operator "" _mps(long double const m) noexcept {
return m*lbt::unit::Velocity{1.0_m/1.0_s};
}

constexpr lbt::unit::Velocity operator "" _cmps(long double const c) noexcept {
return c*lbt::unit::Velocity{1.0_cm/1.0_s};
}

constexpr lbt::unit::Velocity operator "" _mmps(long double const m) noexcept {
return m*lbt::unit::Velocity{1.0_mm/1.0_s};
}

}
}

#endif 

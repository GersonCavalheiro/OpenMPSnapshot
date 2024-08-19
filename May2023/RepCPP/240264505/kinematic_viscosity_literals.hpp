

#ifndef LBT_UNIT_KINEMATIC_VISCOSITY_LITERALS
#define LBT_UNIT_KINEMATIC_VISCOSITY_LITERALS
#pragma once

#include "kinematic_viscosity.hpp"
#include "length.hpp"
#include "length_literals.hpp"
#include "operators.hpp"
#include "time.hpp"
#include "time_literals.hpp"


namespace lbt {
namespace literals {


constexpr lbt::unit::KinematicViscosity operator "" _m2ps(long double const m) noexcept {
return m*lbt::unit::KinematicViscosity{1.0_m*1.0_m/1.0_s};
}

constexpr lbt::unit::KinematicViscosity operator "" _St(long double const s) noexcept {
return s*lbt::unit::KinematicViscosity{1.0e-4};
}

constexpr lbt::unit::KinematicViscosity operator "" _cSt(long double const c) noexcept {
return c*lbt::unit::KinematicViscosity{1.0e-6};
}

}
}

#endif 

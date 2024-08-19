

#ifndef LBT_UNIT_DYNAMIC_VISCOSITY_LITERALS
#define LBT_UNIT_DYNAMIC_VISCOSITY_LITERALS
#pragma once

#include "dynamic_viscosity.hpp"
#include "operators.hpp"
#include "pressure.hpp"
#include "pressure_literals.hpp"
#include "time.hpp"
#include "time_literals.hpp"


namespace lbt {
namespace literals {


constexpr lbt::unit::DynamicViscosity operator "" _Pas(long double const c) noexcept {
return c*lbt::unit::DynamicViscosity{1.0};
}

constexpr lbt::unit::DynamicViscosity operator "" _mPas(long double const c) noexcept {
return c*lbt::unit::DynamicViscosity{1.0_mPa*1.0_s};
}

constexpr lbt::unit::DynamicViscosity operator "" _uPas(long double const c) noexcept {
return c*lbt::unit::DynamicViscosity{1.0_uPa*1.0_s};
}

constexpr lbt::unit::DynamicViscosity operator "" _P(long double const c) noexcept {
return c*lbt::unit::DynamicViscosity{1.0e-1};
}

constexpr lbt::unit::DynamicViscosity operator "" _cP(long double const c) noexcept {
return c*lbt::unit::DynamicViscosity{1.0_mPa*1.0_s};
}

}
}

#endif 

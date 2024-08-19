

#ifndef LBT_UNIT_PRESSURE_LITERALS
#define LBT_UNIT_PRESSURE_LITERALS
#pragma once

#include "pressure.hpp"


namespace lbt {
namespace literals {


constexpr lbt::unit::Pressure operator "" _Pa(long double const p) noexcept {
return p*lbt::unit::Pressure{1.0};
}

constexpr lbt::unit::Pressure operator "" _GPa(long double const g) noexcept {
return g*lbt::unit::Pressure{1.0e+9};
}

constexpr lbt::unit::Pressure operator "" _mPa(long double const m) noexcept {
return m*lbt::unit::Pressure{1.0e-3};
}

constexpr lbt::unit::Pressure operator "" _uPa(long double const m) noexcept {
return m*lbt::unit::Pressure{1.0e-6};
}

constexpr lbt::unit::Pressure operator "" _hPa(long double const h) noexcept {
return h*lbt::unit::Pressure{100.0};
}

constexpr lbt::unit::Pressure operator "" _bar(long double const b) noexcept {
return b*lbt::unit::Pressure{1.0e+5};
}

constexpr lbt::unit::Pressure operator "" _atm(long double const a) noexcept {
return a*lbt::unit::Pressure{101325};
}

}
}

#endif 

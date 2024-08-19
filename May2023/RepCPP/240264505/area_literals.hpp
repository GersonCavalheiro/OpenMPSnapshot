

#ifndef LBT_UNIT_AREA_LITERALS
#define LBT_UNIT_AREA_LITERALS
#pragma once

#include "area.hpp"
#include "length.hpp"
#include "length_literals.hpp"
#include "operators.hpp"


namespace lbt {
namespace literals {


constexpr lbt::unit::Area operator "" _km2(long double const k) noexcept {
return k*lbt::unit::Area{1.0_km*1.0_km};
}

constexpr lbt::unit::Area operator "" _m2(long double const m) noexcept {
return m*lbt::unit::Area{1.0_m*1.0_m};
}

constexpr lbt::unit::Area operator "" _cm2(long double const c) noexcept {
return c*lbt::unit::Area{1.0_cm*1.0_cm};
}

constexpr lbt::unit::Area operator "" _mm2(long double const m) noexcept {
return m*lbt::unit::Area{1.0_mm*1.0_mm};
}

}
}

#endif 



#ifndef LBT_UNIT_VOLUME_LITERALS
#define LBT_UNIT_VOLUME_LITERALS
#pragma once

#include "length.hpp"
#include "length_literals.hpp"
#include "operators.hpp"
#include "volume.hpp"


namespace lbt {
namespace literals {


constexpr lbt::unit::Volume operator "" _km3(long double const k) noexcept {
return k*lbt::unit::Volume{1.0_km*1.0_km*1.0_km};
}

constexpr lbt::unit::Volume operator "" _m3(long double const m) noexcept {
return m*lbt::unit::Volume{1.0_m*1.0_m*1.0_m};
}

constexpr lbt::unit::Volume operator "" _cm3(long double const c) noexcept {
return c*lbt::unit::Volume{1.0_cm*1.0_cm*1.0_cm};
}

constexpr lbt::unit::Volume operator "" _mm3(long double const m) noexcept {
return m*lbt::unit::Volume{1.0_mm*1.0_mm*1.0_mm};
}

}
}

#endif 

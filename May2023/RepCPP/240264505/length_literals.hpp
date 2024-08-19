

#ifndef LBT_UNIT_LENGTH_LITERALS
#define LBT_UNIT_LENGTH_LITERALS
#pragma once

#include "length.hpp"


namespace lbt {
namespace literals {


constexpr lbt::unit::Length operator "" _km(long double const k) noexcept {
return k*lbt::unit::Length{1.0e3};
}

constexpr lbt::unit::Length operator "" _m(long double const m) noexcept {
return m*lbt::unit::Length{1.0};
}

constexpr lbt::unit::Length operator "" _dm(long double const d) noexcept {
return d*lbt::unit::Length{1.0e-1};
}

constexpr lbt::unit::Length operator "" _cm(long double const c) noexcept {
return c*lbt::unit::Length{1.0e-2};
}

constexpr lbt::unit::Length operator "" _mm(long double const m) noexcept {
return m*lbt::unit::Length{1.0e-3};
}

constexpr lbt::unit::Length operator "" _um(long double const u) noexcept {
return u*lbt::unit::Length{1.0e-6};
}

constexpr lbt::unit::Length operator "" _pm(long double const u) noexcept {
return u*lbt::unit::Length{1.0e-12};
}

}
}

#endif 

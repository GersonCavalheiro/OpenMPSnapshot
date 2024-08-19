

#ifndef LBT_UNIT_TIME_LITERALS
#define LBT_UNIT_TIME_LITERALS
#pragma once

#include "time.hpp"


namespace lbt {
namespace literals {


constexpr lbt::unit::Time operator "" _d(long double const d) noexcept {
return d*lbt::unit::Time{86400.0};
}

constexpr lbt::unit::Time operator "" _h(long double const h) noexcept {
return h*lbt::unit::Time{3600.0};
}

constexpr lbt::unit::Time operator "" _min(long double const m) noexcept {
return m*lbt::unit::Time{60.0};
}

constexpr lbt::unit::Time operator "" _s(long double const s) noexcept {
return s*lbt::unit::Time{1.0};
}

constexpr lbt::unit::Time operator "" _ms(long double const m) noexcept {
return m*lbt::unit::Time{1.0e-3};
}

constexpr lbt::unit::Time operator "" _us(long double const u) noexcept {
return u*lbt::unit::Time{1.0e-6};
}

}
}

#endif 



#ifndef LBT_UNIT_MASS_LITERALS
#define LBT_UNIT_MASS_LITERALS
#pragma once

#include "mass.hpp"


namespace lbt {
namespace literals {


constexpr lbt::unit::Mass operator "" _t(long double const t) noexcept {
return t*lbt::unit::Mass{1000.0};
}

constexpr lbt::unit::Mass operator "" _kg(long double const k) noexcept {
return k*lbt::unit::Mass{1.0};
}

constexpr lbt::unit::Mass operator "" _g(long double const g) noexcept {
return g*lbt::unit::Mass{1.0e-3};
}

}
}

#endif 



#ifndef LBT_UNIT_TEMPERATURE_LITERALS
#define LBT_UNIT_TEMPERATURE_LITERALS
#pragma once

#include "temperature.hpp"


namespace lbt {
namespace literals {


constexpr lbt::unit::Temperature operator "" _K(long double const t) noexcept {
return lbt::unit::Temperature{t};
}

constexpr lbt::unit::Temperature operator "" _deg(long double const t) noexcept {
return lbt::unit::Temperature{t + 273.15};
}

}
}

#endif 

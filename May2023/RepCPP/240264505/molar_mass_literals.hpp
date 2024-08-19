

#ifndef LBT_UNIT_MOLAR_MASS_LITERALS
#define LBT_UNIT_MOLAR_MASS_LITERALS
#pragma once

#include "amount_of_substance.hpp"
#include "amount_of_substance_literals.hpp"
#include "mass.hpp"
#include "mass_literals.hpp"
#include "molar_mass.hpp"
#include "operators.hpp"


namespace lbt {
namespace literals {


constexpr lbt::unit::MolarMass operator "" _gpmol(long double const g) noexcept {
return g*lbt::unit::MolarMass{1.0_g/1.0_mol};
}

constexpr lbt::unit::MolarMass operator "" _kgpmol(long double const k) noexcept {
return k*lbt::unit::MolarMass{1.0};
}

}
}

#endif 

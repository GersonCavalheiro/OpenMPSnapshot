

#ifndef LBT_MATERIAL_CARBON_DIOXIDE
#define LBT_MATERIAL_CARBON_DIOXIDE
#pragma once

#include "../../unit/literals.hpp"
#include "../../unit/units.hpp"
#include "ideal_gas.hpp"


namespace lbt {
namespace material {

namespace physical_constants {
using namespace lbt::literals;

class CarbonDioxide {
public:
static constexpr auto molecular_weight = 44.01_gpmol;
static constexpr auto c = 240.0_K;
static constexpr auto t_0 = 293.15_K;
static constexpr auto mu_0 = 14.8_uPas;

protected:
CarbonDioxide() = default;
CarbonDioxide(CarbonDioxide const&) = default;
CarbonDioxide& operator= (CarbonDioxide const&) = default;
CarbonDioxide(CarbonDioxide&&) = default;
CarbonDioxide& operator= (CarbonDioxide&&) = default;
};

}

using CarbonDioxide = IdealGas<physical_constants::CarbonDioxide>;

}
}

#endif 

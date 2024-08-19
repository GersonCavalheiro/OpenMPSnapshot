

#ifndef LBT_MATERIAL_IDEAL_GAS
#define LBT_MATERIAL_IDEAL_GAS
#pragma once

#include "../../constexpr_math/constexpr_math.hpp"
#include "../../unit/literals.hpp"
#include "../../unit/units.hpp"


namespace lbt {
namespace material {




template <typename T>
class IdealGas {
public:

static constexpr lbt::unit::Density equationOfState(lbt::unit::Temperature const t, 
lbt::unit::Pressure const p) noexcept {
return lbt::unit::Density{p.get()/(specific_gas_constant*t.get())};
};
static constexpr lbt::unit::Temperature equationOfState(lbt::unit::Density const rho, 
lbt::unit::Pressure const p) noexcept {
return lbt::unit::Temperature{p.get()/(specific_gas_constant*rho.get())};
};
static constexpr lbt::unit::Pressure equationOfState(lbt::unit::Density const rho, 
lbt::unit::Temperature const t) noexcept {
return lbt::unit::Pressure{specific_gas_constant*rho.get()*t.get()};
};


static constexpr lbt::unit::KinematicViscosity kinematicViscosity(lbt::unit::Temperature const t, 
lbt::unit::Pressure const p) noexcept {
return dynamicViscosity(t)/equationOfState(t, p);
}
static constexpr lbt::unit::KinematicViscosity kinematicViscosity(lbt::unit::Density const rho, 
lbt::unit::Temperature const t) noexcept {
return dynamicViscosity(t)/rho;
}


static constexpr lbt::unit::DynamicViscosity dynamicViscosity(lbt::unit::Temperature const t) noexcept {
return T::mu_0*((T::t_0 + T::c)/(t + T::c)*cem::pow(t/T::t_0, 3.0L/2.0L));
}

protected:
IdealGas() = default;
IdealGas(IdealGas const&) = default;
IdealGas& operator= (IdealGas const&) = default;
IdealGas(IdealGas&&) = default;
IdealGas& operator= (IdealGas&&) = default;

static constexpr long double universal_gas_constant {8.31446261815324L}; 
static constexpr long double avogadro_constant {6.02214076e+23L}; 
static constexpr long double specific_gas_constant {universal_gas_constant/T::molecular_weight.get()}; 
};

}
}

#endif 

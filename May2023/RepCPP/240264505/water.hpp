

#ifndef LBT_MATERIAL_WATER
#define LBT_MATERIAL_WATER
#pragma once

#include "../../constexpr_math/constexpr_math.hpp"
#include "../../unit/literals.hpp"
#include "../../unit/units.hpp"


namespace lbt {
namespace material {


class Water {
public:

static constexpr lbt::unit::Density equationOfState(lbt::unit::Temperature const t,
lbt::unit::Pressure const p) noexcept {
using namespace lbt::literals;
auto const P {p/1.0_bar}; 
auto const T {(t - 273.15_K).get()}; 
auto const lambda {1788.316L + 21.55053L*T - 0.4695911L*cem::ipow(T,2) + 3.096363E-3L*cem::ipow(T,3) - 0.7341182E-5L*cem::ipow(T,4)}; 
auto const P_0 {5918.499L + 58.05267L*T - 1.1253317L*cem::ipow(T,2) + 6.6123869E-3L*cem::ipow(T,3) - 1.4661625E-5L*cem::ipow(T,4)}; 
auto const V_inf {0.6980547L - 0.7435626E-3L*T + 0.3704258E-4L*cem::ipow(T,2) - 0.6315724E-6L*cem::ipow(T,3) +
+ 0.9829576E-8L*cem::ipow(T,4) - 0.1197269E-9L*cem::ipow(T,5) + 0.1005461E-11L*cem::ipow(T,6) +
- 0.5437898E-14L*cem::ipow(T,7) + 0.169946E-16L*cem::ipow(T,8) - 0.2295063E-19L*cem::ipow(T,9)}; 
auto const V {V_inf + lambda/(P_0 + P)}; 
return (1.0_g/1.0_cm3)/V; 
}



static constexpr lbt::unit::KinematicViscosity kinematicViscosity(lbt::unit::Density const rho, 
lbt::unit::Temperature const t) noexcept {
return dynamicViscosity(t)/rho;
}


static constexpr lbt::unit::DynamicViscosity dynamicViscosity(lbt::unit::Temperature const t) noexcept {
using namespace lbt::literals;
constexpr auto a {0.02939_mPas};
constexpr auto b {507.88_K};
constexpr auto c {149.3_K};
return lbt::cem::exp(b/(t-c))*a;
}

protected:
Water() = default;
Water(Water const&) = default;
Water& operator= (Water const&) = default;
Water(Water&&) = default;
Water& operator= (Water&&) = default;
};

}
}

#endif 

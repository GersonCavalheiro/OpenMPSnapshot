

#ifndef LBT_UNIT_OPERATORS
#define LBT_UNIT_OPERATORS
#pragma once

#include "amount_of_substance.hpp"
#include "area.hpp"
#include "density.hpp"
#include "dynamic_viscosity.hpp"
#include "kinematic_viscosity.hpp"
#include "length.hpp"
#include "mass.hpp"
#include "molar_mass.hpp"
#include "pressure.hpp"
#include "time.hpp"
#include "unit_base.hpp"
#include "velocity.hpp"
#include "volume.hpp"


namespace lbt {
namespace unit {


constexpr Area operator* (Length const& a, Length const& b) noexcept {
return Area{a.get()*b.get()};
}


constexpr Volume operator* (Length const& l, Area const& a) noexcept {
return Volume{a.get()*l.get()};
}


constexpr Volume operator* (Area const& a, Length const& l) noexcept {
return Volume{a.get()*l.get()};
}


constexpr Velocity operator/ (Length const& l, Time const& t) noexcept {
return Velocity{l.get()/t.get()};
}


constexpr Density operator/ (Mass const& m, Volume const& v) noexcept {
return Density{m.get()/v.get()};
}


constexpr KinematicViscosity operator/ (Area const& a, Time const& t) noexcept {
return KinematicViscosity{a.get()/t.get()};
}


constexpr KinematicViscosity operator* (Velocity const& v, Length const& l) noexcept {
return KinematicViscosity{v.get()*l.get()};
}
constexpr KinematicViscosity operator* (Length const& l, Velocity const& v) noexcept {
return KinematicViscosity{v.get()*l.get()};
}


constexpr KinematicViscosity operator/ (DynamicViscosity const& mu, Density const& rho) noexcept {
return KinematicViscosity{mu.get()/rho.get()};
}


constexpr DynamicViscosity operator* (Pressure const& p, Time const& t) noexcept {
return DynamicViscosity{p.get()*t.get()};
}
constexpr DynamicViscosity operator* (Time const& t, Pressure const& p) noexcept {
return DynamicViscosity{p.get()*t.get()};
}


constexpr DynamicViscosity operator* (KinematicViscosity const& nu, Density const& rho) noexcept {
return DynamicViscosity{nu.get()*rho.get()};
}
constexpr DynamicViscosity operator* (Density const& rho, KinematicViscosity const& nu) noexcept {
return DynamicViscosity{nu.get()*rho.get()};
}


constexpr MolarMass operator/ (Mass const& m, AmountOfSubstance const& a) noexcept {
return MolarMass{m.get()/a.get()};
}

}
}

#endif 

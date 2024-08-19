

#ifndef LBT_UNIT_REYNOLDS_NUMBER
#define LBT_UNIT_REYNOLDS_NUMBER
#pragma once

#include "length.hpp"
#include "kinematic_viscosity.hpp"
#include "velocity.hpp"


namespace lbt {
namespace unit {


class ReynoldsNumber {
public:

static constexpr auto compute(lbt::unit::Velocity const& velocity, lbt::unit::Length const& length, 
lbt::unit::KinematicViscosity const& kinematic_viscosity) noexcept {
return velocity.get()*length.get()/kinematic_viscosity.get();
}


constexpr ReynoldsNumber(long double const value) noexcept
: value{value} {
return;
}
ReynoldsNumber() = delete;
ReynoldsNumber(ReynoldsNumber const&) = default;
ReynoldsNumber& operator= (ReynoldsNumber const&) = default;
ReynoldsNumber(ReynoldsNumber&&) = default;
ReynoldsNumber& operator= (ReynoldsNumber&&) = default;


constexpr ReynoldsNumber(lbt::unit::Velocity const& velocity, lbt::unit::Length const& length, 
lbt::unit::KinematicViscosity const& kinematic_viscosity) noexcept
: value{compute(velocity, length, kinematic_viscosity)} {
return;
}


constexpr auto get() const noexcept {
return value;
}


constexpr operator long double() const noexcept {
return value;
}

protected:
long double value;
};
}
}

#endif 

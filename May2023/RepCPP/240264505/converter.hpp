

#ifndef LBT_CONVERTER
#define LBT_CONVERTER
#pragma once

#include "unit/units.hpp"


namespace lbt {


class Converter {
public:

explicit constexpr Converter(lbt::unit::Length const physical_length, long double const equivalent_lbm_length, 
lbt::unit::Velocity const physical_velocity, long double const equivalent_lbm_velocity,
lbt::unit::Density const physical_density, long double const equivalent_lbm_density) noexcept
: to_lbm_length_{physical_length.get()/equivalent_lbm_length}, 
to_lbm_velocity_{physical_velocity.get()/equivalent_lbm_velocity}, 
to_lbm_density_{physical_density.get()/equivalent_lbm_density},
to_lbm_time_{to_lbm_length_/to_lbm_velocity_},
to_lbm_kinematic_viscosity_{to_lbm_velocity_*to_lbm_length_},
to_lbm_pressure_{to_lbm_density_*to_lbm_velocity_*to_lbm_velocity_} {
return;
}
Converter() = delete;
Converter(Converter const&) = default;
Converter& operator= (Converter const&) = default;
Converter(Converter&&) = default;
Converter& operator= (Converter&&) = default;


template <typename T>
constexpr T toPhysical(long double const lbm_unit) const noexcept = delete;


template <typename T>
constexpr long double toLbm(T const physical_unit) const noexcept = delete;

protected:
long double to_lbm_length_;
long double to_lbm_velocity_;
long double to_lbm_density_;
long double to_lbm_time_;
long double to_lbm_kinematic_viscosity_;
long double to_lbm_pressure_;
};

template <>
constexpr lbt::unit::Length Converter::toPhysical(long double const length) const noexcept {
return lbt::unit::Length{to_lbm_length_*length};
}
template <>
constexpr long double Converter::toLbm(lbt::unit::Length const length) const noexcept {
return length.get()/to_lbm_length_;
}

template <>
constexpr lbt::unit::Velocity Converter::toPhysical(long double const velocity) const noexcept {
return lbt::unit::Velocity{to_lbm_velocity_*velocity};
}
template <>
constexpr long double Converter::toLbm(lbt::unit::Velocity const velocity) const noexcept {
return velocity.get()/to_lbm_velocity_;
}

template <>
constexpr lbt::unit::Density Converter::toPhysical(long double const density) const noexcept {
return lbt::unit::Density{to_lbm_density_*density};
}
template <>
constexpr long double Converter::toLbm(lbt::unit::Density const density) const noexcept {
return density.get()/to_lbm_density_;
}

template <>
constexpr lbt::unit::Time Converter::toPhysical(long double const time) const noexcept {
return lbt::unit::Time{to_lbm_time_*time};
}
template <>
constexpr long double Converter::toLbm(lbt::unit::Time const time) const noexcept {
return time.get()/to_lbm_time_;
}

template <>
constexpr lbt::unit::KinematicViscosity Converter::toPhysical(long double const kinematic_viscosity) const noexcept {
return lbt::unit::KinematicViscosity{to_lbm_kinematic_viscosity_*kinematic_viscosity};
}
template <>
constexpr long double Converter::toLbm(lbt::unit::KinematicViscosity const kinematic_viscosity) const noexcept {
return kinematic_viscosity.get()/to_lbm_kinematic_viscosity_;
}

template <>
constexpr lbt::unit::Pressure Converter::toPhysical(long double const pressure) const noexcept {
return lbt::unit::Pressure{to_lbm_pressure_*pressure};
}
template <>
constexpr long double Converter::toLbm(lbt::unit::Pressure const pressure) const noexcept {
return pressure.get()/to_lbm_pressure_;
}

}

#endif 


#ifndef LBT_UNIT_KINEMATIC_VISCOSITY
#define LBT_UNIT_KINEMATIC_VISCOSITY
#pragma once

#include "unit_base.hpp"


namespace lbt {
namespace unit {


class KinematicViscosity : public lbt::unit::detail::UnitBase<KinematicViscosity> {
public:

explicit constexpr KinematicViscosity(long double const value = 0.0) noexcept
: UnitBase{value} {
return;
}
KinematicViscosity(KinematicViscosity const&) = default;
KinematicViscosity& operator= (KinematicViscosity const&) = default;
KinematicViscosity(KinematicViscosity&&) = default;
KinematicViscosity& operator= (KinematicViscosity&&) = default;
};

}
}

#endif 

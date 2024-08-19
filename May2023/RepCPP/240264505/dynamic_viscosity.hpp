

#ifndef LBT_UNIT_DYNAMIC_VISCOSITY
#define LBT_UNIT_DYNAMIC_VISCOSITY
#pragma once

#include "unit_base.hpp"


namespace lbt {
namespace unit {



class DynamicViscosity : public lbt::unit::detail::UnitBase<DynamicViscosity> {
public:

explicit constexpr DynamicViscosity(long double const value = 0.0) noexcept
: UnitBase{value} {
return;
}
DynamicViscosity(DynamicViscosity const&) = default;
DynamicViscosity& operator= (DynamicViscosity const&) = default;
DynamicViscosity(DynamicViscosity&&) = default;
DynamicViscosity& operator= (DynamicViscosity&&) = default;
};

}
}

#endif 

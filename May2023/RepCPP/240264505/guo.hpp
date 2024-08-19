

#ifndef LBT_BOUNDARY_GUO
#define LBT_BOUNDARY_GUO
#pragma once

#include <cstdint>

#include "../../general/type_definitions.hpp"
#include "normal.hpp"
#include "orientation.hpp"
#include "type.hpp"


namespace lbt {
namespace boundary {


template <std::int32_t TN, typename A, typename B>
static constexpr auto getComponent(A const tangential_value, B const normal_value) noexcept {
if constexpr (TN == 0) {
return tangential_value;
} else {
return normal_value;
}
}


template <Orientation O, Type TP>
class MacroscopicValues {
public:

template <typename T>
static constexpr lbt::StackArray<T,4> get(lbt::StackArray<T,4> const& boundary_values, lbt::StackArray<T,4> const& interpolated_values) noexcept;
};


template <Orientation O>
class MacroscopicValues<O,Type::Velocity> {
public:
template <typename T>
static constexpr lbt::StackArray<T,4> get(lbt::StackArray<T,4> const& boundary_values, lbt::StackArray<T,4> const& interpolated_values) noexcept {
return { interpolated_values[0], boundary_values[1], boundary_values[2], boundary_values[3] };
}
};


template <Orientation O>
class MacroscopicValues<O,Type::Pressure> {
public:
template <typename T>
static constexpr lbt::StackArray<T,4> get(lbt::StackArray<T,4> const& boundary_values, lbt::StackArray<T,4> const& interpolated_values) noexcept {
return { boundary_values[0],
getComponent<Normal<O>::x>(boundary_values[1], interpolated_values[1]),
getComponent<Normal<O>::y>(boundary_values[2], interpolated_values[2]),
getComponent<Normal<O>::z>(boundary_values[3], interpolated_values[3]) };
}
};

}
}

#endif 

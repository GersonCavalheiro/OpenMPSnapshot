

#ifndef LBT_CEM_MATHEMATICAL_CONSTANTS
#define LBT_CEM_MATHEMATICAL_CONSTANTS
#pragma once

#include <type_traits>


namespace lbt {
namespace cem {

template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
inline constexpr T pi = static_cast<T>(3.1415926535897932385L);

template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
inline constexpr T e = static_cast<T>(2.71828182845904523536L);

template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
inline constexpr T ln2 = static_cast<T>(0.69314718055994530942L);

}
}

#endif 

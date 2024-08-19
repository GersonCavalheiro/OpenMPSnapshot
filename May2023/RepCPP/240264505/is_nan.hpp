

#ifndef LBT_CEM_IS_NAN
#define LBT_CEM_IS_NAN
#pragma once

#include <type_traits>


namespace lbt {
namespace cem {


template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
constexpr bool isNan(T const x) noexcept {
return (x != x);
}

}
}

#endif 

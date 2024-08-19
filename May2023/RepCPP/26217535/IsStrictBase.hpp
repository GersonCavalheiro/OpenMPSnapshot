

#pragma once

#include <type_traits>

namespace alpaka::meta
{
template<typename TBase, typename TDerived>
using IsStrictBase = std::
integral_constant<bool, std::is_base_of_v<TBase, TDerived> && !std::is_same_v<TBase, std::decay_t<TDerived>>>;
} 



#pragma once

#include <alpaka/core/BoostPredef.hpp>

#include <type_traits>

#if BOOST_COMP_PGI
#    define ALPAKA_DECAY_T(Type) typename std::decay<Type>::type
#else
#    define ALPAKA_DECAY_T(Type) std::decay_t<Type>
#endif

namespace alpaka
{
template<typename T, typename U>
inline constexpr auto is_decayed_v = std::is_same_v<ALPAKA_DECAY_T(T), ALPAKA_DECAY_T(U)>;
} 

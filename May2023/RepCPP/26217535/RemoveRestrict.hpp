

#pragma once

#include <alpaka/core/BoostPredef.hpp>

namespace alpaka
{
template<typename T>
struct remove_restrict
{
using type = T;
};

#if BOOST_COMP_MSVC
template<typename T>
struct remove_restrict<T* __restrict>
{
using type = T*;
};
#else
template<typename T>
struct remove_restrict<T* __restrict__>
{
using type = T*;
};
#endif

template<typename T>
using remove_restrict_t = typename remove_restrict<T>::type;
} 

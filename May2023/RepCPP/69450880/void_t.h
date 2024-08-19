



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2017
#  include <type_traits>
#endif

HYDRA_THRUST_BEGIN_NS

#if HYDRA_THRUST_CPP_DIALECT >= 2011

template <typename...> struct voider { using type = void; };

#if HYDRA_THRUST_CPP_DIALECT >= 2017
using std::void_t;
#else
template <typename... Ts> using void_t = typename voider<Ts...>::type;
#endif

#else 

template <
typename = void
, typename = void
, typename = void
, typename = void
, typename = void
, typename = void
, typename = void
, typename = void
, typename = void
, typename = void
>
struct voider
{
typedef void type;
};

#endif

HYDRA_THRUST_END_NS


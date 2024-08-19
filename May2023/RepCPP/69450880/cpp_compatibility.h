

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config/cpp_dialect.h>

#include <cstddef>

#if HYDRA_THRUST_CPP_DIALECT >= 2011
#  ifndef __has_cpp_attribute
#    define __has_cpp_attribute(X) 0
#  endif

#  if __has_cpp_attribute(nodiscard)
#    define HYDRA_THRUST_NODISCARD [[nodiscard]]
#  endif

#  define HYDRA_THRUST_CONSTEXPR constexpr
#  define HYDRA_THRUST_OVERRIDE override
#  define HYDRA_THRUST_DEFAULT = default;
#  define HYDRA_THRUST_NOEXCEPT noexcept
#  define HYDRA_THRUST_FINAL final
#else
#  define HYDRA_THRUST_CONSTEXPR
#  define HYDRA_THRUST_OVERRIDE
#  define HYDRA_THRUST_DEFAULT {}
#  define HYDRA_THRUST_NOEXCEPT throw()
#  define HYDRA_THRUST_FINAL
#endif

#ifndef HYDRA_THRUST_NODISCARD
#  define HYDRA_THRUST_NODISCARD
#endif

#ifdef __CUDA_ARCH__
#  if HYDRA_THRUST_CPP_DIALECT >= 2011
#    define HYDRA_THRUST_INLINE_CONSTANT                 static constexpr
#    define HYDRA_THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT static constexpr
#  else
#    define HYDRA_THRUST_INLINE_CONSTANT                 static const __device__
#    define HYDRA_THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT static const
#  endif
#else
#  if HYDRA_THRUST_CPP_DIALECT >= 2011
#    define HYDRA_THRUST_INLINE_CONSTANT                 static constexpr
#    define HYDRA_THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT static constexpr
#  else
#    define HYDRA_THRUST_INLINE_CONSTANT                 static const
#    define HYDRA_THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT static const
#  endif
#endif


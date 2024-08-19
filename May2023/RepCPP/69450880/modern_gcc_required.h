

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config/cpp_dialect.h>

#ifndef HYDRA_THRUST_MODERN_GCC_REQUIRED_NO_ERROR
#  if defined(HYDRA_THRUST_GCC_VERSION) && !defined(HYDRA_THRUST_MODERN_GCC)
#    error GCC 5 or later is required for this Thrust feature; please upgrade your compiler.
#  endif
#endif


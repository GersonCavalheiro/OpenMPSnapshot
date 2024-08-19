

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config/compiler.h>

#if defined(HYDRA_THRUST_GCC_VERSION) && (HYDRA_THRUST_GCC_VERSION >= 40800)
#  if defined(__NVCC__) && (CUDART_VERSION >= 6000)
#    pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#  endif 
#endif 


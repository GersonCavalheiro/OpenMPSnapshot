



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

#if defined(__CUDACC__)

#define __hydra_thrust_forceinline__ __forceinline__

#else


#define __hydra_thrust_forceinline__

#endif


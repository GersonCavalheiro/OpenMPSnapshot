

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/preprocessor.h>


#if HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC

#ifndef _DEBUG

#include <intrin.h>
#pragma intrinsic(_ReadWriteBarrier)
#define __hydra_thrust_compiler_fence() _ReadWriteBarrier()
#else

#define __hydra_thrust_compiler_fence() do {} while (0)

#endif 

#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC

#if HYDRA_THRUST_GCC_VERSION >= 40200 
#define __hydra_thrust_compiler_fence() __sync_synchronize()
#else
#define __hydra_thrust_compiler_fence() do {} while (0)
#endif 

#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_CLANG
#define __hydra_thrust_compiler_fence() __sync_synchronize()
#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_UNKNOWN

#define __hydra_thrust_compiler_fence() do {} while (0)

#endif


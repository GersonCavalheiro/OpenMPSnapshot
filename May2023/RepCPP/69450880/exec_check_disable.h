



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

#if defined(__CUDACC__) && !(defined(__CUDA__) && defined(__clang__))

#define __hydra_thrust_exec_check_disable__ #pragma nv_exec_check_disable

#else

#define __hydra_thrust_exec_check_disable__

#endif



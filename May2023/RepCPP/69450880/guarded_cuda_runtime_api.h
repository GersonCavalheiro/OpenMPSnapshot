

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>


#if !defined(__HOST_DEFINES_H__)

#ifdef __host__
#undef __host__
#endif 

#ifdef __device__
#undef __device__
#endif 

#endif 

#include <cuda_runtime_api.h>


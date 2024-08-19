



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>


#if HYDRA_THRUST_DEVICE_COMPILER != HYDRA_THRUST_DEVICE_COMPILER_NVCC


#ifndef __host__
#define __host__
#endif 

#ifndef __device__
#define __device__
#endif 

#endif


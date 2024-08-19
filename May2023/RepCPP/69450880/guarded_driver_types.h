

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>



#if !defined(__GNUC__) || ((10000 * __GNUC__ + 100 * __GNUC_MINOR__ + __GNUC_PATCHLEVEL__) >= 40500)
#  ifdef __host__
#    pragma push_macro("__host__")
#    undef __host__
#    define HYDRA_THRUST_HOST_NEEDS_RESTORATION
#  endif
#  ifdef __device__
#    pragma push_macro("__device__")
#    undef __device__
#    define HYDRA_THRUST_DEVICE_NEEDS_RESTORATION
#  endif
#else 
#  if !defined(__DRIVER_TYPES_H__)
#    ifdef __host__
#      undef __host__
#    endif
#    ifdef __device__
#      undef __device__
#    endif
#  endif 
#endif 


#include <driver_types.h>


#if !defined(__GNUC__) || ((10000 * __GNUC__ + 100 * __GNUC_MINOR__ + __GNUC_PATCHLEVEL__) >= 40500)
#  ifdef HYDRA_THRUST_HOST_NEEDS_RESTORATION
#    pragma pop_macro("__host__")
#    undef HYDRA_THRUST_HOST_NEEDS_RESTORATION
#  endif
#  ifdef HYDRA_THRUST_DEVICE_NEEDS_RESTORATION
#    pragma pop_macro("__device__")
#    undef HYDRA_THRUST_DEVICE_NEEDS_RESTORATION
#  endif
#endif 


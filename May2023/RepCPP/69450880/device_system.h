

#pragma once

#define HYDRA_THRUST_DEVICE_SYSTEM_CUDA    1
#define HYDRA_THRUST_DEVICE_SYSTEM_OMP     2
#define HYDRA_THRUST_DEVICE_SYSTEM_TBB     3
#define HYDRA_THRUST_DEVICE_SYSTEM_CPP     4

#ifndef HYDRA_THRUST_DEVICE_SYSTEM
#define HYDRA_THRUST_DEVICE_SYSTEM HYDRA_THRUST_DEVICE_SYSTEM_CUDA
#endif 


#define HYDRA_THRUST_DEVICE_BACKEND_CUDA HYDRA_THRUST_DEVICE_SYSTEM_CUDA
#define HYDRA_THRUST_DEVICE_BACKEND_OMP  HYDRA_THRUST_DEVICE_SYSTEM_OMP
#define HYDRA_THRUST_DEVICE_BACKEND_TBB  HYDRA_THRUST_DEVICE_SYSTEM_TBB

#ifdef HYDRA_THRUST_DEVICE_BACKEND
#  if HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC
#    pragma message("----------------------------------------------------------------------------------")
#    pragma message("| WARNING: HYDRA_THRUST_DEVICE_BACKEND is deprecated; use HYDRA_THRUST_DEVICE_SYSTEM instead |")
#    pragma message("----------------------------------------------------------------------------------")
#  else
#    warning ----------------------------------------------------------------------------------
#    warning | WARNING: HYDRA_THRUST_DEVICE_BACKEND is deprecated; use HYDRA_THRUST_DEVICE_SYSTEM instead |
#    warning ----------------------------------------------------------------------------------
#  endif 
#  undef HYDRA_THRUST_DEVICE_SYSTEM
#  define HYDRA_THRUST_DEVICE_SYSTEM HYDRA_THRUST_DEVICE_BACKEND
#endif 

#if HYDRA_THRUST_DEVICE_SYSTEM == HYDRA_THRUST_DEVICE_SYSTEM_CUDA
#define __HYDRA_THRUST_DEVICE_SYSTEM_NAMESPACE cuda
#elif HYDRA_THRUST_DEVICE_SYSTEM == HYDRA_THRUST_DEVICE_SYSTEM_OMP
#define __HYDRA_THRUST_DEVICE_SYSTEM_NAMESPACE omp
#elif HYDRA_THRUST_DEVICE_SYSTEM == HYDRA_THRUST_DEVICE_SYSTEM_TBB
#define __HYDRA_THRUST_DEVICE_SYSTEM_NAMESPACE tbb
#elif HYDRA_THRUST_DEVICE_SYSTEM == HYDRA_THRUST_DEVICE_SYSTEM_CPP
#define __HYDRA_THRUST_DEVICE_SYSTEM_NAMESPACE cpp
#endif

#define __HYDRA_THRUST_DEVICE_SYSTEM_ROOT hydra/detail/external/hydra_thrust/system/__HYDRA_THRUST_DEVICE_SYSTEM_NAMESPACE


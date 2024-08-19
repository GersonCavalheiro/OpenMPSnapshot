

#pragma once

#define HYDRA_THRUST_HOST_SYSTEM_CPP    1
#define HYDRA_THRUST_HOST_SYSTEM_OMP    2
#define HYDRA_THRUST_HOST_SYSTEM_TBB    3

#ifndef HYDRA_THRUST_HOST_SYSTEM
#define HYDRA_THRUST_HOST_SYSTEM HYDRA_THRUST_HOST_SYSTEM_CPP
#endif 


#define HYDRA_THRUST_HOST_BACKEND_CPP HYDRA_THRUST_HOST_SYSTEM_CPP
#define HYDRA_THRUST_HOST_BACKEND_OMP HYDRA_THRUST_HOST_SYSTEM_OMP
#define HYDRA_THRUST_HOST_BACKEND_TBB HYDRA_THRUST_HOST_SYSTEM_TBB

#ifdef HYDRA_THRUST_HOST_BACKEND
#  if HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC
#    pragma message("------------------------------------------------------------------------------")
#    pragma message("| WARNING: HYDRA_THRUST_HOST_BACKEND is deprecated; use HYDRA_THRUST_HOST_SYSTEM instead |")
#    pragma message("------------------------------------------------------------------------------")
#  else
#    warning ------------------------------------------------------------------------------
#    warning | WARNING: HYDRA_THRUST_HOST_BACKEND is deprecated; use HYDRA_THRUST_HOST_SYSTEM instead |
#    warning ------------------------------------------------------------------------------
#  endif 
#  undef HYDRA_THRUST_HOST_SYSTEM
#  define HYDRA_THRUST_HOST_SYSTEM HYDRA_THRUST_HOST_BACKEND
#endif 

#if HYDRA_THRUST_HOST_SYSTEM == HYDRA_THRUST_HOST_SYSTEM_CPP
#define __HYDRA_THRUST_HOST_SYSTEM_NAMESPACE cpp
#elif HYDRA_THRUST_HOST_SYSTEM == HYDRA_THRUST_HOST_SYSTEM_OMP
#define __HYDRA_THRUST_HOST_SYSTEM_NAMESPACE omp
#elif HYDRA_THRUST_HOST_SYSTEM == HYDRA_THRUST_HOST_SYSTEM_TBB
#define __HYDRA_THRUST_HOST_SYSTEM_NAMESPACE tbb
#endif

#define __HYDRA_THRUST_HOST_SYSTEM_ROOT hydra/detail/external/hydra_thrust/system/__HYDRA_THRUST_HOST_SYSTEM_NAMESPACE


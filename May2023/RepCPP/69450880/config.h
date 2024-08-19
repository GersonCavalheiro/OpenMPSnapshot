
#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

#define HYDRA_THRUST_UNUSED_VAR(expr) do { (void)(expr); } while (0)

#if defined(__CUDACC__)
#  if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__>= 350 && defined(__CUDACC_RDC__))
#    define __HYDRA_THRUST_HAS_CUDART__ 1
#    define HYDRA_THRUST_RUNTIME_FUNCTION __host__ __device__ __forceinline__
#  else
#    define __HYDRA_THRUST_HAS_CUDART__ 0
#    define HYDRA_THRUST_RUNTIME_FUNCTION __host__ __forceinline__
#  endif
#else
#  define __HYDRA_THRUST_HAS_CUDART__ 0
#  define HYDRA_THRUST_RUNTIME_FUNCTION __host__ __forceinline__
#endif

#ifdef __CUDA_ARCH__
#define HYDRA_THRUST_DEVICE_CODE
#endif

#ifdef HYDRA_THRUST_AGENT_ENTRY_NOINLINE
#define HYDRA_THRUST_AGENT_ENTRY_INLINE_ATTR __noinline__
#else
#define HYDRA_THRUST_AGENT_ENTRY_INLINE_ATTR __forceinline__
#endif

#define HYDRA_THRUST_DEVICE_FUNCTION __device__ __forceinline__
#define HYDRA_THRUST_HOST_FUNCTION __host__     __forceinline__
#define HYDRA_THRUST_FUNCTION __host__ __device__ __forceinline__
#if 0
#define HYDRA_THRUST_ARGS(...) __VA_ARGS__
#define HYDRA_THRUST_STRIP_PARENS(X) X
#define HYDRA_THRUST_AGENT_ENTRY(ARGS) HYDRA_THRUST_FUNCTION static void entry(HYDRA_THRUST_STRIP_PARENS(HYDRA_THRUST_ARGS ARGS))
#else
#define HYDRA_THRUST_AGENT_ENTRY(...) HYDRA_THRUST_AGENT_ENTRY_INLINE_ATTR __device__ static void entry(__VA_ARGS__)
#endif

#ifdef HYDRA_THRUST_DEBUG_SYNC
#define HYDRA_THRUST_DEBUG_SYNC_FLAG true
#else
#define HYDRA_THRUST_DEBUG_SYNC_FLAG false
#endif

#define HYDRA_THRUST_CUB_NS_PREFIX namespace hydra_thrust {   namespace cuda_cub {
#define HYDRA_THRUST_CUB_NS_POSTFIX }  }


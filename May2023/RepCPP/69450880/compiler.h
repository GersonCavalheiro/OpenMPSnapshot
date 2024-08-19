



#pragma once

#define HYDRA_THRUST_HOST_COMPILER_UNKNOWN 0
#define HYDRA_THRUST_HOST_COMPILER_MSVC    1
#define HYDRA_THRUST_HOST_COMPILER_GCC     2
#define HYDRA_THRUST_HOST_COMPILER_CLANG   3

#define HYDRA_THRUST_DEVICE_COMPILER_UNKNOWN 0
#define HYDRA_THRUST_DEVICE_COMPILER_MSVC    1
#define HYDRA_THRUST_DEVICE_COMPILER_GCC     2
#define HYDRA_THRUST_DEVICE_COMPILER_NVCC    3
#define HYDRA_THRUST_DEVICE_COMPILER_CLANG   4

#if   defined(_MSC_VER)
#define HYDRA_THRUST_HOST_COMPILER HYDRA_THRUST_HOST_COMPILER_MSVC
#elif defined(__clang__)
#define HYDRA_THRUST_HOST_COMPILER HYDRA_THRUST_HOST_COMPILER_CLANG
#define HYDRA_THRUST_CLANG_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#elif defined(__GNUC__)
#define HYDRA_THRUST_HOST_COMPILER HYDRA_THRUST_HOST_COMPILER_GCC
#define HYDRA_THRUST_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#if (HYDRA_THRUST_GCC_VERSION >= 50000)
#define HYDRA_THRUST_MODERN_GCC
#else
#define HYDRA_THRUST_LEGACY_GCC
#endif
#else
#define HYDRA_THRUST_HOST_COMPILER HYDRA_THRUST_HOST_COMPILER_UNKNOWN
#endif 

#if defined(__CUDACC__)
#define HYDRA_THRUST_DEVICE_COMPILER HYDRA_THRUST_DEVICE_COMPILER_NVCC
#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC
#define HYDRA_THRUST_DEVICE_COMPILER HYDRA_THRUST_DEVICE_COMPILER_MSVC
#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC
#define HYDRA_THRUST_DEVICE_COMPILER HYDRA_THRUST_DEVICE_COMPILER_GCC
#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_CLANG
#if defined(__CUDA__)
#define HYDRA_THRUST_DEVICE_COMPILER HYDRA_THRUST_DEVICE_COMPILER_NVCC
#else
#define HYDRA_THRUST_DEVICE_COMPILER HYDRA_THRUST_DEVICE_COMPILER_CLANG
#endif
#else
#define HYDRA_THRUST_DEVICE_COMPILER HYDRA_THRUST_DEVICE_COMPILER_UNKNOWN
#endif

#ifdef _OPENMP
#define HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE HYDRA_THRUST_TRUE
#else
#define HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE HYDRA_THRUST_FALSE
#endif 


#if (HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC) && !defined(__CUDA_ARCH__)
#define HYDRA_THRUST_DISABLE_MSVC_WARNING_BEGIN(x)                                \
__pragma(warning(push))                                                   \
__pragma(warning(disable : x))                                            \

#define HYDRA_THRUST_DISABLE_MSVC_WARNING_END(x)                                  \
__pragma(warning(pop))                                                    \

#else
#define HYDRA_THRUST_DISABLE_MSVC_WARNING_BEGIN(x)
#define HYDRA_THRUST_DISABLE_MSVC_WARNING_END(x)
#endif

#if (HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_CLANG) && !defined(__CUDA_ARCH__)
#define HYDRA_THRUST_IGNORE_CLANG_WARNING_IMPL(x)                                 \
HYDRA_THRUST_PP_STRINGIZE(clang diagnostic ignored x)                           \

#define HYDRA_THRUST_IGNORE_CLANG_WARNING(x)                                      \
HYDRA_THRUST_IGNORE_CLANG_WARNING_IMPL(HYDRA_THRUST_PP_STRINGIZE(x))                  \


#define HYDRA_THRUST_DISABLE_CLANG_WARNING_BEGIN(x)                               \
_Pragma("clang diagnostic push")                                          \
_Pragma(HYDRA_THRUST_IGNORE_CLANG_WARNING(x))                                   \

#define HYDRA_THRUST_DISABLE_CLANG_WARNING_END(x)                                 \
_Pragma("clang diagnostic pop")                                           \

#else
#define HYDRA_THRUST_DISABLE_CLANG_WARNING_BEGIN(x)
#define HYDRA_THRUST_DISABLE_CLANG_WARNING_END(x)
#endif

#if (HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC) && !defined(__CUDA_ARCH__)
#define HYDRA_THRUST_IGNORE_GCC_WARNING_IMPL(x)                                   \
HYDRA_THRUST_PP_STRINGIZE(GCC diagnostic ignored x)                             \

#define HYDRA_THRUST_IGNORE_GCC_WARNING(x)                                        \
HYDRA_THRUST_IGNORE_GCC_WARNING_IMPL(HYDRA_THRUST_PP_STRINGIZE(x))                    \


#define HYDRA_THRUST_DISABLE_GCC_WARNING_BEGIN(x)                                 \
_Pragma("GCC diagnostic push")                                            \
_Pragma(HYDRA_THRUST_IGNORE_GCC_WARNING(x))                                     \

#define HYDRA_THRUST_DISABLE_GCC_WARNING_END(x)                                   \
_Pragma("GCC diagnostic pop")                                             \

#else
#define HYDRA_THRUST_DISABLE_GCC_WARNING_BEGIN(x)
#define HYDRA_THRUST_DISABLE_GCC_WARNING_END(x)
#endif

#define HYDRA_THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN               \
HYDRA_THRUST_DISABLE_MSVC_WARNING_BEGIN(4244 4267)                                \

#define HYDRA_THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END                 \
HYDRA_THRUST_DISABLE_MSVC_WARNING_END(4244 4267)                                  \

#define HYDRA_THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING(x)                  \
HYDRA_THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN                     \
x;                                                                          \
HYDRA_THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END                       \


#define HYDRA_THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING_BEGIN               \
HYDRA_THRUST_DISABLE_MSVC_WARNING_BEGIN(4800)                                     \

#define HYDRA_THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING_END                 \
HYDRA_THRUST_DISABLE_MSVC_WARNING_END(4800)                                       \

#define HYDRA_THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING(x)                  \
HYDRA_THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING_BEGIN                     \
x;                                                                          \
HYDRA_THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING_END                       \


#define HYDRA_THRUST_DISABLE_CLANG_SELF_ASSIGNMENT_WARNING_BEGIN                    \
HYDRA_THRUST_DISABLE_CLANG_WARNING_BEGIN(-Wself-assign)                           \

#define HYDRA_THRUST_DISABLE_CLANG_SELF_ASSIGNMENT_WARNING_END                      \
HYDRA_THRUST_DISABLE_CLANG_WARNING_END(-Wself-assign)                             \

#define HYDRA_THRUST_DISABLE_CLANG_SELF_ASSIGNMENT_WARNING(x)                       \
HYDRA_THRUST_DISABLE_CLANG_SELF_ASSIGNMENT_WARNING_BEGIN                          \
x;                                                                          \
HYDRA_THRUST_DISABLE_CLANG_SELF_ASSIGNMENT_WARNING_END                            \


#define HYDRA_THRUST_DISABLE_CLANG_AND_GCC_INITIALIZER_REORDERING_WARNING_BEGIN     \
HYDRA_THRUST_DISABLE_CLANG_WARNING_BEGIN(-Wreorder)                               \
HYDRA_THRUST_DISABLE_GCC_WARNING_BEGIN(-Wreorder)                                 \

#define HYDRA_THRUST_DISABLE_CLANG_AND_GCC_INITIALIZER_REORDERING_WARNING_END       \
HYDRA_THRUST_DISABLE_CLANG_WARNING_END(-Wreorder)                                 \
HYDRA_THRUST_DISABLE_GCC_WARNING_END(-Wreorder)                                   \

#define HYDRA_THRUST_DISABLE_CLANG_AND_GCC_INITIALIZER_REORDERING_WARNING(x)        \
HYDRA_THRUST_DISABLE_CLANG_AND_GCC_INITIALIZER_REORDERING_WARNING_BEGIN           \
x;                                                                          \
HYDRA_THRUST_DISABLE_CLANG_AND_GCC_INITIALIZER_REORDERING_WARNING_END             \


#if   HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC
#define HYDRA_THRUST_DEPRECATED __declspec(deprecated)
#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_CLANG
#define HYDRA_THRUST_DEPRECATED __attribute__((deprecated))
#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC
#define HYDRA_THRUST_DEPRECATED __attribute__((deprecated))
#else
#define HYDRA_THRUST_DEPRECATED
#endif


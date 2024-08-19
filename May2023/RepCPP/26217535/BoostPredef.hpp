

#pragma once

#include <boost/predef.h>

#ifdef __INTEL_COMPILER
#    warning                                                                                                          \
"The Intel Classic compiler (icpc) is no longer supported. Please upgrade to the Intel LLVM compiler (ipcx)."
#endif

#if !defined(BOOST_LANG_HIP)
#    if defined(__HIPCC__) && (defined(__CUDACC__) || defined(__HIP__))
#        include <hip/hip_runtime.h>
#        undef abort
#        define BOOST_LANG_HIP BOOST_VERSION_NUMBER(HIP_VERSION_MAJOR, HIP_VERSION_MINOR, 0)
#        if defined(BOOST_LANG_CUDA) && BOOST_LANG_CUDA
#            undef BOOST_LANG_CUDA
#            define BOOST_LANG_CUDA BOOST_VERSION_NUMBER_NOT_AVAILABLE
#        endif
#    else
#        define BOOST_LANG_HIP BOOST_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

#if !defined(BOOST_ARCH_HSA)
#    if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1 && defined(__HIP__)
#        define BOOST_ARCH_HSA BOOST_VERSION_NUMBER_AVAILABLE
#    else
#        define BOOST_ARCH_HSA BOOST_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

#if !defined(BOOST_COMP_HIP)
#    if defined(__HIP__)
#        define BOOST_COMP_HIP BOOST_VERSION_NUMBER_AVAILABLE
#    else
#        define BOOST_COMP_HIP BOOST_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

#if defined(__clang__) && defined(__CUDA__)
#    define BOOST_COMP_CLANG_CUDA BOOST_COMP_CLANG
#else
#    define BOOST_COMP_CLANG_CUDA BOOST_VERSION_NUMBER_NOT_AVAILABLE
#endif

#if defined(BOOST_COMP_PGI) && defined(BOOST_COMP_PGI_EMULATED)
#    undef BOOST_COMP_PGI
#    define BOOST_COMP_PGI BOOST_COMP_PGI_EMULATED
#endif

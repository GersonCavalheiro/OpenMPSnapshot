

#pragma once

#include <alpaka/core/BoostPredef.hpp>

#if(BOOST_COMP_NVCC && BOOST_ARCH_PTX)
#    if BOOST_LANG_CUDA >= BOOST_VERSION_NUMBER(11, 3, 0)
#        define ALPAKA_UNREACHABLE(...) __builtin_unreachable()
#    else
#        define ALPAKA_UNREACHABLE(...) return __VA_ARGS__
#    endif
#elif BOOST_COMP_MSVC
#    define ALPAKA_UNREACHABLE(...) __assume(false)
#elif BOOST_COMP_GNUC || BOOST_COMP_CLANG
#    define ALPAKA_UNREACHABLE(...) __builtin_unreachable()
#else
#    define ALPAKA_UNREACHABLE(...)
#endif

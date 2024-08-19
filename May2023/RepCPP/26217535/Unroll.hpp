

#pragma once

#include <alpaka/core/BoostPredef.hpp>

#if BOOST_ARCH_PTX
#    define ALPAKA_UNROLL_STRINGIFY(x) #    x
#    define ALPAKA_UNROLL(...) _Pragma(ALPAKA_UNROLL_STRINGIFY(unroll __VA_ARGS__))
#elif BOOST_COMP_IBM || BOOST_COMP_SUNPRO || BOOST_COMP_HPACC
#    define ALPAKA_UNROLL_STRINGIFY(x) #    x
#    define ALPAKA_UNROLL(...) _Pragma(ALPAKA_UNROLL_STRINGIFY(unroll(__VA_ARGS__)))
#elif BOOST_COMP_PGI
#    define ALPAKA_UNROLL(...) _Pragma("unroll")
#else
#    define ALPAKA_UNROLL(...)
#endif

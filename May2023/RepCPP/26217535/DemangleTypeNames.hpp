

#pragma once

#include <alpaka/core/BoostPredef.hpp>

#include <boost/core/demangle.hpp>

namespace alpaka::core
{
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wexit-time-destructors"
#    pragma clang diagnostic ignored "-Wmissing-variable-declarations"
#endif
template<typename T>
inline const std::string demangled = boost::core::demangle(typeid(T).name());
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
} 
